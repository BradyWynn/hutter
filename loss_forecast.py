"""Heuristic full-stream payload forecasts; model spread is not a confidence interval."""
import math
import numpy as np

BYTES_PER_NAT = 1 / (8 * math.log(2))


def power_average(left, right, alpha):
    if alpha == 1:
        return np.log(right / left) / (right - left)
    return (right ** (1-alpha) - left ** (1-alpha)) / ((1-alpha)*(right-left))


class LossForecast:
    def __init__(self, total_targets, literal_bytes, dictionary_bytes, interval=256, ema_beta=0.8):
        if interval <= 0:
            raise ValueError('forecast interval must be positive')
        if not 0 <= ema_beta < 1:
            raise ValueError('forecast ema_beta must be in [0, 1)')
        self.total = total_targets
        self.literal_bytes = literal_bytes
        self.dictionary_bytes = dictionary_bytes
        self.interval = interval
        self.ema_beta = ema_beta
        self.ema = None
        self.targets = 0
        self.nats = 0.0
        self.steps = 0
        self.edges = [0]
        self.costs = [0.0]
        # Per-block spread of loss across segments, when the caller supplies per-segment
        # costs. Reported as a diagnostic: it measures how much of the loss variation is
        # file heterogeneity rather than learning, which the trend fit cannot separate.
        self.block_spread = []
        self.seg_nats = None
        self.seg_tokens = 0.0

    def update(self, count, nats, seg_nats=None, force=False):
        if count <= 0 or not math.isfinite(nats) or nats < 0:
            raise ValueError('Expected positive target count and finite nonnegative NLL')
        if seg_nats is not None:
            seg = np.asarray(seg_nats, dtype=float)
            if seg.ndim != 1 or len(seg) < 2:
                raise ValueError('seg_nats must be a vector with one entry per segment')
            if self.seg_nats is None:
                self.seg_nats = np.zeros(len(seg))
            elif len(seg) != len(self.seg_nats):
                raise ValueError('segment count changed mid-run')
            self.seg_nats += seg
            self.seg_tokens += count / len(seg)
        self.targets += count
        self.nats += nats
        self.steps += 1
        if self.targets > self.total:
            raise ValueError('Scored more targets than the full stream contains')
        if not force and self.targets < self.total and self.steps % self.interval:
            return {}
        self.edges.append(self.targets)
        self.costs.append(self.nats)
        self.block_spread.append(self._close_block())
        measured = self.literal_bytes + self.nats * BYTES_PER_NAT
        if self.targets == self.total:
            return self._metrics(measured, measured, measured, 0.0, 0.0, smooth=False)
        # Need enough blocks for a chronological holdout. Never fit the first block:
        # it contains the singular startup regime and its cost is already measured.
        if len(self.edges) < 10:
            return {}
        left = np.asarray(self.edges[1:-1], dtype=float) / self.total
        right = np.asarray(self.edges[2:], dtype=float) / self.total
        counts = np.diff(self.edges)[1:]
        losses = np.diff(self.costs)[1:] / counts
        split = max(4, int(len(losses) * .75))
        if split >= len(losses):
            return {}
        # Blocks are weighted by token count. Weighting by the inverse of the measured
        # cross-segment variance instead was tried and is neutral to slightly worse: the
        # per-block variance estimate is itself noisy, and marginal variance is the wrong
        # correction for residuals that are autocorrelated rather than heteroscedastic.
        weights = counts
        candidates = []
        for fraction in (1.0, .5):
            start = int(split * (1-fraction))
            for alpha in (0, .25, .5, 1.0):
                def fit(lo, hi):
                    y, w = losses[lo:hi], weights[lo:hi]
                    if alpha == 0:
                        return float(np.average(y, weights=w)), 0.0
                    x = power_average(left[lo:hi], right[lo:hi], alpha)
                    # Nonnegative asymptote and amplitude, including boundary optima.
                    design = np.column_stack([np.ones(len(x)), x])
                    coef = np.linalg.lstsq(design * np.sqrt(w[:, None]), y*np.sqrt(w), rcond=None)[0]
                    options = [(float(np.average(y, weights=w)), 0.0),
                               (0.0, max(0.0, float(np.sum(w*x*y)/np.sum(w*x*x))))]
                    if np.all(coef >= 0):
                        options.append(tuple(coef))
                    return min(options, key=lambda c: np.sum(w*(y-c[0]-c[1]*x)**2))
                floor, amp = fit(start, split)
                x_test = power_average(left[split:], right[split:], alpha) if alpha else 0
                error_nats = np.sum(counts[split:]*(floor+amp*x_test-losses[split:]))
                # Refit on all observed blocks, using the same window fraction.
                lo = int(len(losses)*(1-fraction))
                floor, amp = fit(lo, len(losses))
                future_x = power_average(self.targets/self.total, 1.0, alpha) if alpha else 0
                future_loss = floor + amp*future_x
                predicted = measured + (self.total-self.targets)*future_loss*BYTES_PER_NAT
                se, rho = self._future_stderr(alpha, floor, amp, future_x, left[lo:],
                                              right[lo:], losses[lo:], weights[lo:])
                candidates.append((abs(error_nats), predicted, future_loss, error_nats,
                                   se*(self.total-self.targets)*BYTES_PER_NAT, rho))
        best = min(candidates, key=lambda c: c[0])
        values = [c[1] for c in candidates]
        return self._metrics(best[1], min(values), max(values), best[3]*BYTES_PER_NAT, best[2],
                             stderr=best[4], autocorr=best[5])

    def _close_block(self):
        """Spread of per-segment mean loss across the block, in nats per token.

        Segments sit far apart in the file, so at the same within-segment position they
        are near-independent samples of local content difficulty. This separates how much
        of the loss varies because the file is heterogeneous from how much it varies
        because the model is learning. Returns nan when no per-segment costs are supplied.
        """
        if self.seg_nats is None or self.seg_tokens <= 0:
            return float('nan')
        spread = float(np.std(self.seg_nats / self.seg_tokens, ddof=1))
        self.seg_nats = np.zeros(len(self.seg_nats))
        self.seg_tokens = 0.0
        return spread

    def _future_stderr(self, alpha, floor, amp, future_x, left, right, y, w):
        """Standard error of the projected mean loss on the remaining targets.

        Weighted-least-squares covariance, inflated by the lag-1 residual
        autocorrelation: successive blocks cover contiguous text, so they carry less
        information than their count suggests.

        This is conditional on the chosen curve shape being right, so it captures sampling
        error only. Replayed against a real run its 95% band covered the eventual value
        just 26-66% of the time, because most of the early error is the curve shape being
        wrong rather than the samples being noisy. Read it as a floor on uncertainty, not
        as a confidence interval.
        """
        n = len(y)
        x = power_average(left, right, alpha) if alpha else np.zeros(n)
        design = np.column_stack([np.ones(n), x]) if alpha else np.ones((n, 1))
        resid = y - (floor + amp*x)
        dof = max(n - design.shape[1], 1)
        sigma2 = float(np.sum(w*resid**2) / dof)
        centred = resid - resid.mean()
        denom = float(np.sum(centred*centred))
        rho = float(np.sum(centred[:-1]*centred[1:]) / denom) if n > 3 and denom > 0 else 0.0
        rho = min(max(rho, 0.0), 0.95)
        try:
            cov = np.linalg.inv(design.T @ (design*w[:, None])) * sigma2 * (1+rho) / (1-rho)
        except np.linalg.LinAlgError:
            return float('nan'), rho
        v = np.array([1.0, future_x]) if alpha else np.array([1.0])
        return math.sqrt(max(float(v @ cov @ v), 0.0)), rho

    def _metrics(self, estimate, low, high, backtest_error, future_loss, smooth=True,
                 stderr=0.0, autocorr=0.0):
        # Successive fits share nearly all their data and block residuals are
        # autocorrelated, so a single fit wobbles more than it is actually uncertain.
        # Smoothing the reported estimate damps that; the raw fit stays available and
        # the exact total at complete coverage is never smoothed.
        if smooth and self.ema is not None:
            self.ema = self.ema_beta * self.ema + (1 - self.ema_beta) * estimate
        else:
            self.ema = estimate
        return {
            'forecast/final_auc_loss': self.ema,
            'forecast/final_auc_loss_raw': estimate,
            'forecast/final_total_bytes': self.ema + self.dictionary_bytes,
            'forecast/model_spread_low_bytes': low,
            'forecast/model_spread_high_bytes': high,
            'forecast/backtest_error_bytes': backtest_error,
            'forecast/remaining_nats_per_token': future_loss,
            'forecast/projection_stderr_bytes': stderr,
            'forecast/residual_autocorr': autocorr,
            'forecast/segment_loss_spread': self.block_spread[-1] if self.block_spread else float('nan'),
            'forecast/observed_fraction': self.targets / self.total,
            'forecast/scored_tokens': self.targets,
        }
