# Prequential size accounting

`prequential.py` is the main streaming trainer. `train.py` is the older windowed
trainer; both now share dictionary validation in `coding_data.py`.

## Streaming trainer

Each equal segment starts with a fixed-width token. All remaining tokens within
segments are scored by the model, including short final chunks and partial
updates. Gradients are weighted by the actual number of targets per update. The
fewer than `batch` tokens beyond the equal segments are coded literally at the
end using `ceil(log2(vocab_size))` bits each; their cost is reported separately.
This preserves existing segment boundaries and cache behavior.

`max_tokens` caps model-scored targets, rounded down to a multiple of `batch`. It
excludes initial literals and is not a contiguous original-byte prefix. A capped
run reports `complete_file=false` unless it reaches every segment end. Unvisited
data is not charged or used for replay; final bits per original byte is only
reported for complete runs. Compare capped runs only at matching coverage.

`coded_tokens` includes segment starts, model-scored targets, and any literal
tail. `predicted_tokens` counts only model-scored targets. `first_token_bytes`,
`literal_tail_tokens`, and `literal_tail_bytes` expose the literal costs.
`estimated_total_bytes` includes these and the dictionary and is the sweep
objective. Final summary totals include the literal tail, which is transmitted
after the final logged training update.

Run `.venv/bin/python test_prequential_accounting.py` for streaming integration
tests (multiple members, partial chunks/updates, replay, eviction, singleton
segments, and capped runs). Existing `test_stream.py` and `test_cache.py` also
cover stream and cache invariants.

## Windowed trainer

`train.py` scores the complete token stream once before learning from each update.
The first token costs `ceil(log2(vocab_size))` bits under a fixed-width code. All
remaining tokens, including a partial final sequence/update, contribute their
pre-update ensemble negative log probability. Context resets at each sequence
remain part of this predictor. Loss and gradients are weighted by actual token
counts when a final update contains unequal micro-batches.

Configuration in `DEFAULTS` (also accepted by `train({...})`):

- `token_path`: little-endian uint16 BPE stream.
- `source_path`: original bytes, used to verify exact reconstruction.
- `dictionary_first`, `dictionary_second`: little-endian int32 merge arrays.
- `vocab_size`: 256 byte symbols plus the number of merges.
- `context_length`: sequence length for both online training and replay.

The first `vocab_size - 256` entries of each merge array define the dictionary.
Larger files are allowed, but only this prefix is counted. Before training, the
entire token stream is decoded in chunks and compared byte-for-byte with the
source. Mismatches, out-of-range tokens, invalid merges and incomplete source
coverage fail validation. Successful checks are cached within the process using
file paths, sizes and modification times, so sweeps don't repeat the same scan.
Dictionary storage is conservatively charged as two raw int32 arrays:
`8 * (vocab_size - 256)` bytes. No unmeasured dictionary-compression savings are
assumed.

Metrics:

- `auc_loss` / `final_auc_loss`: ideal payload bytes, including the first token.
- `tokens` / summary `coded_tokens`: actual number of tokens accounted for.
- `predicted_tokens`: tokens scored by the model (excludes the first token).
- `dictionary_bytes`: the raw dictionary representation above.
- `estimated_total_bytes`: payload plus dictionary; the sweep objective.
- `estimated_bits_per_byte`: final estimated total bits / original source bytes.

These remain estimates, not archive sizes: arithmetic-coder precision,
termination/padding, metadata and decoder program costs are not measured. A real
codec round trip is still needed for an archive claim. Historical runs omitted
first/tail/dictionary costs and did not train the output bias, so mark new runs as
a changed baseline. Both trainers use the dictionary and payload accounting described here.

Verification: `.venv/bin/python test_accounting.py` exercises exact coverage,
dictionary mismatch rejection, singleton streams, non-1024 contexts, partial
updates, replay, bias updates, and reconstruction of reported payload costs.

## Live full-file forecasts

`prequential.py` records `forecast/*` W&B metrics every `forecast_interval`
updates (default 256), after at least nine observation blocks. It forecasts the
whole file even when `max_tokens` ends the experiment early. Measured costs are
never replaced by forecasts, and forecasts are not the sweep objective.

The estimator retains measured cumulative NLL and extrapolates only the
remaining targets. Candidates are constant loss and nonnegative
`floor + amplitude * progress**(-alpha)` curves with alpha 0.25, 0.5 or 1,
fitted on full or recent history. The initial observation block is excluded
from fitting, but its actual bits remain counted. Candidates are ranked by
absolute cumulative prediction error on a chronological 25% holdout, then refit
on observed history. This holdout selects the model; it is not an independent
validation of the selected forecast's uncertainty.

- `forecast/final_auc_loss`: projected complete payload, including all literals.
- `forecast/final_auc_loss_raw`: the same projection before smoothing.
- `forecast/final_total_bytes`: projected payload plus verified dictionary.
- `forecast/model_spread_low_bytes`, `forecast/model_spread_high_bytes`: range
  of candidate payload projections, **not a statistical confidence interval**.
- `forecast/backtest_error_bytes`: signed predicted minus measured holdout cost.
- `forecast/observed_fraction`, `forecast/scored_tokens`: coverage when estimated.
- `forecast/remaining_nats_per_token`: projected mean loss on remaining targets.
- `forecast/projection_stderr_bytes`: sampling error of the projection, **not a
  confidence interval** — see below.
- `forecast/residual_autocorr`: lag-1 autocorrelation of block residuals.
- `forecast/segment_loss_spread`: spread of loss across segments within the block,
  nan unless per-segment costs are supplied. Also logged per step under the same
  name outside the `forecast/` namespace.

Successive fits share nearly all their data, and block residuals are
autocorrelated, so a single fit moves around more than its information content
warrants. On a 40675-step enwik9 run the measured block-residual spread was 1.97%
of the fitted loss with lag-1 autocorrelation 0.36 and an integrated
autocorrelation time of 2.9 blocks, leaving 53 effective observations out of 157.
The reported projection is therefore an EMA over successive forecasts with
`forecast_ema` (default 0.8); `forecast_ema=0` reports each raw fit. On that run
smoothing cut forecast-to-forecast jitter from 0.76% to 0.24% and moved the point
where the projection stays within 2% of its final value from step 25087 to 15359.
It damps variance, not the underlying identifiability limit: the fitted trend
there spanned only ~11% from end to end, so the curve exponent is weakly
constrained and the projection is still worth little before roughly a third of
the targets are scored.

`prequential.py` passes per-segment costs to the forecaster. Segments sit far
apart in the file, so at equal within-segment position they sample content
difficulty near-independently, which separates file heterogeneity from learning;
`segment_loss_spread` reports it. This is a diagnostic and does not improve the
projection. Two uses for it were tried and rejected on measurement: weighting
blocks by inverse cross-segment variance is neutral to slightly worse than
weighting by token count, because the per-block variance estimate is itself noisy
and marginal variance is the wrong correction for residuals that are
autocorrelated rather than heteroscedastic; and comparing residual size against
measured content noise does not detect an out-of-family trend, because a smooth
curve partly tracks correlated noise and so deflates its own residuals either way.

`projection_stderr_bytes` is the weighted-least-squares standard error of the
projection, inflated by `(1+rho)/(1-rho)` for the residual autocorrelation. It is
conditional on the selected curve shape, so it measures sampling error only.
Replayed against a real run, its nominal 95% band contained the eventual value
26-66% of the time: most of the early error is the curve shape being wrong, not
the blocks being noisy. Treat it as a floor on uncertainty.

At complete coverage the forecast equals the measured payload exactly and is not
smoothed. This method assumes future losses follow the observed trend. Wikipedia distribution
shifts, delayed learning, and future learning-rate changes can defeat it. A
short-run cooldown based on `max_tokens` also changes the training trajectory
relative to an uncapped run. Use projections to screen runs, not to claim final
compression or automatically stop promising experiments. Reliability on actual
runs still needs testing against completed, matching configurations.
