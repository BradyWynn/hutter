#!/usr/bin/env bash
# Server setup. Run from a fresh clone after copying the token stream in:
#
#   git clone https://github.com/BradyWynn/hutter.git && cd hutter
#   scp <local>:hutter/5k_enwik9 .          # 497 MB; must match the committed first/second
#   bash setup.sh
#
# Environment overrides:
#   TORCH_INDEX  PyTorch wheel index for the server's CUDA driver (default cu130; e.g.
#                https://download.pytorch.org/whl/cu128 for an older driver)
#   PYTHON       interpreter for the venv when uv is not installed (default python3.12)
#   HINT_FILES   1 = write the hint arrays to disk (~13 GB), 0 = don't (train.py then builds
#                them in memory at startup, ~7 min of CPU per process). Default: 1 only if at
#                least 20 GB are free.
#
# Disk: ~8 GB without hint files (env ~6.5, tokens 0.5, enwik9 1), ~21 GB with them.
# Safe to rerun: every step skips work that is already done.
set -euo pipefail
cd "$(dirname "$0")"

TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}
PYTHON=${PYTHON:-python3.12}
PY=.venv/bin/python

# ---- 1. Python environment ----
if [ ! -x "$PY" ]; then
	if command -v uv >/dev/null; then uv venv --python 3.12 .venv; else "$PYTHON" -m venv .venv; fi
fi
# no download cache: the torch wheel and its CUDA libraries would otherwise be stored twice
if command -v uv >/dev/null; then
	PIP=(uv pip install --no-cache --python "$PY")
else
	"$PY" -m pip install --quiet --no-cache-dir --upgrade pip
	PIP=("$PY" -m pip install --no-cache-dir)
fi
"${PIP[@]}" torch==2.9.1 --index-url "$TORCH_INDEX"
"${PIP[@]}" -r requirements.txt
"$PY" -c "import torch; assert torch.cuda.is_available(), 'torch sees no GPU: try another TORCH_INDEX for this driver'; print('torch', torch.__version__, '|', torch.cuda.get_device_name())"

# ---- 2. Data ----
if [ ! -f enwik9 ]; then
	echo "downloading enwik9..."
	if command -v wget >/dev/null; then wget -q http://mattmahoney.net/dc/enwik9.zip; else curl -sSLO http://mattmahoney.net/dc/enwik9.zip; fi
	"$PY" -m zipfile -e enwik9.zip .     # no unzip/apt needed
	rm enwik9.zip
fi
if [ ! -f 5k_enwik9 ]; then
	echo "missing 5k_enwik9: copy it from the machine that made it (see the top of this file)" >&2
	exit 1
fi
# the token stream must decode, through the committed dictionary, to exactly enwik9
"$PY" -c "from coding_data import validate_dictionary as v; print('tokens verified:', v('5k_enwik9', 'enwik9', 'first', 'second', 5256))"

# ---- 3. Hint arrays (optional on disk: train.py builds identical ones in memory when absent) ----
free_gb=$(df -Pk . | awk 'NR==2 {print int($4 / 1048576)}')
HINT_FILES=${HINT_FILES:-$([ "$free_gb" -ge 20 ] && echo 1 || echo 0)}
if [ "$HINT_FILES" = 1 ]; then
	if [ ! -f 5k_enwik9.hint_tok.npy ] || [ ! -f 5k_enwik9.freq_tok.npy ]; then
		"$PY" hints.py               # match hints + v1 frequency hints (~5 GB), ~3 min
	fi
	if [ ! -f 5k_enwik9.freq2_tok.npy ]; then
		"$PY" hints.py --freq2       # v2 frequency hints (~7.5 GB), ~4 min, needs ~7 GB RAM
	fi
else
	echo "skipping hint files (${free_gb} GB free): train.py builds the hints in memory at startup (~7 min, ~5 GB RAM)"
fi

# ---- 4. Weights & Biases ----
if [ -z "${WANDB_API_KEY:-}" ] && ! grep -qs "api.wandb.ai" ~/.netrc; then
	echo "not logged in to wandb: run  .venv/bin/wandb login  (or set WANDB_API_KEY)"
fi
echo "setup done. train with:  $PY train.py"
