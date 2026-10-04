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
#
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
if command -v uv >/dev/null; then
	PIP=(uv pip install --python "$PY")
else
	"$PY" -m pip install --quiet --upgrade pip
	PIP=("$PY" -m pip install)
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

# ---- 3. Hint arrays (~13 GB, deterministic, cheaper to rebuild than to upload) ----
if [ ! -f 5k_enwik9.hint_tok.npy ] || [ ! -f 5k_enwik9.freq_tok.npy ]; then
	"$PY" hints.py               # match hints + v1 frequency hints, ~3 min
fi
if [ ! -f 5k_enwik9.freq2_tok.npy ]; then
	"$PY" hints.py --freq2       # v2 frequency hints, ~4 min, needs ~7 GB RAM
fi

# ---- 4. Weights & Biases ----
if [ -z "${WANDB_API_KEY:-}" ] && ! grep -qs "api.wandb.ai" ~/.netrc; then
	echo "not logged in to wandb: run  .venv/bin/wandb login  (or set WANDB_API_KEY)"
fi
echo "setup done. train with:  $PY train.py"
