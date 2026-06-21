#!/usr/bin/env bash
# One-shot VPS bootstrap for G.O.D text tournament improve & test loop.
# Usage: ./poc/vps_bootstrap.sh [--skip-setup] [--skip-verify] [--gpu-ids 0]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SKIP_SETUP=0
SKIP_VERIFY=0
GPU_IDS=(0)

usage() {
  cat <<'EOF'
VPS bootstrap for improve & test loop

Usage: poc/vps_bootstrap.sh [options]

Options:
  --skip-setup     Skip Docker image build + manifest/fixture cache
  --skip-verify    Skip post-setup harness verification
  --gpu-ids IDS    GPU device IDs for verify step (default: 0)
  -h, --help       Show this help

Typical first run on a fresh GPU VPS:
  poc/vps_bootstrap.sh

After editing miner-text/:
  poc/run_text_poc.sh build-trainer
  poc/run_text_poc.sh train-eval-smoke --gpu-ids 0

Agent knowledge doc: poc/AGENT_IMPROVE_TEST_LOOP.md
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-setup) SKIP_SETUP=1; shift ;;
    --skip-verify) SKIP_VERIFY=1; shift ;;
    --gpu-ids)
      shift
      GPU_IDS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        GPU_IDS+=("$1")
        shift
      done
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

log() { printf '\n==> %s\n' "$*"; }

log "G.O.D VPS bootstrap (root: $ROOT)"

log "Preflight: NVIDIA driver"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. Install NVIDIA driver first." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

log "Preflight: Docker"
if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker not found. Install Docker first." >&2
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "ERROR: docker daemon not reachable. Add user to docker group or use sudo." >&2
  exit 1
fi

log "Preflight: Docker GPU access"
if ! docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: docker --gpus all failed. Install nvidia-container-toolkit." >&2
  exit 1
fi
echo "Docker GPU OK"

log "Python venv + harness dependencies"
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r poc/requirements.txt

export PYTHONPATH="${ROOT}/original:${ROOT}:${PYTHONPATH:-}"

if [[ "$SKIP_SETUP" -eq 0 ]]; then
  log "Docker setup (validator + trainer + manifest + fixtures)"
  log "WARNING: trainer image build can take 1-2 hours on first run."
  poc/run_text_poc.sh setup
else
  log "Skipping setup (--skip-setup)"
fi

if [[ "$SKIP_VERIFY" -eq 0 ]]; then
  log "Harness verification (re-eval winner vs API scores)"
  GPU_ARGS=(--gpu-ids "${GPU_IDS[@]}")
  poc/run_text_poc.sh verify "${GPU_ARGS[@]}"
else
  log "Skipping verify (--skip-verify)"
fi

log "Bootstrap complete"
cat <<EOF

Next steps:
  1. Read agent guide:  poc/AGENT_IMPROVE_TEST_LOOP.md
  2. Improve code:      miner-text/scripts/
  3. Rebuild trainer:   poc/run_text_poc.sh build-trainer
  4. Train + eval:      poc/run_text_poc.sh train-eval-smoke --gpu-ids ${GPU_IDS[*]}

Long jobs — use tmux:
  tmux new -s train
  poc/run_text_poc.sh train-eval-smoke --gpu-ids ${GPU_IDS[*]}
EOF
