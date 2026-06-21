#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export PYTHONPATH="${ROOT}/original:${ROOT}:${PYTHONPATH:-}"

MODE="${1:-help}"
shift || true

case "$MODE" in
  help|-h|--help)
    cat <<'EOF'
Text Tournament POC Harness
===========================

Usage: poc/run_text_poc.sh <mode> [args...]

Setup / data:
  setup             Build validator + trainer images, refresh manifest, cache fixtures
  refresh           Refresh manifest.json from API
  cache             Cache test fixtures locally

Eval only:
  verify            Re-eval winner repos vs API scores
  smoke             Quick eval (winner repo)
  regression        3 tasks vs --model
  boss              6 boss tasks vs --model

Train:
  build-trainer     Build downloader + trainer docker images
  train-smoke       Train smallest GRPO boss task
  train-regression  Train instruct + DPO + GRPO
  train-boss        Train all 6 boss tasks

Train + eval (full loop):
  train-eval-smoke
  train-eval-regression
  train-eval-boss   Full confidence gate (4/6 wins)

Examples:
  poc/run_text_poc.sh setup
  poc/run_text_poc.sh build-trainer
  poc/run_text_poc.sh train-eval-smoke --gpu-ids 0
  poc/run_text_poc.sh train-eval-boss --gpu-ids 0 --trainer-repo miner-text

VPS first-time setup:
  poc/vps_bootstrap.sh --gpu-ids 0
  Agent guide: poc/AGENT_IMPROVE_TEST_LOOP.md
EOF
    ;;
  setup)
    docker build -f original/dockerfiles/validator.dockerfile \
      -t weightswandering/tuning_vali:latest original/
    python poc/run_text_poc.py --mode build-trainer "$@"
    python poc/run_text_poc.py --mode refresh-manifest
    python poc/run_text_poc.py --mode cache-fixtures
    echo "Setup complete."
    ;;
  refresh)
    python poc/run_text_poc.py --mode refresh-manifest "$@"
    ;;
  cache)
    python poc/run_text_poc.py --mode cache-fixtures "$@"
    ;;
  verify)
    python poc/run_text_poc.py --mode verify-baseline "$@"
    ;;
  smoke)
    python poc/run_text_poc.py --mode smoke "$@"
    ;;
  regression)
    python poc/run_text_poc.py --mode regression "$@"
    ;;
  boss)
    python poc/run_text_poc.py --mode boss-battery "$@"
    ;;
  build-trainer)
    python poc/run_text_poc.py --mode build-trainer "$@"
    ;;
  train-smoke)
    python poc/run_text_poc.py --mode train-smoke "$@"
    ;;
  train-regression)
    python poc/run_text_poc.py --mode train-regression "$@"
    ;;
  train-boss)
    python poc/run_text_poc.py --mode train-boss "$@"
    ;;
  train-eval-smoke)
    python poc/run_text_poc.py --mode train-eval-smoke "$@"
    ;;
  train-eval-regression)
    python poc/run_text_poc.py --mode train-eval-regression "$@"
    ;;
  train-eval-boss)
    python poc/run_text_poc.py --mode train-eval-boss "$@"
    ;;
  *)
    echo "Unknown mode: $MODE (run with 'help')" >&2
    exit 1
    ;;
esac
