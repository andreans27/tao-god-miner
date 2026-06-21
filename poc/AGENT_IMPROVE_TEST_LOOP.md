# AI Agent Knowledge: G.O.D Text Improve & Test Loop

This document is the **source of truth** for an AI coding agent running on a GPU VPS.
Read this before editing code or running commands.

**Goal:** Improve `miner-text/` training strategy and prove it beats the last text tournament winner on real boss-round tasks — without GitHub or tournament registration.

---

## 1. Workspace map

```
GOD/                              # project root — always cd here first
├── miner-text/                   # ★ EDIT THIS — tournament training repo (local)
│   ├── dockerfiles/standalone-text-trainer.dockerfile
│   ├── scripts/
│   │   ├── text_trainer.py       # Docker entrypoint
│   │   ├── train_instruct.py     # InstructTextTask training
│   │   ├── train_dpo.py          # DpoTask training
│   │   ├── train_grpo.py         # GrpoTask training
│   │   ├── lr_finder.py          # LR from baseline stats
│   │   ├── lr_search.py          # In-process LR search
│   │   ├── checkpoint_avg_callback.py  # Soup + overfit rollback
│   │   ├── kl_trainer.py         # KL-regularised instruct
│   │   ├── customized_trainer.py # Wall-clock budget / eval callbacks
│   │   ├── dev_pass.py           # Final dev-set reclaim
│   │   ├── instruct_config.py / dpo_config.py / grpo_config.py
│   │   └── data_filter.py / adaptive_max_length.py
│   ├── LICENSE.md + NOTICE       # Required for real tournament; do not remove
│   └── sync_submit_hash.py       # Only needed before real tournament submit
│
├── poc/                          # Local harness — do NOT submit to tournament
│   ├── run_text_poc.sh           # Main CLI wrapper
│   ├── run_text_poc.py           # Modes: train-eval-*, verify, boss-battery
│   ├── vps_bootstrap.sh          # One-shot VPS setup
│   ├── manifest.json             # Real data from last text tournament boss round
│   ├── fixtures/                 # Cached test JSON (survives 7-day URL expiry)
│   ├── cache/                    # Downloaded models + train datasets
│   ├── outputs/                  # Training checkpoints from local runs
│   └── results/                  # JSON reports (pass/fail gate)
│
├── original/                     # Validator/miner subnet repo — reference only for POC
│   └── validator/evaluation/     # How scoring works (read-only context)
│
└── winning_repos/text-winner-latest/  # Reference copy — do not edit; compare here
```

### What to edit vs leave alone

| Path | Agent may edit? | Notes |
|------|-----------------|-------|
| `miner-text/scripts/**` | **Yes** | Core strategy improvements |
| `miner-text/dockerfiles/**` | Careful | Only if deps/image broken |
| `poc/**` | Only if harness broken | Not the mining strategy |
| `original/**` | **No** | Unless explicitly fixing eval harness imports |
| `winning_repos/**` | **No** | Read-only reference |

---

## 2. Scoring rules (must match validator)

Ranking uses **test set eval_loss** from validator Docker (`weightswandering/tuning_vali:latest`).

| Task type | Better score | Win condition vs opponent |
|-----------|--------------|---------------------------|
| `InstructTextTask` | Lower loss | `your_loss < opponent_loss` |
| `DpoTask` | Lower loss | `your_loss < opponent_loss` |
| `GrpoTask` | Higher reward (= eval_loss in harness) | `your_loss > opponent_loss` |

**Boss round gate:** Win **≥ 4 of 6** boss tasks vs winner.

**KL tasks:** If env `USE_KL=1` and `KL_COEF` set, training objective must include matching KL term (`kl_trainer.py`). Check manifest per task.

**Invalid submission:** `is_finetune=false` in eval → treat as loss.

Opponent default: **winner HF repo** from `poc/manifest.json` (`gradients-io-tournaments/...`).

---

## 3. Real tournament data (manifest)

Manifest: `poc/manifest.json` — sourced from API tournament `tourn_2cd19ba0c3b9771f_20260618`.

| Suite | Tasks | Purpose |
|-------|-------|---------|
| **smoke** | 1× GrpoTask (`Sheared-LLaMA-1.3B`, 1.5h) | Fast iteration |
| **regression** | 1 instruct + 1 DPO + 1 GRPO | Per-type check |
| **boss_battery** | 6 tasks (2+2+2) | Pre-tournament confidence gate |

Refresh manifest from API (optional, if stale):

```bash
poc/run_text_poc.sh refresh
poc/run_text_poc.sh cache
```

**Always use cached fixtures** (`poc/fixtures/`) — test URLs expire after ~7 days.

---

## 4. VPS bootstrap (first time)

```bash
cd ~/GOD
chmod +x poc/vps_bootstrap.sh poc/run_text_poc.sh
poc/vps_bootstrap.sh --gpu-ids 0
```

This checks GPU + Docker, creates `.venv`, builds images, caches fixtures, runs `verify`.

**First `setup` takes 1–2 hours** (trainer image: flash-attn, vllm, axolotl).

Use **tmux** for long runs:

```bash
tmux new -s setup
poc/vps_bootstrap.sh
# Ctrl+B, then D to detach
```

### VPS requirements

- Ubuntu 22.04+, NVIDIA driver, Docker, `nvidia-container-toolkit`
- GPU: ≥24GB VRAM for smoke; ≥80GB recommended for 7B+ boss tasks
- RAM: 64GB+ ; Disk: 150GB+ free
- Internet required for: Docker build, HF model download, API manifest fetch

### Optional env

```bash
export HUGGINGFACE_TOKEN=hf_...   # gated models only
```

GitHub repo for `miner-text` is **NOT required** for improve & test loop.

---

## 5. Improve & test loop (agent workflow)

### Standard iteration

```bash
cd ~/GOD
source .venv/bin/activate

# 1. Edit strategy (miner-text/scripts/...)
# 2. Rebuild trainer image after code changes
poc/run_text_poc.sh build-trainer

# 3. Train + eval vs winner on smoke task
poc/run_text_poc.sh train-eval-smoke --gpu-ids 0

# 4. Read report
cat poc/results/report_*.json | tail -50
```

### Escalation ladder

| Step | Command | Pass gate |
|------|---------|-----------|
| 1 | `train-eval-smoke` | Win smoke task vs winner |
| 2 | `train-eval-regression` | 3/3 task types |
| 3 | `train-eval-boss` | 4/6 boss tasks |

Only escalate after previous step passes.

### After code change

Always `build-trainer` before train. Use `--skip-build` only if Dockerfile and `miner-text/scripts/` unchanged (not supported in shell wrapper — call Python with `--skip-build` if needed).

### Flags (pass through to `run_text_poc.py`)

```bash
poc/run_text_poc.sh train-eval-smoke --gpu-ids 0 1
poc/run_text_poc.py --mode train-eval-smoke --hours-override 0.5 --gpu-ids 0  # quick debug only
poc/run_text_poc.py --mode train-eval-smoke --skip-build --gpu-ids 0
```

`--hours-override` produces non-representative scores — debug only, not for gate decisions.

---

## 6. What train-eval does internally

1. **Download** base model + training dataset → `poc/cache/` (downloader Docker from `original/`)
2. **Train** with `miner-text` Docker → `poc/outputs/{task_id}/poc-{task_id[:8]}/`
3. **Success check:** `success.txt` exists in output dir
4. **Stage** checkpoint into `~/.cache/huggingface/hub/` as `poc-local/poc-{id}`
5. **Eval** with validator Docker on cached test fixture
6. **Compare** vs winner repo; write `poc/results/report_*.json`

No GitHub upload in default flow.

---

## 7. High-value improvement areas

Last boss round pattern (challenger lost overall despite winning GRPO):

| Area | Files | Why |
|------|-------|-----|
| Instruct/DPO loss | `train_instruct.py`, `train_dpo.py`, `instruct_config.py`, `dpo_config.py` | Challenger lost instruct+DPO margins |
| GRPO reward | `train_grpo.py`, `grpo_config.py` | Already strong; keep edge |
| LR | `lr_finder.py`, `lr_search.py` | DPO very LR-sensitive |
| Time budget | `customized_trainer.py` | Multi-run wall-clock orchestration |
| KL alignment | `kl_trainer.py` | When `USE_KL=1` on task |
| Data quality | `data_filter.py` | Outlier removal before train |
| Checkpoint selection | `checkpoint_avg_callback.py`, `dev_pass.py` | Final model quality |

### Do NOT

- Copy `winning_repos/` verbatim with cosmetic renames (subnet dedup detects this)
- Log strategy details that reveal duplicate of public winner (use `quiet_mode.py` patterns)
- Change output path from `/app/checkpoints/{task_id}/{expected_repo_name}` for real tournament
- Break `is_finetune` (architecture mismatch in `config.json` — see `patch_submission_architectures`)

---

## 8. Debugging failures

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| `Validator image not found` | Setup incomplete | `poc/run_text_poc.sh setup` |
| `trainer image` build OOM | VPS RAM low | More RAM or build with swap |
| Train exits, no `success.txt` | OOM / timeout / crash | Check docker logs; reduce batch or hours |
| `is_finetune=false` | Bad checkpoint / wrong arch | Fix config patch; verify LoRA merge |
| Eval URL expired | Fixtures missing | `poc/run_text_poc.sh cache` |
| `verify` DRIFT | Validator image stale | Rebuild validator docker |
| CUDA OOM during train | Model too large for GPU | Use smoke task GPU first |

### Useful commands

```bash
docker images | grep -E 'poc-|tuning_vali'
docker ps -a | head
ls -la poc/outputs/*/
ls -la poc/results/
python -m pytest poc/tests/ -q   # harness unit tests (no GPU)
```

---

## 9. Agent decision checklist

Before ending a session, verify:

- [ ] Changes are in `miner-text/scripts/` (not `poc/` unless harness fix)
- [ ] `build-trainer` run after edits
- [ ] `train-eval-smoke` executed and report saved
- [ ] Compared margin vs winner documented in commit message or report
- [ ] No secrets committed (HF tokens, `.env`)
- [ ] Did not edit `winning_repos/` or `original/miner/training_repo.json` unless asked

---

## 10. Real tournament (out of scope for VPS loop)

Only when human asks to submit:

1. Push `miner-text/` to GitHub
2. Edit `original/miner/training_repo.json` (repo URL + commit)
3. Run `python miner-text/sync_submit_hash.py`
4. Run dedup check vs winner before entry
5. Start miner from `original/` (`task miner`)

**The VPS improve loop does not require any of the above.**

---

## 11. Quick reference commands

```bash
# Bootstrap VPS
poc/vps_bootstrap.sh --gpu-ids 0

# Iteration
poc/run_text_poc.sh build-trainer
poc/run_text_poc.sh train-eval-smoke --gpu-ids 0
poc/run_text_poc.sh train-eval-regression --gpu-ids 0
poc/run_text_poc.sh train-eval-boss --gpu-ids 0

# Eval-only (existing HF checkpoint)
poc/run_text_poc.sh boss --model your-hf/repo --gpu-ids 0

# Unit tests (no GPU)
source .venv/bin/activate
PYTHONPATH=original:. pytest poc/tests/ -q
```

---

## 12. Key file outputs

| Output | Meaning |
|--------|---------|
| `poc/results/report_train-eval-smoke_*.json` | Pass/fail + per-task margins |
| `poc/results/train_train-smoke_*.json` | Training success per task |
| `poc/outputs/{task_id}/poc-*/` | Local checkpoint |
| `GATE: PASS` in stdout | Smoke/regression/boss gate cleared |

**Confidence labels:** `READY` | `NOT_READY` | `PARTIAL` (see `poc/scoring.py`).

---

*Last updated: 2026-06-21. Tournament reference: `tourn_2cd19ba0c3b9771f_20260618`.*
