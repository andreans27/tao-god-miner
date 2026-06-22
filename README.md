# TAO G.O.D — Text Tournament Miner

Private workspace for competing on **[Gradients on Demand (G.O.D)](https://gradients.io)** — Bittensor subnet **56** (mainnet) / **241** (testnet).

This is **not** a standard mining repo. Validators run **your training code** in Docker on their GPUs. You compete in weekly **text tournaments** (Instruct / DPO / GRPO) by submitting an open-source training repository.

**GitHub:** [andreans27/tao-god-miner](https://github.com/andreans27/tao-god-miner) (private)

---

## What this repo contains

```
GOD/
├── miner-text/          ★ YOUR CODE — training repo submitted to tournaments
├── poc/                 Local improve & test harness (not submitted)
├── original/            G.O.D subnet reference (validator eval, miner endpoint)
└── winning_repos/       Read-only copies of past tournament winners
```

| Directory | Purpose | Edit? |
|-----------|---------|-------|
| `miner-text/` | Tournament training scripts + Dockerfile | **Yes** — main strategy work |
| `poc/` | Train/eval vs last winner on real boss-round tasks | Harness only |
| `original/` | Validator docker, scoring logic, miner API | Reference / miner setup |
| `winning_repos/` | Winner source for comparison | **No** — read only |

---

## Goal

Beat the **last text tournament winner** on real boss-round tasks, then enter a live tournament with confidence.

Last reference tournament: `tourn_2cd19ba0c3b9771f_20260618` (data in `poc/manifest.json`).

**Boss round gate:** win **≥ 4 of 6** tasks vs winner (2 instruct + 2 DPO + 2 GRPO).

---

## Quick start (GPU VPS)

```bash
git clone https://github.com/andreans27/tao-god-miner.git GOD
cd GOD

# One-time setup: Docker images + manifest + test fixtures
chmod +x poc/vps_bootstrap.sh poc/run_text_poc.sh
poc/vps_bootstrap.sh --gpu-ids 0

# Improve loop
vim miner-text/scripts/...          # edit strategy
poc/run_text_poc.sh build-trainer   # rebuild after code changes
poc/run_text_poc.sh train-eval-smoke --gpu-ids 0
```

**GitHub is not required** for the local improve & test loop. Training runs from the local `miner-text/` folder via Docker.

---

## Improve & test ladder

| Step | Command | Pass criteria |
|------|---------|---------------|
| 1 | `poc/run_text_poc.sh train-eval-smoke` | Win 1× GRPO smoke task vs winner |
| 2 | `poc/run_text_poc.sh train-eval-regression` | 3/3 (instruct + DPO + GRPO) |
| 3 | `poc/run_text_poc.sh train-eval-boss` | 4/6 boss-round tasks |

Reports: `poc/results/report_*.json`

---

## Scoring (validator-equivalent)

| Task type | Better score |
|-----------|--------------|
| InstructTextTask | Lower eval loss |
| DpoTask | Lower eval loss |
| GrpoTask | Higher reward (eval_loss in harness) |

Eval uses the same Docker image as validators: `weightswandering/tuning_vali:latest`.

---

## Key files to improve

| Area | Files in `miner-text/scripts/` |
|------|--------------------------------|
| Entrypoint | `text_trainer.py`, `entrypoint.sh` |
| Training | `train_instruct.py`, `train_dpo.py`, `train_grpo.py` |
| Config / LR | `instruct_config.py`, `dpo_config.py`, `grpo_config.py`, `lr_finder.py`, `lr_search.py` |
| Checkpoint quality | `checkpoint_avg_callback.py`, `dev_pass.py`, `customized_trainer.py` |
| KL tasks | `kl_trainer.py` |
| Data | `data_filter.py`, `adaptive_max_length.py` |

Docker image: `miner-text/dockerfiles/standalone-text-trainer.dockerfile`

---

## For AI agents

**Read first:** [`poc/AGENT_IMPROVE_TEST_LOOP.md`](poc/AGENT_IMPROVE_TEST_LOOP.md)

That document is the operational source of truth: workspace rules, commands, scoring, debugging, and checklists.

**Rules for agents:**
- Edit `miner-text/scripts/` for strategy changes
- Do **not** edit `winning_repos/` or copy winner code with cosmetic renames (subnet dedup)
- Rebuild trainer (`build-trainer`) after code changes before train-eval
- Do not commit secrets (`.env`, HF tokens)

---

## Real tournament submit (later)

When ready to enter a live tournament:

1. Push `miner-text/` changes to GitHub
2. Update `original/miner/training_repo.json` (repo URL + commit hash)
3. Run `python miner-text/sync_submit_hash.py`
4. Start miner from `original/`: `task miner` → exposes `GET /training_repo/text`
5. Run dedup check vs winner before entry

See `original/docs/miners.md` for full subnet rules.

---

## VPS sweet spot

| Phase | GPU | RAM | Disk |
|-------|-----|-----|------|
| Smoke iteration | L4 / A10 24GB | 32–48 GB | 120 GB |
| **Daily improve** | **A100 40GB** | **64 GB** | **200 GB** |
| Full boss battery | A100 80GB | 128 GB | 250 GB |

---

## API & data

- Tournament manifest: `poc/manifest.json` (from `https://api.gradients.io`)
- Refresh: `poc/run_text_poc.sh refresh && poc/run_text_poc.sh cache`
- Test fixtures cached in `poc/fixtures/` (URLs expire after ~7 days)

---

## License

`miner-text/` includes `LICENSE.md` and `NOTICE` required for tournament submission (match G.O.D subnet requirements).
