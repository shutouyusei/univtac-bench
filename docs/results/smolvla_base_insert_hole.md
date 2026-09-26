# Baseline: SmolVLA (no tactile) on insert_hole, Isaac Sim 5.1

Recorded 2026-09-26. This is the reference number every tactile variant on the
lerobot bridge is compared against.

## Result

| eval | success | rate | 95% Wilson | rotate=0 | rotate=pi | early stop |
|---|---|---|---|---|---|---|
| 100 seeds, eval-time channel order as trained on the server at the time (see caveat) | 19/100 | 19.0% | [12.5, 27.8] | 0/42 | 19/58 | 81/100 |
| 50 seeds, eval-time channel order matched to training | 9/50 | 18.0% | [9.8, 30.8] | 1/22 | 8/28 | 41/50 |

On the 50 seeds shared by both evals: 10/50 vs 9/50, with 4 seeds succeeding in
both, 6 only in the first, 5 only in the second. The channel-order caveat below
does not move the baseline. Per-seed outcomes: `data/smolvla_base_insert_hole_seeds.json`.

Success is concentrated on the hole orientation `rotate=pi` (62% of the demos);
`rotate=0` is essentially unsolved. Typical failure: xy aligned, insertion too
shallow, prism slips more than 4 cm in hand and the episode early-stops.

## Setup

- **Policy**: `lerobot/smolvla_base` fine-tuned with `policy/lerobot/train.sh smolvla`.
  Action expert and state projection trained; VLM and vision encoder frozen
  (SmolVLA defaults `freeze_vision_encoder=true`, `train_expert_only=true`).
  Chunk 50, `n_action_steps` 50, 10 flow-matching steps at inference (`base_s10`).
- **Data**: `insert_hole/clean51`, first 100 episodes, converted with
  `policy/lerobot/process_data.py` to `local/insert_hole-clean51-100`
  (head camera 256x256, state = joint[:8], action = next joint[:8], fps 60).
  Instruction: the first `seen` sentence of `instructions/insert_hole.json`.
- **Training**: 15000 optimizer updates at effective batch 256 (64 x 4 accumulation),
  bf16, lr 5e-5 cosine to 5e-6, warmup 2000, grad clip 1.0, seed 42.
  train/loss 0.197 (first 10%) -> 0.015 (last 10%). wandb run `heav5eke`
  (project `univtac-bench`). Code at commit `e2a0f38` (branch `lerobot-three-arm`).
- **Eval**: `task_config/clean51.yml`, seeds 1000000 upward, one seed per episode,
  eval controller gains (1.25e6 / 2500). 100-seed run: `scripts/eval_policy.py`
  serially (2026-09-24). 50-seed run: `scripts/parallel_eval_policy.py`, 2 workers,
  `LEROBOT_FLIP_CAMERAS=1 LEROBOT_FLIP_TACTILE=0` (2026-09-26, commit `e76c736`).
- **Environment**: Isaac Sim 5.1 (`isaac51` line), RTX 5090, `UniVTAC` and `lerobot` conda envs.

## Caveat: camera channel order

`policy/lerobot/convert/pipeline.py` applied `bgr_to_rgb` to frames that are
already RGB, so the training dataset holds channel-swapped camera images, while
the inference server fed live frames unswapped. The 50-seed re-eval with the
server swapping too (matching training) gives the same rate, so the baseline
stands either way. The converter bug is still to be fixed; a retrain on
correct RGB is a separate experiment, not a correction of this number.

## Context

| reference (Isaac 4.5, 50 demos, 100 rollouts) | insert_hole, no tactile |
|---|---|
| UniVTAC paper, ACT vision-only | 19% |
| FTP-1 paper, pi0.5 | 31% |
| TacForcing paper, pi0.5 | 39% |
| this repo, ACT vision-only (Isaac 5.1, 100 demos, n=100) | 10% |
| this repo, ACT + tactile encoder fine-tuned (Isaac 5.1, 100 demos, n=100) | 29% |

SmolVLA is 450M parameters with 100M trained and the vision tower frozen;
pi0.5 is about 3B with the vision tower fine-tuned. A vision-only rate between
ACT and pi0.5 is the expected place for it.

## Open direction (not run)

Fine-tune the vision encoder as well (`--policy.freeze_vision_encoder=false`,
and `--policy.train_expert_only=false` to also train the VLM), so the policy can
use the post-contact prism tilt visible in the head camera. Whether this is
tested, and against which claim, is decided separately.
