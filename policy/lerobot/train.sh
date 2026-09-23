#!/usr/bin/env bash
# Train stock SmolVLA or the tacforcing plugin on a dataset from process_data.py.
#
#   bash policy/lerobot/train.sh smolvla    policy/lerobot/data/local/insert_hole-clean51-50 300
#   bash policy/lerobot/train.sh tacforcing policy/lerobot/data/local/insert_hole-clean51-50 300 [extra lerobot-train args]
#
# Output: policy/lerobot/outputs/train/<policy>_<dataset name>/checkpoints/last/pretrained_model
# (the directory deploy_<policy>.yml points at). Env: LEROBOT_PYTHON (default
# ~/miniforge3/envs/lerobot/bin/python), BATCH_SIZE (8), WANDB (0).
#
# smolvla:    lerobot/smolvla_base fine-tuned as is; the dataset's head camera is
#             renamed to the checkpoint's camera1 (wrist -> camera2 for two-camera
#             tasks). Tactile is not an input: this is the "base" row.
# tacforcing: the plugin built from --policy.type, initialised from
#             lerobot/smolvla_base through its init_from; tactile_channels comes
#             from the dataset's source_metadata.json. B=5, H=50, one sampling
#             step per block (K=10 steps), as in the paper.
set -euo pipefail

POLICY="${1:?smolvla|tacforcing}"
DATASET="${2:?dataset root written by process_data.py}"
STEPS="${3:-300}"
shift 3 2>/dev/null || shift $#

cd "$(dirname "$0")/../.."
LEROBOT_PYTHON="${LEROBOT_PYTHON:-$HOME/miniforge3/envs/lerobot/bin/python}"
LEROBOT_TRAIN="$(dirname "$LEROBOT_PYTHON")/lerobot-train"
DATASET="$(realpath "$DATASET")"
META="$DATASET/source_metadata.json"
[ -f "$META" ] || { echo "missing $META (not a process_data.py output)" >&2; exit 1; }

meta() { "$LEROBOT_PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])" "$META" "$1"; }
REPO_ID="$(meta repo_id)"
TACTILE_CHANNELS="$(meta tactile_channels)"
RENAME_MAP="$("$LEROBOT_PYTHON" - "$META" <<'EOF'
import json, sys
cams = json.load(open(sys.argv[1]))["cameras"]
print(json.dumps({f"observation.images.{c}": f"observation.images.camera{i + 1}" for i, c in enumerate(cams)}))
EOF
)"

NAME="${POLICY}_$(basename "$DATASET")"
OUT="policy/lerobot/outputs/train/${NAME}"
WANDB_ARGS=(--wandb.enable=false)
if [ "${WANDB:-0}" = "1" ]; then
  WANDB_ARGS=(--wandb.enable=true --wandb.project="${WANDB_PROJECT:-univtac-bench}" --wandb.disable_artifact=true)
fi

case "$POLICY" in
  smolvla)
    POLICY_ARGS=(
      --policy.path=lerobot/smolvla_base
      --rename_map="$RENAME_MAP"
    )
    ;;
  tacforcing)
    POLICY_ARGS=(
      --policy.type=tacforcing
      --policy.init_from=lerobot/smolvla_base
      --policy.tactile_channels="$TACTILE_CHANNELS"
      --policy.block_size=5
      --policy.chunk_size=50
      --policy.n_action_steps=50
      --policy.steps_per_block=1
    )
    ;;
  *)
    echo "unknown policy '$POLICY' (smolvla|tacforcing)" >&2
    exit 1
    ;;
esac

rm -rf "$OUT"
echo ">> $POLICY on $REPO_ID ($DATASET), $STEPS steps -> $OUT"
"$LEROBOT_TRAIN" \
  --dataset.repo_id="$REPO_ID" \
  --dataset.root="$DATASET" \
  "${POLICY_ARGS[@]}" \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir="$OUT" \
  --job_name="$NAME" \
  --steps="$STEPS" \
  --save_freq="$STEPS" \
  --batch_size="${BATCH_SIZE:-8}" \
  --seed=42 \
  "${WANDB_ARGS[@]}" \
  "$@"
