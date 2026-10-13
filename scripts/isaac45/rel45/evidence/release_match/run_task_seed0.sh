#!/bin/bash
# One seed of one task collected on Isaac Sim 4.5 (RTX 5090, same render overrides as the evals), with tactile
# patches chosen by variant:
#   run_task_seed0.sh <task> <variant> <seed>
#   F        marker noise 0 (run_marker_noise.py) + marker grid on the camera axis (run_marker_axis.py), public grasp
#   F<mm>    same + adaptive-grasp threshold <mm> (run_grasp_thr.py), e.g. F28.05 (fails: markers outside the hull)
#   G        marker noise 0 + a9abc0d marker weights only (OLD_MARKER_SETUP=0: no setup re-call), public grasp
#   G<mm>    same + adaptive-grasp threshold <mm>
#   A        public code as is
# Output hdf5 under tactile_seed0/out/<task>/seed0_<variant>/ (compared with the released demo of the same seed by
# compare_tasks.py). User asked for these checks 2026-10-13; EULA accepted by the user.
set -uo pipefail
TASK=$1; V=$2; SEED=$3
D=/home/yusei/.claude/jobs/7fcd22c8/tmp/tactile_seed0
ROOT=/home/yusei/Desktop/Robotics/univtac-isaac45
PY=/home/yusei/miniforge3/envs/UniVTAC45/bin/python
export OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=0
export UNIVTAC45_RENDER_OVERRIDES="/rtx/directLighting/sampledLighting/enabled=false;/rtx-transient/dldenoiser/enabled=false"
# Optional camera-render suffix on the variant: _R2 = sampled lighting off only (DL denoiser left at the balanced
# preset); _R3 = no overrides + UniVTAC issue #16 RenderCfg (run_render_cfg.py). Default (no suffix) = R1 above.
RPRE=(); EXTRA=()
# Optional "_s<seed>" suffix only names the output folder (seed0_<V>) so several seeds of one variant can coexist.
VV=$V
case $VV in *_s[0-9]*) VV=${VV%_s*} ;; esac
case $VV in
  *_R2) export UNIVTAC45_RENDER_OVERRIDES="/rtx/directLighting/sampledLighting/enabled=false"; BASE=${VV%_R2} ;;
  *_R3) export UNIVTAC45_RENDER_OVERRIDES=""; RPRE=("$D/run_render_cfg.py"); BASE=${VV%_R3} ;;
  # R5-R8: sampled lighting ON (balanced preset), DL denoiser OFF, varying samples per pixel / sampled-lighting denoiser
  *_R5) export UNIVTAC45_RENDER_OVERRIDES="/rtx-transient/dldenoiser/enabled=false"; BASE=${VV%_R5} ;;
  # R9: balanced preset untouched (sampled lighting ON, DL denoiser ON) - for the 5.1 DLSS library swap test
  *_R9) export UNIVTAC45_RENDER_OVERRIDES=""; BASE=${VV%_R9} ;;
  # R10 / R11 (5.1 DLSS libs): quality preset via --rendering_mode; R11 adds the issue #16 RenderCfg (DLAA, DLSS 2...)
  *_R10) export UNIVTAC45_RENDER_OVERRIDES="" UNIVTAC45_RENDERING_MODE=quality; BASE=${VV%_R10} ;;
  *_R11) export UNIVTAC45_RENDER_OVERRIDES="" UNIVTAC45_RENDERING_MODE=quality; RPRE=("$D/run_render_cfg.py"); BASE=${VV%_R11} ;;
  *_R6) export UNIVTAC45_RENDER_OVERRIDES="/rtx-transient/dldenoiser/enabled=false;/rtx/directLighting/sampledLighting/clampSamplesPerPixelToNumberOfLights=false;/rtx/directLighting/sampledLighting/samplesPerPixel=8"; BASE=${VV%_R6} ;;
  *_R7) export UNIVTAC45_RENDER_OVERRIDES="/rtx-transient/dldenoiser/enabled=false;/rtx/directLighting/sampledLighting/clampSamplesPerPixelToNumberOfLights=false;/rtx/directLighting/sampledLighting/samplesPerPixel=32"; BASE=${VV%_R7} ;;
  *_R8) export UNIVTAC45_RENDER_OVERRIDES="/rtx-transient/dldenoiser/enabled=false;/rtx/directLighting/sampledLighting/clampSamplesPerPixelToNumberOfLights=false;/rtx/directLighting/sampledLighting/samplesPerPixel=8;/rtx/directLighting/sampledLighting/denoisingTechnique=5"; BASE=${VV%_R8} ;;
  *) BASE=$VV ;;
esac
case $BASE in
  A) PRE=() ;;
  F) export MARKER_NOISE=0; PRE=("$D/run_marker_noise.py" "$D/run_marker_axis.py") ;;
  F*) export MARKER_NOISE=0 GRASP_THR=${BASE#F}
      PRE=("$D/run_grasp_thr.py" "$D/run_marker_noise.py" "$D/run_marker_axis.py") ;;
  G) export MARKER_NOISE=0 OLD_MARKER_SETUP=0; PRE=("$D/run_old_marker.py" "$D/run_marker_noise.py") ;;
  G*) export MARKER_NOISE=0 OLD_MARKER_SETUP=0 GRASP_THR=${BASE#G}
      PRE=("$D/run_old_marker.py" "$D/run_grasp_thr.py" "$D/run_marker_noise.py") ;;
  *) echo "variant A, F, F<mm>, G or G<mm> (optional suffix _R2 / _R3)" >&2; exit 2 ;;
esac
PRE+=("${RPRE[@]}")
[ -f "$D/seed0_$V.yml" ] || cp "$D/seed0_A.yml" "$D/seed0_$V.yml"
cd "$ROOT"
echo "task $TASK variant $V seed $SEED render overrides [${UNIVTAC45_RENDER_OVERRIDES}] $(date)"
# Kit does not exit after an uncaught exception in the collect script; the timeout keeps the queue moving.
timeout -k 30 900 "$PY" "${PRE[@]}" scripts/isaac45/run_with_render_overrides.py scripts/collect_data.py "$TASK" \
  "$D/seed0_$V.yml" --start_seed "$SEED" --max_seed "$SEED" --episode_num 1 "${EXTRA[@]}"
CODE=$?
echo "done $TASK $V exit $CODE $(date)"
exit $CODE
