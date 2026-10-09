# Slow-side tactile forecast on failing seeds: demo-like or jam-aware? (fixed 2026-10-09, before any run)

Question (user, 2026-10-09): on seeds where the slow-fast adapter policy fails insert_hole, does the slow side's
predicted tactile look like a clean, demo-like insertion (normative) or does it foresee / follow the jam
(descriptive)? Premise of the idea "reward the fast for matching the slow's predicted tactile".

## Runs
- P: ep50_sfp_tr5_block_adapter_k10_tp_st (stflow-slowfast 97313d4, ckpt last), insert_hole clean51, seeds
  1000000-1000099, adapter univtac branch stflow-predlog (e3b801e) with stflow_trace + stflow_prediction_log
  {frames_every: 5}. tactile_scale 1 (trained setting).
- E: scripted expert, same seeds, collect mode (scripts/collect_data.py), hdf5 like the demos.

## Quantities (all in the read-out space: per-fingertip mean-pooled, layer-normed tokens, 2 x 512)
- Analysed steps: block starts of P (query % 5 == 0), where the read-out's target frame is the current frame.
- A = reading at that step; p = slow-side read-out; C ("copy") = reading at the chunk start (query 0), the last
  reading the slow side saw (slow_tactile).
- D = E's reading at the DTW-matched frame of the same seed: E's rgb_marker frames go through the policy's own
  preprocessing (JPEG as stored, resize_rgb 224, the model's tokenizer). DTW: P's measured joints 0-7 (trace) vs
  E's embodiment/joint[:, :8], each dim divided by its std over the 50 training demos, Euclidean cost,
  open end on E (P may stop early).
- d(u, v) = RMS of the difference over the 1024 numbers.
- r = (d(p, D) - d(p, A)) / (d(p, D) + d(p, A)) in [-1, 1]: -1 = p at the demo-like reference, +1 = p at the
  actual reading. r_copy = same with C in place of p.
- tau = 95th percentile of d(A, D) over all analysed steps of P's successful episodes.
- Onset (failing episodes) = first analysed step with d(A, D) > tau at it and at the next analysed step.
  Failures without such a step are counted and left out of the onset-aligned plots.
- E seeds that the expert itself failed: left out (both P and E of that seed).

## Outputs
- Table 1 (success episodes): mean d(p, A), d(C, A), d(D, A) -> is the forecast better than copying at all.
- Fig B (a) d(A, D) vs steps from onset (bins of 10), failures mean +- 95 % bootstrap CI over episodes, tau line;
  (b) r vs steps from onset, same; r_copy as dashed; success episodes' r (all analysed steps) as a band
  (mean +- CI over episodes).
- Table 2 (failures, analysed steps after onset): mean r split by whether the step's chunk started before the
  onset ("forecast made before the jam") or at/after it ("made from a jammed observation"); also before onset.
- Fig A: one failing episode chosen by rule: among failures with an onset, the one whose onset is closest to the
  median onset (ties: lower seed). Columns: onset -20, -10, 0, +10, +20, +30 steps (nearest analysed step).
  Rows: P's fingertip frames (left | right); E's frames at the DTW match; the frames whose read-out vector is
  nearest to p in a bank of all E frames and all P frames except this seed's (label shown); d(p,A), d(p,D), r.
- Reading (not a decision rule, for the user): r near -1 after onset = normative; near +1 before onset =
  foresight; near +1 only after onset and close to r_copy = following the current state.
