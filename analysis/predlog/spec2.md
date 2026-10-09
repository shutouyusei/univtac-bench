# Checks (ii) and (iii) on the predlog data (fixed 2026-10-09, before running check2.py)

Data and quantities as in spec.md (same runs, same analysed steps, same p / A / D, same onset and tau). Groups:
failure steps after onset (main), failure steps before onset, success episodes' steps.

## (ii) Does the forecast prefer lighter contact? (reward loophole "press less")
No expert frame is without contact (the prism is held from the first saved frame), so "no contact" cannot be
tested; the test is lighter vs heavier contact inside the held range.
- Contact level of an expert frame: c = mean over the two fingertips of the maximum of press_depth over the pad.
  Standardised within the seed: c_z = (c - mean_seed) / std_seed over that seed's expert frames.
- Per analysed step: j_D = the DTW-matched expert frame (D); j_NN = the expert frame of the SAME seed whose reading
  is nearest to p (any time). Report per group: median and IQR of c_z(j_NN) - c_z(j_D) (negative = the forecast
  sits at lighter contact than the time-matched expert), fraction < -1 (more than one within-seed std lighter),
  and median |j_NN - j_D| / episode length (phase offset).
- Control: the same with A in place of p for success steps (what the actual reading's nearest frame looks like).

## (iii) Is the forecast scene-specific (plan) or a phase average?
- Phase of an analysed step: phi = j_D / (len_E - 1). For another expert seed s', its frame at phi is
  round(phi * (len_E' - 1)) and D_s' its reading. M = mean of D_s' over all other analysed seeds (phase average).
- Per step: d(p, D_same), mean over s' of d(p, D_s') (= d_other), d(p, M).
  Specificity S_p = d_other - d(p, D_same) (> 0: closer to its own scene's expert than to other scenes').
  Average-likeness: d(p, M) - d(p, D_same) (< 0: closer to the phase average than to its own scene).
- Control (ceiling of scene specificity in this space): S_A = same quantity with A in place of p on success steps.
- Report per group: per-episode means, then mean over episodes with 95 % bootstrap CI.

## Outputs
check2_table.md (both checks per group), check2_fig.png: (left) histogram of c_z(j_NN) - c_z(j_D) per group;
(right) S_p per group with S_A control as bars with CI. No decision rule; the user reads them.
