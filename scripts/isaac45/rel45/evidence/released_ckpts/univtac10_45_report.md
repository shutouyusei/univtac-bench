## 1. Tactile arm (univtac), 10 seeds per task

| task | ours /10 | 95 % Wilson | authors same 10 | authors 100-seed | paper | authors-100 in | paper in |
|---|---|---|---|---|---|---|---|
| lift_bottle | 4/10 | 17-69 | 4/10 | 46 | 71 | yes | no |
| pull_out_key | 2/10 | 6-51 | 5/10 | 48 | 46 | yes | yes |
| lift_can | 1/10 | 2-40 | 2/10 | 29 | 29 | yes | yes |
| put_bottle_in_shelf | 3/10 | 11-60 | 1/10 | 8 | 31 | no | yes |
| insert_hole | 1/10 | 2-40 | 3/10 | 39 | 24 | yes | yes |
| insert_HDMI | 0/10 | 0-28 | 1/10 | 14 | 28 | yes | no |
| insert_tube | 6/10 | 31-83 | 7/10 | 64 | 56 | yes | yes |
| grasp_classify | 10/10 | 72-100 | 10/10 | 100 | 99 | yes | yes |

pooled over 8 tasks: ours 27/80 = 33.8 % (95 % 24-45); authors same seeds 33/80; authors 100-seed mean 43.5 %; paper mean 48.0 %
Spearman ours vs authors-100: 0.71; ours vs paper: 0.91; authors-100 vs paper: 0.71

## 2. Tactile effect (univtac minus vision_only), percentage points

| task | ours vis /10 | ours tac /10 | ours diff | tac-only / vis-only seeds | McNemar p | authors-100 vis | authors-100 tac | authors-100 diff | paper diff | sign ours/authors/paper |
|---|---|---|---|---|---|---|---|---|---|---|
| lift_bottle | 3/10 | 4/10 | +10 | 1 / 0 | 1.00 | 34 | 46 | +12 | +29 | +/+/+ |
| pull_out_key | 6/10 | 2/10 | -40 | 0 / 4 | 0.12 | 39 | 48 | +9 | +18 | -/+/+ |
| lift_can | 2/10 | 1/10 | -10 | 0 / 1 | 1.00 | 30 | 29 | -1 | +9 | -/-/+ |
| put_bottle_in_shelf | 3/10 | 3/10 | +0 | 1 / 1 | 1.00 | 11 | 8 | -3 | +3 | 0/-/+ |
| insert_hole | 4/10 | 1/10 | -30 | 0 / 3 | 0.25 | 19 | 39 | +20 | +5 | -/+/+ |
| insert_HDMI | 0/10 | 0/10 | +0 | 0 / 0 | 1.00 | 13 | 14 | +1 | +13 | 0/+/+ |
| insert_tube | 8/10 | 6/10 | -20 | 0 / 2 | 0.50 | 69 | 64 | -5 | +11 | -/-/+ |
| grasp_classify | 3/10 | 10/10 | +70 | 7 / 0 | 0.02 | 44 | 100 | +56 | +49 | +/+/+ |

pooled over 8 tasks: ours diff -2.5 pt (tac-only 9 / vis-only 11 seeds, McNemar p 0.824); authors-100 diff +11.1 pt; paper diff +17.1 pt
sign agreement ours vs paper 2/8; ours vs authors-100 4/8; authors-100 vs paper 5/8
Spearman of per-task diffs: ours vs paper 0.43; ours vs authors-100 0.26
