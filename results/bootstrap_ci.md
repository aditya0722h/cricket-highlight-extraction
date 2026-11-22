# Bootstrap confidence intervals

Run date: 2026-11-22T14:14:56.750314+05:30
Git commit: `822c708f67b8e11562f9649aed52dad9e3d26af2`

Interval bootstrap: 1000 resamples, fixed seed 20261122. Predictions are fixed; labeled intervals are sampled with replacement.

## cricket_match_1: n=20 intervals

| Metric | Result |
|---|---:|
| ROC-AUC visual | 0.4785 [0.4398, 0.5159] |
| ROC-AUC audio | 0.6576 [0.6178, 0.6947] |
| ROC-AUC speech | 0.4938 [0.4508, 0.5318] |
| Strict F1 at k=10 audio | 0.2667 [0.0667, 0.2667] |
| Strict F1 at k=10 equal_fusion | 0.0667 [0.0000, 0.0667] |

## cricket_match_2: n=20 intervals

| Metric | Result |
|---|---:|
| ROC-AUC visual | 0.5828 [0.5390, 0.6297] |
| ROC-AUC audio | 0.6524 [0.6059, 0.6948] |
| ROC-AUC speech | 0.6103 [0.5677, 0.6561] |
| Strict F1 at k=10 audio | 0.0667 [0.0000, 0.0667] |
| Strict F1 at k=10 equal_fusion | 0.0000 [0.0000, 0.0000] |

## pooled_both_matches: n=40 intervals

| Metric | Result |
|---|---:|
| ROC-AUC visual | 0.5292 [0.5003, 0.5568] |
| ROC-AUC audio | 0.6554 [0.6267, 0.6828] |
| ROC-AUC speech | 0.5544 [0.5211, 0.5852] |
| Strict F1 at k=10 audio | 0.1667 [0.0333, 0.1667] |
| Strict F1 at k=10 equal_fusion | 0.0333 [0.0000, 0.0333] |
