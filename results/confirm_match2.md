# Second-match confirmation

Run date: 2026-11-22T14:14:40.353391+05:30
Git commit: `822c708f67b8e11562f9649aed52dad9e3d26af2`

This is the untouched cricket_match_2, human-verified, single-labeler set. Peak defaults were fixed before this run: 120 s rolling z-score, 5 s smoothing, 30 s separation, and 8--30 s variable windows.

## Channel scores against human labels

| Channel | ROC-AUC | Average precision |
|---|---:|---:|
| visual | 0.5828 | 0.0274 |
| audio | 0.6524 | 0.0593 |
| speech | 0.6103 | 0.0365 |

## Peak-selection results

Strict uses IoU >= 0.5 maximum one-to-one matching. The looser metric allows a window to overlap a label expanded by +/-10 seconds, also one-to-one.

| Variant | k | Strict P | Strict R | Strict F1 | Looser P | Looser R | Looser F1 | Random strict F1 | Even strict F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| visual | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0016 +/- 0.0112 | 0.0000 |
| audio | 5 | 0.0000 | 0.0000 | 0.0000 | 0.8000 | 0.2000 | 0.3200 | 0.0040 +/- 0.0174 | 0.0000 |
| speech | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0048 +/- 0.0190 | 0.0000 |
| equal_fusion | 5 | 0.0000 | 0.0000 | 0.0000 | 0.4000 | 0.1000 | 0.1600 | 0.0016 +/- 0.0112 | 0.0000 |
| visual | 10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0067 +/- 0.0200 | 0.0000 |
| audio | 10 | 0.1000 | 0.0500 | 0.0667 | 0.7000 | 0.3500 | 0.4667 | 0.0087 +/- 0.0224 | 0.0000 |
| speech | 10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0093 +/- 0.0231 | 0.0000 |
| equal_fusion | 10 | 0.0000 | 0.0000 | 0.0000 | 0.2000 | 0.1000 | 0.1333 | 0.0080 +/- 0.0217 | 0.0000 |
| visual | 20 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0125 +/- 0.0249 | 0.0000 |
| audio | 20 | 0.1000 | 0.1000 | 0.1000 | 0.5000 | 0.5000 | 0.5000 | 0.0115 +/- 0.0222 | 0.0000 |
| speech | 20 | 0.0000 | 0.0000 | 0.0000 | 0.0500 | 0.0500 | 0.0500 | 0.0140 +/- 0.0246 | 0.0000 |
| equal_fusion | 20 | 0.0000 | 0.0000 | 0.0000 | 0.2500 | 0.2500 | 0.2500 | 0.0115 +/- 0.0233 | 0.0500 |
| visual | 40 | 0.0250 | 0.0500 | 0.0333 | 0.1000 | 0.2000 | 0.1333 | 0.0153 +/- 0.0228 | 0.0333 |
| audio | 40 | 0.0750 | 0.1500 | 0.1000 | 0.2750 | 0.5500 | 0.3667 | 0.0167 +/- 0.0224 | 0.0333 |
| speech | 40 | 0.0250 | 0.0500 | 0.0333 | 0.1250 | 0.2500 | 0.1667 | 0.0147 +/- 0.0202 | 0.0333 |
| equal_fusion | 40 | 0.0500 | 0.1000 | 0.0667 | 0.2250 | 0.4500 | 0.3000 | 0.0150 +/- 0.0228 | 0.0667 |

## Preregistered audio question

Answer: yes. At the preregistered k=10, audio strict F1 was 0.0667; the random mean was 0.0087.
