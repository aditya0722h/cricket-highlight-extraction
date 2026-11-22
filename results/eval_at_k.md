# Evaluation at k

Run date: 2026-11-22T14:02:21.045564+05:30  
Git commit: `822c708f67b8e11562f9649aed52dad9e3d26af2`

Model metrics use the project's IoU ≥ 0.5 maximum one-to-one matching. Random results are the mean over 100 deterministic seeds; `±` is population standard deviation.

## pseudo_labels: pseudo-labels

| Match | Variant | k | Precision | Recall | F1 | Random-baseline precision |
|---|---|---:|---:|---:|---:|---:|
| cricket_match_1 | audio_only | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0267 ± 0.0904 |
| cricket_match_1 | speech_only | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0267 ± 0.0904 |
| cricket_match_1 | visual_only | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0267 ± 0.0904 |
| cricket_match_1 | full_fusion | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0267 ± 0.0904 |
| cricket_match_1 | audio_only | 10 | 0.3000 | 0.0441 | 0.0769 | 0.0170 ± 0.0401 |
| cricket_match_1 | speech_only | 10 | 0.2000 | 0.0294 | 0.0513 | 0.0170 ± 0.0401 |
| cricket_match_1 | visual_only | 10 | 0.0000 | 0.0000 | 0.0000 | 0.0170 ± 0.0401 |
| cricket_match_1 | full_fusion | 10 | 0.1000 | 0.0147 | 0.0256 | 0.0170 ± 0.0401 |
| cricket_match_1 | audio_only | 20 | 0.1500 | 0.0441 | 0.0682 | 0.0210 ± 0.0310 |
| cricket_match_1 | speech_only | 20 | 0.1000 | 0.0294 | 0.0455 | 0.0210 ± 0.0310 |
| cricket_match_1 | visual_only | 20 | 0.0000 | 0.0000 | 0.0000 | 0.0210 ± 0.0310 |
| cricket_match_1 | full_fusion | 20 | 0.1500 | 0.0441 | 0.0682 | 0.0210 ± 0.0310 |
| cricket_match_1 | audio_only | 40 | 0.1500 | 0.0882 | 0.1111 | 0.0178 ± 0.0198 |
| cricket_match_1 | speech_only | 40 | 0.1000 | 0.0588 | 0.0741 | 0.0178 ± 0.0198 |
| cricket_match_1 | visual_only | 40 | 0.0000 | 0.0000 | 0.0000 | 0.0178 ± 0.0198 |
| cricket_match_1 | full_fusion | 40 | 0.1000 | 0.0588 | 0.0741 | 0.0178 ± 0.0198 |
| cricket_match_1 | audio_only | 60 | 0.1667 | 0.1471 | 0.1562 | 0.0178 ± 0.0164 |
| cricket_match_1 | speech_only | 60 | 0.0667 | 0.0588 | 0.0625 | 0.0178 ± 0.0164 |
| cricket_match_1 | visual_only | 60 | 0.0000 | 0.0000 | 0.0000 | 0.0178 ± 0.0164 |
| cricket_match_1 | full_fusion | 60 | 0.1167 | 0.1029 | 0.1094 | 0.0178 ± 0.0164 |
| cricket_match_1 | audio_only | 80 | 0.1250 | 0.1471 | 0.1351 | 0.0166 ± 0.0140 |
| cricket_match_1 | speech_only | 80 | 0.0750 | 0.0882 | 0.0811 | 0.0166 ± 0.0140 |
| cricket_match_1 | visual_only | 80 | 0.0000 | 0.0000 | 0.0000 | 0.0166 ± 0.0140 |
| cricket_match_1 | full_fusion | 80 | 0.1250 | 0.1471 | 0.1351 | 0.0166 ± 0.0140 |
| cricket_match_2 | audio_only | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0100 ± 0.0569 |
| cricket_match_2 | speech_only | 3 | 0.6667 | 0.0299 | 0.0571 | 0.0100 ± 0.0569 |
| cricket_match_2 | visual_only | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0100 ± 0.0569 |
| cricket_match_2 | full_fusion | 3 | 0.3333 | 0.0149 | 0.0286 | 0.0100 ± 0.0569 |
| cricket_match_2 | audio_only | 10 | 0.0000 | 0.0000 | 0.0000 | 0.0130 ± 0.0336 |
| cricket_match_2 | speech_only | 10 | 0.2000 | 0.0299 | 0.0519 | 0.0130 ± 0.0336 |
| cricket_match_2 | visual_only | 10 | 0.0000 | 0.0000 | 0.0000 | 0.0130 ± 0.0336 |
| cricket_match_2 | full_fusion | 10 | 0.1000 | 0.0149 | 0.0260 | 0.0130 ± 0.0336 |
| cricket_match_2 | audio_only | 20 | 0.0500 | 0.0149 | 0.0230 | 0.0145 ± 0.0248 |
| cricket_match_2 | speech_only | 20 | 0.1000 | 0.0299 | 0.0460 | 0.0145 ± 0.0248 |
| cricket_match_2 | visual_only | 20 | 0.1000 | 0.0299 | 0.0460 | 0.0145 ± 0.0248 |
| cricket_match_2 | full_fusion | 20 | 0.1000 | 0.0299 | 0.0460 | 0.0145 ± 0.0248 |
| cricket_match_2 | audio_only | 40 | 0.1500 | 0.0896 | 0.1121 | 0.0168 ± 0.0197 |
| cricket_match_2 | speech_only | 40 | 0.0500 | 0.0299 | 0.0374 | 0.0168 ± 0.0197 |
| cricket_match_2 | visual_only | 40 | 0.0750 | 0.0448 | 0.0561 | 0.0168 ± 0.0197 |
| cricket_match_2 | full_fusion | 40 | 0.1250 | 0.0746 | 0.0935 | 0.0168 ± 0.0197 |
| cricket_match_2 | audio_only | 60 | 0.1500 | 0.1343 | 0.1417 | 0.0150 ± 0.0159 |
| cricket_match_2 | speech_only | 60 | 0.0333 | 0.0299 | 0.0315 | 0.0150 ± 0.0159 |
| cricket_match_2 | visual_only | 60 | 0.0667 | 0.0597 | 0.0630 | 0.0150 ± 0.0159 |
| cricket_match_2 | full_fusion | 60 | 0.1167 | 0.1045 | 0.1102 | 0.0150 ± 0.0159 |
| cricket_match_2 | audio_only | 80 | 0.1750 | 0.2090 | 0.1905 | 0.0154 ± 0.0146 |
| cricket_match_2 | speech_only | 80 | 0.0250 | 0.0299 | 0.0272 | 0.0154 ± 0.0146 |
| cricket_match_2 | visual_only | 80 | 0.0500 | 0.0597 | 0.0544 | 0.0154 ± 0.0146 |
| cricket_match_2 | full_fusion | 80 | 0.1000 | 0.1194 | 0.1088 | 0.0154 ± 0.0146 |
| football_match_1 | audio_only | 3 | 0.3333 | 0.0164 | 0.0312 | 0.0333 ± 0.1000 |
| football_match_1 | speech_only | 3 | 0.3333 | 0.0164 | 0.0312 | 0.0333 ± 0.1000 |
| football_match_1 | visual_only | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0333 ± 0.1000 |
| football_match_1 | full_fusion | 3 | 0.3333 | 0.0164 | 0.0312 | 0.0333 ± 0.1000 |
| football_match_1 | audio_only | 10 | 0.2000 | 0.0328 | 0.0563 | 0.0250 ± 0.0517 |
| football_match_1 | speech_only | 10 | 0.1000 | 0.0164 | 0.0282 | 0.0250 ± 0.0517 |
| football_match_1 | visual_only | 10 | 0.0000 | 0.0000 | 0.0000 | 0.0250 ± 0.0517 |
| football_match_1 | full_fusion | 10 | 0.2000 | 0.0328 | 0.0563 | 0.0250 ± 0.0517 |
| football_match_1 | audio_only | 20 | 0.2000 | 0.0656 | 0.0988 | 0.0235 ± 0.0335 |
| football_match_1 | speech_only | 20 | 0.0500 | 0.0164 | 0.0247 | 0.0235 ± 0.0335 |
| football_match_1 | visual_only | 20 | 0.0500 | 0.0164 | 0.0247 | 0.0235 ± 0.0335 |
| football_match_1 | full_fusion | 20 | 0.3000 | 0.0984 | 0.1481 | 0.0235 ± 0.0335 |
| football_match_1 | audio_only | 40 | 0.2000 | 0.1311 | 0.1584 | 0.0225 ± 0.0249 |
| football_match_1 | speech_only | 40 | 0.0250 | 0.0164 | 0.0198 | 0.0225 ± 0.0249 |
| football_match_1 | visual_only | 40 | 0.0250 | 0.0164 | 0.0198 | 0.0225 ± 0.0249 |
| football_match_1 | full_fusion | 40 | 0.1500 | 0.0984 | 0.1188 | 0.0225 ± 0.0249 |
| football_match_1 | audio_only | 60 | 0.1500 | 0.1475 | 0.1488 | 0.0213 ± 0.0189 |
| football_match_1 | speech_only | 60 | 0.0167 | 0.0164 | 0.0165 | 0.0213 ± 0.0189 |
| football_match_1 | visual_only | 60 | 0.0167 | 0.0164 | 0.0165 | 0.0213 ± 0.0189 |
| football_match_1 | full_fusion | 60 | 0.1500 | 0.1475 | 0.1488 | 0.0213 ± 0.0189 |
| football_match_1 | audio_only | 80 | 0.1500 | 0.1967 | 0.1702 | 0.0227 ± 0.0161 |
| football_match_1 | speech_only | 80 | 0.0250 | 0.0328 | 0.0284 | 0.0227 ± 0.0161 |
| football_match_1 | visual_only | 80 | 0.0250 | 0.0328 | 0.0284 | 0.0227 ± 0.0161 |
| football_match_1 | full_fusion | 80 | 0.1375 | 0.1803 | 0.1560 | 0.0227 ± 0.0161 |

## human_verified: human-verified, single labeler


### cricket_match_1: human-verified, 20 intervals, single labeler

| Match | Variant | k | Precision | Recall | F1 | Random-baseline precision |
|---|---|---:|---:|---:|---:|---:|
| cricket_match_1 | audio_only | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 ± 0.0000 |
| cricket_match_1 | speech_only | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 ± 0.0000 |
| cricket_match_1 | visual_only | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 ± 0.0000 |
| cricket_match_1 | full_fusion | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 ± 0.0000 |
| cricket_match_1 | audio_only | 10 | 0.0000 | 0.0000 | 0.0000 | 0.0020 ± 0.0140 |
| cricket_match_1 | speech_only | 10 | 0.0000 | 0.0000 | 0.0000 | 0.0020 ± 0.0140 |
| cricket_match_1 | visual_only | 10 | 0.0000 | 0.0000 | 0.0000 | 0.0020 ± 0.0140 |
| cricket_match_1 | full_fusion | 10 | 0.0000 | 0.0000 | 0.0000 | 0.0020 ± 0.0140 |
| cricket_match_1 | audio_only | 20 | 0.1000 | 0.1000 | 0.1000 | 0.0050 ± 0.0150 |
| cricket_match_1 | speech_only | 20 | 0.0000 | 0.0000 | 0.0000 | 0.0050 ± 0.0150 |
| cricket_match_1 | visual_only | 20 | 0.0000 | 0.0000 | 0.0000 | 0.0050 ± 0.0150 |
| cricket_match_1 | full_fusion | 20 | 0.0000 | 0.0000 | 0.0000 | 0.0050 ± 0.0150 |
| cricket_match_1 | audio_only | 40 | 0.0500 | 0.1000 | 0.0667 | 0.0043 ± 0.0094 |
| cricket_match_1 | speech_only | 40 | 0.0000 | 0.0000 | 0.0000 | 0.0043 ± 0.0094 |
| cricket_match_1 | visual_only | 40 | 0.0000 | 0.0000 | 0.0000 | 0.0043 ± 0.0094 |
| cricket_match_1 | full_fusion | 40 | 0.0000 | 0.0000 | 0.0000 | 0.0043 ± 0.0094 |
| cricket_match_1 | audio_only | 60 | 0.0333 | 0.1000 | 0.0500 | 0.0038 ± 0.0074 |
| cricket_match_1 | speech_only | 60 | 0.0167 | 0.0500 | 0.0250 | 0.0038 ± 0.0074 |
| cricket_match_1 | visual_only | 60 | 0.0000 | 0.0000 | 0.0000 | 0.0038 ± 0.0074 |
| cricket_match_1 | full_fusion | 60 | 0.0000 | 0.0000 | 0.0000 | 0.0038 ± 0.0074 |
| cricket_match_1 | audio_only | 80 | 0.0250 | 0.1000 | 0.0400 | 0.0034 ± 0.0058 |
| cricket_match_1 | speech_only | 80 | 0.0125 | 0.0500 | 0.0200 | 0.0034 ± 0.0058 |
| cricket_match_1 | visual_only | 80 | 0.0000 | 0.0000 | 0.0000 | 0.0034 ± 0.0058 |
| cricket_match_1 | full_fusion | 80 | 0.0000 | 0.0000 | 0.0000 | 0.0034 ± 0.0058 |

### cricket_match_2: human-verified, 20 intervals, single labeler

| Match | Variant | k | Precision | Recall | F1 | Random-baseline precision |
|---|---|---:|---:|---:|---:|---:|
| cricket_match_2 | audio_only | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0033 ± 0.0332 |
| cricket_match_2 | speech_only | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0033 ± 0.0332 |
| cricket_match_2 | visual_only | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0033 ± 0.0332 |
| cricket_match_2 | full_fusion | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0033 ± 0.0332 |
| cricket_match_2 | audio_only | 10 | 0.0000 | 0.0000 | 0.0000 | 0.0030 ± 0.0171 |
| cricket_match_2 | speech_only | 10 | 0.0000 | 0.0000 | 0.0000 | 0.0030 ± 0.0171 |
| cricket_match_2 | visual_only | 10 | 0.0000 | 0.0000 | 0.0000 | 0.0030 ± 0.0171 |
| cricket_match_2 | full_fusion | 10 | 0.0000 | 0.0000 | 0.0000 | 0.0030 ± 0.0171 |
| cricket_match_2 | audio_only | 20 | 0.0000 | 0.0000 | 0.0000 | 0.0030 ± 0.0119 |
| cricket_match_2 | speech_only | 20 | 0.0000 | 0.0000 | 0.0000 | 0.0030 ± 0.0119 |
| cricket_match_2 | visual_only | 20 | 0.0000 | 0.0000 | 0.0000 | 0.0030 ± 0.0119 |
| cricket_match_2 | full_fusion | 20 | 0.0500 | 0.0500 | 0.0500 | 0.0030 ± 0.0119 |
| cricket_match_2 | audio_only | 40 | 0.0000 | 0.0000 | 0.0000 | 0.0035 ± 0.0087 |
| cricket_match_2 | speech_only | 40 | 0.0000 | 0.0000 | 0.0000 | 0.0035 ± 0.0087 |
| cricket_match_2 | visual_only | 40 | 0.0000 | 0.0000 | 0.0000 | 0.0035 ± 0.0087 |
| cricket_match_2 | full_fusion | 40 | 0.0250 | 0.0500 | 0.0333 | 0.0035 ± 0.0087 |
| cricket_match_2 | audio_only | 60 | 0.0000 | 0.0000 | 0.0000 | 0.0038 ± 0.0074 |
| cricket_match_2 | speech_only | 60 | 0.0167 | 0.0500 | 0.0250 | 0.0038 ± 0.0074 |
| cricket_match_2 | visual_only | 60 | 0.0000 | 0.0000 | 0.0000 | 0.0038 ± 0.0074 |
| cricket_match_2 | full_fusion | 60 | 0.0167 | 0.0500 | 0.0250 | 0.0038 ± 0.0074 |
| cricket_match_2 | audio_only | 80 | 0.0000 | 0.0000 | 0.0000 | 0.0038 ± 0.0063 |
| cricket_match_2 | speech_only | 80 | 0.0125 | 0.0500 | 0.0200 | 0.0038 ± 0.0063 |
| cricket_match_2 | visual_only | 80 | 0.0125 | 0.0500 | 0.0200 | 0.0038 ± 0.0063 |
| cricket_match_2 | full_fusion | 80 | 0.0125 | 0.0500 | 0.0200 | 0.0038 ± 0.0063 |
