# Evaluation and measured results

## Data and interpretation

The evaluation inputs were three supplied full-match videos (two cricket, one football) and their official highlight reels. Videos and processed channel outputs are local, git-ignored data; they are not included in this repository. The initial evaluation used pseudo ground truth only; human-labeled intervals (20 per match, single labeler) were later created for the two cricket matches and are reported in the [Human-labeled evaluation](#human-labeled-evaluation) section. The pseudo-label scores below measure agreement with **pseudo ground truth**, not verified highlight accuracy.

`auto_generated_ground_truth.py` compares MobileNet spatial embeddings sampled at about 1 FPS, uses cosine similarity >= 0.92, requires at least three consecutive matched frames, and merges resulting intervals. It identified:

| Match | Candidate clips matched | Candidate clips unmatched | Merged pseudo-label intervals |
|---|---:|---:|---:|
| Cricket match 1 | 71 | 287 | 68 |
| Cricket match 2 | 78 | 264 | 67 |
| Football match 1 | 68 | 239 | 61 |

These are algorithmic clip candidates. Unmatched reel segments can result from edits, transitions, replay footage, or scenes not visible in the full match; matches are not manually verified. Treat the intervals as weak pseudo-labels only.

## Human-labeled evaluation

Two cricket matches were independently labeled by a single human labeler (20 highlight intervals per match, labeled before seeing any model output). Matching uses IoU >= 0.5 with strict one-to-one maximum matching. A clearly labeled looser metric allows a predicted window to overlap a truth interval expanded by ±10 seconds, also one-to-one.

### Original pipeline

The original pipeline (fixed six-second windows, global min-max fusion) produced near-zero strict F1 on both matches with human labels. At k = 3 all variants scored strict F1 = 0.0000 on both matches; at k = 10 all variants (audio_only, speech_only, visual_only, full_fusion) also scored F1 = 0.0000 on both matches. Full table: [`eval_at_k.md`](../results/eval_at_k.md), human_verified section.

### Channel diagnostic: ROC-AUC with 95% bootstrap CIs

Per-second channel scores ranked against human labels. Bootstrap: 1000 resamples, fixed seed, labeled intervals sampled with replacement.

| Channel | Match 1 AUC [95% CI] | Match 2 AUC [95% CI] | Pooled AUC [95% CI] |
|---|---|---|---|
| Visual | 0.4785 [0.4398, 0.5159] | 0.5828 [0.5390, 0.6297] | 0.5292 [0.5003, 0.5568] |
| Audio | 0.6576 [0.6178, 0.6947] | 0.6524 [0.6059, 0.6948] | 0.6554 [0.6267, 0.6828] |
| Speech | 0.4938 [0.4508, 0.5318] | 0.6103 [0.5677, 0.6561] | 0.5544 [0.5211, 0.5852] |

### Peak-based selection — match 2 confirmation

Local rolling z-score normalization (120 s window), 5 s smoothing, 30 s peak separation, variable 8–30 s windows. These defaults were fixed before the match-2 run. Peak selection was developed and diagnosed on match 1; match 2 was an untouched confirmation set.

**Match 2 (confirmation) — audio peak selection with baselines:**

| Variant | k | Strict P | Strict R | Strict F1 | Looser P | Looser R | Looser F1 | Random strict F1 | Even strict F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| audio | 5 | 0.0000 | 0.0000 | 0.0000 | 0.8000 | 0.2000 | 0.3200 | 0.0040 +/- 0.0174 | 0.0000 |
| audio | 10 | 0.1000 | 0.0500 | 0.0667 | 0.7000 | 0.3500 | 0.4667 | 0.0087 +/- 0.0224 | 0.0000 |
| audio | 20 | 0.1000 | 0.1000 | 0.1000 | 0.5000 | 0.5000 | 0.5000 | 0.0115 +/- 0.0222 | 0.0000 |
| audio | 40 | 0.0750 | 0.1500 | 0.1000 | 0.2750 | 0.5500 | 0.3667 | 0.0167 +/- 0.0224 | 0.0333 |

The complete match-2 table (all variants at all k) is in [`confirm_match2.md`](../results/confirm_match2.md).

### Interpretation

Only the audio channel (short-time energy) carries a replicated ranking signal (ROC-AUC 0.6576 on match 1 and 0.6524 on match 2). Visual and speech were inconsistent and are not claimed as signals. Audio peak selection was above the random baseline on match 2 at k=10, 20 and 40 on strict IoU matching, but absolute performance is low (at k=10 it is a single matched interval: F1 0.0667 vs random mean 0.0087). The audio signal was discovered on match 1 labels, so match 1 results are optimistic; match 2 is the untouched confirmation set. This is evidence of a weak but replicated audio signal, not a reliable highlight detector.

### Notes on these results

- F1 bootstrap CIs are not reported because resampling intervals can only lower matches, so the interval's upper bound equals the point estimate; AUC CIs are reported instead.
- The looser ±10 s metric has no random baseline computed yet; do not quote it as evidence of beating chance.
- Results files were generated at commit `822c708f...`, before the final code commit `e054022` and results commit `65bbec7`.
- Single labeler, 20 intervals per match, two matches, no football human labels, no tuning on match 2.

## Evaluation method and ablations (pseudo-labels)

[`evaluate.py`](../evaluate.py) was run with the three full-match channel outputs and the corresponding pseudo-label intervals. It leaves scoring logic unchanged. A prediction and annotation are a match at intersection-over-union >= 0.5; maximum one-to-one interval matching determines true positives. A positive timestamp row represents `[t, t + 1)` seconds. Every ablation selects at most three non-overlapping six-second windows:

| Variant | Visual | Audio | Speech |
|---|---:|---:|---:|
| Audio only | 0 | 1 | 0 |
| Speech only | 0 | 0 | 1 |
| Visual only | 1 | 0 | 0 |
| Full fusion | 0.4 | 0.4 | 0.2 |

| Match | Variant | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| Cricket 1 | Audio only | 0.0000 | 0.0000 | 0.0000 |
| Cricket 1 | Speech only | 0.0000 | 0.0000 | 0.0000 |
| Cricket 1 | Visual only | 0.0000 | 0.0000 | 0.0000 |
| Cricket 1 | Full fusion | 0.0000 | 0.0000 | 0.0000 |
| Cricket 2 | Audio only | 0.0000 | 0.0000 | 0.0000 |
| Cricket 2 | Speech only | 0.6667 | 0.0299 | 0.0571 |
| Cricket 2 | Visual only | 0.0000 | 0.0000 | 0.0000 |
| Cricket 2 | Full fusion | 0.3333 | 0.0149 | 0.0286 |
| Football 1 | Audio only | 0.3333 | 0.0164 | 0.0313 |
| Football 1 | Speech only | 0.3333 | 0.0164 | 0.0313 |
| Football 1 | Visual only | 0.0000 | 0.0000 | 0.0000 |
| Football 1 | Full fusion | 0.3333 | 0.0164 | 0.0313 |

The low recall is also affected by the evaluation's top-3 prediction cap versus 61–68 pseudo-label intervals. These scores should not be read as evidence of general sports highlight accuracy.

Machine-readable inputs and reports:

- Cricket 1: [`ground truth`](../results/cricket_match_1_ground_truth.json), [`matching report`](../results/cricket_match_1_ground_truth_report.json), [`evaluation`](../results/cricket_match_1_evaluation.json).
- Cricket 2: [`ground truth`](../results/cricket_match_2_ground_truth.json), [`matching report`](../results/cricket_match_2_ground_truth_report.json), [`evaluation`](../results/cricket_match_2_evaluation.json).
- Football 1: [`ground truth`](../results/football_match_1_ground_truth.json), [`matching report`](../results/football_match_1_ground_truth_report.json), [`evaluation`](../results/football_match_1_evaluation.json).

## Cricket weight-learning experiment

Weights were trained on all of cricket match 1 and evaluated on the different cricket match 2; no seconds from a match were split across train and test. Grid search evaluated 231 simplex candidates at 0.05 increments, selecting by training-match F1. The logistic regression uses per-second channel features and balanced class weights.

| Method | Weights (visual, audio, speech) | Test precision | Test recall | Test F1 |
|---|---|---:|---:|---:|
| Hand-set baseline | (0.4, 0.4, 0.2) | 0.3333 | 0.0149 | 0.0286 |
| Simplex grid search | (0.0, 0.4, 0.6) | 0.6667 | 0.0299 | 0.0571 |
| Logistic regression | Learned coefficients | 0.0000 | 0.0000 | 0.0000 |

This is one train/test split over two cricket matches and uses pseudo-labels. It does not establish generalization. There is only one football match, so no held-out football weight experiment was run. See [`weight_learning_cricket.json`](../results/weight_learning_cricket.json).

## Runtime and hardware

Elapsed times are per-channel subprocess wall-clock time. Peak memory is sampled process-tree resident set size in host RAM, not GPU VRAM. Visual match 1 used the earlier `cap.read` frame-sampling path; subsequent visual timings use `grab/retrieve` to skip decoding unsampled frames. Sampled frames were verified pixel-identical, but the timing is not directly comparable.

| Match | Channel | Wall seconds | Peak RSS (MiB) | Seconds per video minute |
|---|---|---:|---:|---:|
| Cricket 1 (206.27 min) | Visual | 2482.184 | 712.54 | 12.034 |
|  | Audio | 63.730 | 239.79 | 0.309 |
|  | Speech | 410.616 | 1634.68 | 1.991 |
| Cricket 2 (194.59 min) | Visual | 1743.494 | 705.52 | 8.960 |
|  | Audio | 48.513 | 241.07 | 0.249 |
|  | Speech | 392.260 | 1065.10 | 2.016 |
| Football 1 (157.00 min) | Visual | 1620.226 | 660.20 | 10.320 |
|  | Audio | 41.404 | 236.50 | 0.264 |
|  | Speech | 416.031 | 1079.97 | 2.650 |

Observed environment: Windows 11, Python 3.14.3, NVIDIA GeForce RTX 2050 (4 GiB VRAM), CUDA available, 6 physical/12 logical CPU cores, and 7.73 GiB system RAM. Visual and speech used CUDA when available; audio processing used CPU. Runs used faster-whisper `small`.

The official cricket highlights reel produced three full-fusion windows totaling 18 seconds (2.034% of its 885.121-second duration). The official football highlights reel produced three windows totaling 18 seconds (2.514% of its 716-second duration). These are **sanity checks only**, not accuracy measurements. Details: [`cricket sanity`](../results/sanity_cricket_highlights.json), [`football sanity`](../results/sanity_football_highlights.json).

Consolidated machine-readable benchmark: [`runtime.json`](../results/runtime.json).

## Tests

- `python -m pytest -q` — **PASS**, 7 tests.
- `python test_pipeline.py`: synthetic integration smoke test; it checks execution/export, not detection accuracy.

## Reproduction commands

Run from the repository root with the supplied videos already placed under the ignored `data/videos/` directory. The benchmark reuses valid existing channel outputs; use `--force` to recompute.

```powershell
python benchmark_channels.py --video data\videos\cricket_full_match.mp4 --output-dir data\processed\cricket_full_match --model-size small
python benchmark_channels.py --video data\videos\cricket_full_match_2.mp4 --output-dir data\processed\cricket_full_match_2 --model-size small
python benchmark_channels.py --video data\videos\football_full_match.mp4 --output-dir data\processed\football_full_match --model-size small
python benchmark_channels.py --video data\videos\cricket_official_highlights.mp4 --output-dir data\processed\cricket_official_highlights --model-size small
python benchmark_channels.py --video data\videos\football_official_highlights.mp4 --output-dir data\processed\football_official_highlights --model-size small
```

Pseudo-label generation and full-match ablation commands:

```powershell
python auto_generated_ground_truth.py --full-visual-json data\processed\cricket_full_match\visual_output.json --highlight-visual-json data\processed\cricket_official_highlights\visual_output.json --output results\cricket_match_1_ground_truth.json --report results\cricket_match_1_ground_truth_report.json
python evaluate.py --sport cricket --visual data\processed\cricket_full_match\visual_output.json --audio data\processed\cricket_full_match\audio_output.json --speech data\processed\cricket_full_match\speech_output.json --ground-truth results\cricket_match_1_ground_truth.json --output results\cricket_match_1_evaluation.json
python auto_generated_ground_truth.py --full-visual-json data\processed\cricket_full_match_2\visual_output.json --highlight-visual-json data\processed\cricket_official_highlights_2\visual_output.json --output results\cricket_match_2_ground_truth.json --report results\cricket_match_2_ground_truth_report.json
python evaluate.py --sport cricket --visual data\processed\cricket_full_match_2\visual_output.json --audio data\processed\cricket_full_match_2\audio_output.json --speech data\processed\cricket_full_match_2\speech_output.json --ground-truth results\cricket_match_2_ground_truth.json --output results\cricket_match_2_evaluation.json
python auto_generated_ground_truth.py --full-visual-json data\processed\football_full_match\visual_output.json --highlight-visual-json data\processed\football_official_highlights\visual_output.json --output results\football_match_1_ground_truth.json --report results\football_match_1_ground_truth_report.json
python evaluate.py --sport football --visual data\processed\football_full_match\visual_output.json --audio data\processed\football_full_match\audio_output.json --speech data\processed\football_full_match\speech_output.json --ground-truth results\football_match_1_ground_truth.json --output results\football_match_1_evaluation.json
```

The single-match sanity checks omit ground truth:

```powershell
python evaluate.py --sport cricket --visual data\processed\cricket_official_highlights\visual_output.json --audio data\processed\cricket_official_highlights\audio_output.json --speech data\processed\cricket_official_highlights\speech_output.json --output results\sanity_cricket_highlights.json
python evaluate.py --sport football --visual data\processed\football_official_highlights\visual_output.json --audio data\processed\football_official_highlights\audio_output.json --speech data\processed\football_official_highlights\speech_output.json --output results\sanity_football_highlights.json
```

Weight learning (entire cricket match 1 for training, match 2 for testing):

```powershell
python training_weights.py --sport cricket --train-match-id cricket_match_1 --test-match-id cricket_match_2 --train-visual data\processed\cricket_full_match\visual_output.json --train-audio data\processed\cricket_full_match\audio_output.json --train-speech data\processed\cricket_full_match\speech_output.json --train-ground-truth results\cricket_match_1_ground_truth.json --test-visual data\processed\cricket_full_match_2\visual_output.json --test-audio data\processed\cricket_full_match_2\audio_output.json --test-speech data\processed\cricket_full_match_2\speech_output.json --test-ground-truth results\cricket_match_2_ground_truth.json --grid-resolution 20 --top-k 3 --window-sec 6 --overlap-iou 0.5 --output results\weight_learning_cricket.json
```

The benchmarked speech implementation bounds memory while retaining a single transcription pass. A comparison on the cricket highlights reel reproduced the previous speech output exactly. Videos, channel files, model weights, and generated visual frames remain excluded by `.gitignore`.

## Peak-based selection

[`peak_fusion.py`](../fusion_channel/peak_fusion.py) adds a separate local-normalization experiment without modifying the original fusion engine or scorer. Defaults were fixed before reviewing human labels: 120-second centered rolling z-score normalization clipped to [-3, 3], 5-second smoothing, equal weights, 30-second peak separation, and variable 8--30 second windows. The complete strict, looser, random, evenly-spaced, old-method, and sensitivity tables are in [`peak_evaluation.md`](../results/peak_evaluation.md); detailed values are in [`peak_evaluation.json`](../results/peak_evaluation.json).

### Human-verified: cricket_match_1, 20 intervals, single labeler

| Variant | k | New strict F1 | Old strict F1 | New loose F1 | Old loose F1 |
|---|---:|---:|---:|---:|---:|
| visual | 10 | 0.0000 | 0.0000 | 0.0667 | 0.1333 |
| audio | 10 | 0.2667 | 0.0000 | 0.3333 | 0.1333 |
| speech | 10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| equal_fusion | 10 | 0.0667 | 0.0000 | 0.1333 | 0.0667 |
| visual | 20 | 0.0000 | 0.0000 | 0.1000 | 0.1000 |
| audio | 20 | 0.2000 | 0.1000 | 0.3000 | 0.2000 |
| speech | 20 | 0.0500 | 0.0000 | 0.0500 | 0.0500 |
| equal_fusion | 20 | 0.0500 | 0.0000 | 0.1500 | 0.0500 |
| visual | 40 | 0.0000 | 0.0000 | 0.1333 | 0.1000 |
| audio | 40 | 0.1667 | 0.0667 | 0.3000 | 0.1333 |
| speech | 40 | 0.0333 | 0.0000 | 0.1333 | 0.0667 |
| equal_fusion | 40 | 0.0333 | 0.0000 | 0.1333 | 0.1333 |
| visual | 60 | 0.0000 | 0.0000 | 0.1500 | 0.1250 |
| audio | 60 | 0.1250 | 0.0500 | 0.2250 | 0.1250 |
| speech | 60 | 0.0250 | 0.0250 | 0.1500 | 0.0750 |
| equal_fusion | 60 | 0.1000 | 0.0000 | 0.1750 | 0.1000 |

### Pseudo-label results

The full pseudo-label table for all three matches, all four variants, and every k is in [`peak_evaluation.md`](../results/peak_evaluation.md). The separate channel signal report is [`channel_signal.md`](../results/channel_signal.md). The human set is one manually labeled match with 20 intervals and one labeler; no hyperparameters were tuned on it. The sensitivity table contains every 60/120/240-second rolling setting crossed with every 3/5/10-second smoothing setting, without choosing a winner.

Interpretation: audio's human-label ROC-AUC is 0.6576; visual (0.4785), speech (0.4938), and the original fused score (0.4902) do not show comparable ranking signal. The new audio peak selector exceeds its random strict-F1 mean at every reported k. Equal fusion exceeds its random strict-F1 mean at k=10, 20, 40, and 60, but this single-labeler, small-sample result is not evidence of reliable generalization.

## Evaluation at k

[`eval_at_k.py`](../eval_at_k.py) reads the existing full-match channel JSONs and evaluates the unchanged fusion selector at 6-second windows with k = 3, 10, 20, 40, 60, and 80. It uses the same IoU >= 0.5 maximum one-to-one interval matching as [`evaluate.py`](../evaluate.py). The complete generated table, with rows for every match, variant, and k, is in [`results/eval_at_k.md`](../results/eval_at_k.md), with machine-readable values in [`results/eval_at_k.json`](../results/eval_at_k.json). Random-baseline precision is the mean +/- population standard deviation across 100 deterministic selections using the fusion selector's 3-second spacing rule.

| Match | Best full-fusion k | Full-fusion precision | Full-fusion recall | Full-fusion F1 | Random-baseline precision |
|---|---:|---:|---:|---:|---:|
| cricket_match_1 | 80 | 0.1250 | 0.1471 | 0.1351 | 0.0166 +/- 0.0140 |
| cricket_match_2 | 60 | 0.1167 | 0.1045 | 0.1102 | 0.0150 +/- 0.0159 |
| football_match_1 | 80 | 0.1375 | 0.1803 | 0.1560 | 0.0227 +/- 0.0161 |

Interpretation: evaluating beyond k = 3 removes the mechanical recall ceiling, and several variants have precision above the random-baseline mean. These figures remain pseudo-label results, not independently verified event detection. A file made with [`label_tool.py`](../label_tool.py) is evaluated separately and labelled human-verified, with its actual interval count and the limitation that it is a single-labeler set.
