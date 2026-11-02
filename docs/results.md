# Evaluation and measured results

## Data and interpretation

The evaluation inputs were three supplied full-match videos (two cricket, one football) and their official highlight reels. Videos and processed channel outputs are local, git-ignored data; they are not included in this repository. No human-annotated match ground truth was available. All match-level scores below therefore measure agreement with **pseudo ground truth**, not verified highlight accuracy.

`auto_generated_ground_truth.py` compares MobileNet spatial embeddings sampled at about 1 FPS, uses cosine similarity >= 0.92, requires at least three consecutive matched frames, and merges resulting intervals. It identified:

| Match | Candidate clips matched | Candidate clips unmatched | Merged pseudo-label intervals |
|---|---:|---:|---:|
| Cricket match 1 | 71 | 287 | 68 |
| Cricket match 2 | 78 | 264 | 67 |
| Football match 1 | 68 | 239 | 61 |

These are algorithmic clip candidates. Unmatched reel segments can result from edits, transitions, replay footage, or scenes not visible in the full match; matches are not manually verified. Treat the intervals as weak pseudo-labels only.

## Evaluation method and ablations

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
