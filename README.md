# Cricket Highlight Extraction

B.Tech final year project — multimodal AI system that automatically detects and extracts highlight clips from cricket match footage, optimized for low-VRAM hardware (target: NVIDIA RTX 2050, 4GB).

Due: 17th (this month)

## Team & Modules

| Module | Owner | Folder | What it outputs |
|---|---|---|---|
| A. Visual Channel | Aditya | `visual_channel/` | `V_spatial` (MobileNetV3 embeddings) + `V_motion` (Optical Flow) |
| B. Audio Channel | TBD | `audio_channel/` | STE, Spectral Centroid, ZCR |
| C. Speech / NLP | TBD | `speech_channel/` | Transcripts, sentiment, keyword/pause flags |
| D. Fusion + Deployment | TBD | `fusion/` | Score fusion, clip selection, rendering |

## The Schema — read this before writing any code

Every module outputs a JSON file: **a list of entries, one per timestamp**, so Module D can merge everyone's output by joining on `"t"`.

```json
[
  {"t": 0.0, "<your_feature_name>": <value>, ...},
  {"t": 1.0, "<your_feature_name>": <value>, ...}
]
```

- `t` = timestamp in **seconds**, matching the video's sparse sampling rate (1 FPS default, 2 FPS for fast action)
- Each module adds its own named fields alongside `t` — don't rename `t` itself, that's the join key
- Keep field names matching the project docs' notation where possible (`V_motion`, `STE`, etc.) so the fusion formula in the docs maps directly to code

### Example: Visual Channel output (already built)

```json
{"t": 5.0, "V_spatial": [0.12, -0.03, ...], "V_motion": 8.3}
```

- `V_spatial`: 576-dim embedding vector from MobileNetV3-Small (classifier head removed)
- `V_motion`: average Farneback Optical Flow magnitude between this frame and the previous one

### What Audio Channel should output

```json
{"t": 5.0, "STE": 0.42, "spectral_centroid": 1850.3, "ZCR": 0.11}
```

### What Speech/NLP Channel should output

```json
{"t": 5.0, "transcript_segment": "...", "sentiment_score": 0.8, "keyword_hit": true, "silence_after": false}
```

## Visual Channel — how to run it

```bash
cd visual_channel
python visual_channel_pipeline.py <path_to_video.mp4>
```

Outputs `visual_channel_output.json` in the same folder. Runs the full pipeline: frame extraction (1 FPS) → MobileNetV3 embeddings (FP16, GPU) → Farneback Optical Flow (CPU) → merge.

**Requirements:** Python 3.12, PyTorch with CUDA, OpenCV. See `visual_channel/` scripts for the individual pipeline stages if you want to understand or modify any single step.

## Fusion formula (reference, from project docs)

```
S_total(t) = wv · V_motion_norm + ws · V_spatial_norm + wa · STE_norm + wt · Speech_norm + wc · S_clip_norm
```

Weights vary by video genre — see the architecture docs for the sports/cricket weighting table.

## Deadline checklist

- [x] Visual Channel — built, validated on real cricket footage
- [ ] Audio Channel
- [ ] Speech/NLP Channel
- [ ] Fusion + clip selection
- [ ] Integration test (all 4 modules merged)
- [ ] Final render + demo prep
