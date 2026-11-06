# AutoHighlight portfolio site

This is a static HTML, CSS, and vanilla JavaScript site. It has no build step or runtime dependencies. The demo reads the model-generated JSON and MP4 files from `assets/`.

## Preview locally

From the repository root, start a static server:

```powershell
python -m http.server 8000 --directory site
```

Open <http://localhost:8000>. A local server is required because the page fetches the JSON files; opening `index.html` directly may block those requests.

## Deploy to GitHub Pages

The [Pages workflow](../.github/workflows/pages.yml) deploys the `/site` directory on pushes to `main` or `final-evaluation`, or when manually dispatched. Set **Settings → Pages → Build and deployment → Source** to **GitHub Actions**. The generated highlight clips are H.264 and compressed for static hosting. Source-video license status is not verified; confirm permission before public deployment.

## Update model output

The primary demo uses `assets/cricket-full-match-demo.json` and `assets/cricket-full-match-demo.mp4`. The two official-reel examples are secondary. Each metadata file lists its source input, segment offset when applicable, duration, and selected intervals with fused scores. The scores are ranking values, not calibrated probabilities. Add an attribution and verify footage rights before publishing any source video.

## Recompute the ablation

From the repository root, after the cricket match 1 channel outputs and pseudo-labels are available, run:

```powershell
python run_ablation.py
```

The script uses the same top-3, six-second-window, IoU 0.5 interval evaluation as `evaluate.py`. It writes a new `results/ablation.json` and copies the same data to `site/assets/ablation.json` for the static table. To choose another output path, pass `--output` and `--site-output`.

The results use automatically generated visual-similarity pseudo-labels, not human annotations. If reproducing from a clean checkout, first provide the ignored match videos/channel JSON and ground-truth JSON files.

For hit-rate versus selection budget and per-variant precision, recall, and F1 at `k = 3, 5, 10, 20, 50`, run:

```powershell
python run_ablation_k.py
```

This writes `results/ablation_k.json` and `site/assets/ablation_k.json`. Hit-rate counts selected windows with any positive-duration overlap with at least one pseudo-label interval; precision, recall, and F1 use the existing IoU 0.5 one-to-one matcher. These are pseudo-label comparisons, not human-verified accuracy.
