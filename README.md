Here is the updated `README.md` file. It reflects your current setup, including the Flask backend, Supabase PostgreSQL integration, local replay files, and your planned React + Vite + Tailwind frontend stack.

---

### Updated `README.md`

```markdown
# Cricket Highlight Extraction

B.Tech final year project — multimodal AI system that automatically detects and extracts highlight clips from cricket match footage, managed via a Flask backend and Supabase PostgreSQL database, optimized for low-VRAM hardware (target: NVIDIA RTX 2050, 4GB)[cite: 19].

Due: 12th (this month)[cite: 19]

## Architecture & Modules

| Module | Owner | Folder | What it outputs |
|---|---|---|---|
| A. Visual Channel | Aditya | `visual_channel/` | `V_spatial` (MobileNetV3 embeddings) + `V_motion` (Optical Flow)[cite: 19] |
| B. Audio Channel | TBD | `audio_channel/` | STE, Spectral Centroid, ZCR[cite: 19] |
| C. Speech / NLP | TBD | `speech_channel/` | Transcripts, sentiment, keyword/pause flags[cite: 19] |
| D. Fusion + Deployment | Team | Root / `fusion/` | Score fusion, clip selection, rendering, Flask API[cite: 19] |

## Tech Stack

* **Backend:** Flask, Supabase (PostgreSQL), Python, OpenCV, MoviePy, yt-dlp[cite: 19]
* **Frontend:** React, Vite, Tailwind CSS
* **Hardware Target:** NVIDIA RTX 2050 (4GB VRAM) optimized[cite: 19]

## Flask Backend & API Endpoints

The project includes a Flask backend that queues and processes highlight extraction and model fine-tuning jobs asynchronously with Supabase[cite: 19].

- **POST `/api/jobs/replay`**: Queue a local video file (e.g., `match.mp4`) for highlight extraction[cite: 19].
- **POST `/api/jobs/live`**: Queue a live stream URL (`.m3u8`) for processing[cite: 19].
- **POST `/api/model/train`**: Fine-tune or update detection thresholds using ground-truth JSON files[cite: 19].
- **GET `/api/jobs`**: Retrieve all recent job statuses, progress percentages, and error logs[cite: 19].
- **GET `/api/training/logs`**: Review model training history[cite: 19].

## How to Run the Pipeline Locally

1. **Start the Flask Server:**
   ```bash
   python run.py

```

2. **Trigger a Replay Job (PowerShell):**
```powershell
$body = @{ video_path = "match.mp4" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:5000/api/jobs/replay" -Method Post -ContentType "application/json" -Body $body

```


3. **Check Output Highlights:**
Processed highlight clips and the master reel will automatically be saved into the `output_highlights/` directory with synchronized audio.



## Deadline Checklist

* [x] Backend API & Supabase Integration — built and tested


* [x] Visual Channel / OpenCV Pipeline & Audio-Retained Export — built and validated on local video files


* [x] Real-time job progress tracking — implemented


* [ ] Audio Channel
* [ ] Speech/NLP Channel
* [ ] Fusion + clip selection refinement
* [ ] React + Vite Dashboard UI integration
* [ ] Final demo prep

```

<Elicitations message="What would you like to do next?">
  <Elicitation label="Create React frontend component" query="How do I write a React + Tailwind component to poll the /api/jobs endpoint and show live progress bars?"/>
  <Elicitation label="Update project synopsis PDF" query="Do you need help updating the final capstone synopsis document with these architecture details?"/>
</Elicitations>

```