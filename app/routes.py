import sys
import subprocess
import threading
import re
import os
from urllib.parse import urlsplit
from flask import Blueprint, request, jsonify, current_app
from app import db
from app.model import VideoJob, TrainingLog

main_bp = Blueprint('main', __name__)


def resolve_project_path(path):
    """Resolve relative input paths consistently from the repository root."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.abspath(os.path.join(project_root, path))


def resolve_stream_url(source):
    """Use yt-dlp to extract a direct playable stream link if given a webpage URL."""
    if source.startswith(('http://', 'https://')) and not source.endswith(('.mp4', '.m3u8', '.ts')):
        try:
            print(f"[INFO] Resolving stream URL via yt-dlp: {source}")
            result = subprocess.run(
                ["yt-dlp", "-g", source],
                capture_output=True,
                text=True,
                check=True
            )
            direct_url = result.stdout.strip().split('\n')[0]
            if direct_url:
                print(f"[INFO] Successfully resolved direct stream: {direct_url}")
                return direct_url
        except Exception as e:
            print(f"[WARNING] yt-dlp resolution failed, passing original URL: {e}")
    return source


@main_bp.route('/', methods=['GET'])
def index():
    return jsonify({
        "message": "Welcome to the Autohighlight Pipeline API!",
        "status": "online",
        "phases": {
            "phase_1_replay": "POST /api/jobs/replay",
            "phase_2_train": "POST /api/model/train",
            "phase_3_live": "POST /api/jobs/live"
        }
    }), 200

# --- BACKGROUND WORKER FOR PIPELINE & TRAINING ---

def run_pipeline_task(app_context, job_id, script_name, target_path, output_dir=None):
    """Run a pipeline while persisting progress emitted by its stdout."""
    with app_context:
        job = VideoJob.query.get(job_id)
        try:
            job.status = 'processing'
            job.progress = 0
            job.progress_message = 'Starting video processing'
            db.session.commit()

            # Resolve web URLs to direct playable streams using yt-dlp before executing main.py
            processed_target_path = resolve_stream_url(target_path)

            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            script_path = os.path.join(project_root, script_name)
            
            # Build command with resolved target path and optional output directory
            cmd = [sys.executable, script_path, processed_target_path]
            if output_dir:
                cmd.extend(["--output_dir", resolve_project_path(output_dir)])

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=project_root,
            )
            output_lines = []

            for line in process.stdout:
                output_lines.append(line)
                progress_match = re.match(r'\[PROGRESS\]\s+(\d+)', line)
                frame_match = re.match(r'\[FRAME\]\s+(\d+)', line)

                if progress_match:
                    progress = max(0, min(100, int(progress_match.group(1))))
                    if progress != job.progress:
                        job.progress = progress
                        job.progress_message = f'{progress}% complete'
                        db.session.commit()
                elif frame_match:
                    job.progress_message = (
                        f"Processed {frame_match.group(1)} frames"
                    )
                    db.session.commit()

            return_code = process.wait()
            if return_code != 0:
                details = ''.join(output_lines[-20:]).strip()
                raise RuntimeError(
                    f"Process failed with exit code {return_code}: {details}"
                )

            job.status = 'completed'
            job.progress = 100
            job.progress_message = 'Processing complete'
        except Exception as e:
            job.status = 'failed'
            job.error_message = str(e)
            job.progress_message = 'Processing failed'
        
        db.session.commit()

def run_training_task(app_context, log_id, match_name, ground_truth_path):
    """Background worker for fine-tuning model weights using training_weights.py."""
    with app_context:
        log_entry = TrainingLog.query.get(log_id)
        try:
            cmd = [sys.executable, "training_weights.py", "--gt_path", ground_truth_path]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                raise Exception(f"Training failed: {result.stderr}")

            log_entry.status = 'completed'
            log_entry.weights_version = f"v_updated_{log_id}"
        except Exception as e:
            log_entry.status = 'failed'
        
        db.session.commit()

# --- API ENDPOINTS ---

@main_bp.route('/api/jobs/replay', methods=['POST'])
def start_replay():
    """Phase 1: Process a full-match replay file or URL."""
    data = request.get_json()
    if not data or 'video_path' not in data:
        return jsonify({"error": "Missing 'video_path' in JSON body"}), 400

    video_input = data['video_path']
    output_dir = data.get('output_dir', './outputs')

    if not isinstance(video_input, str) or not video_input.strip():
        return jsonify({"error": "'video_path' must be a non-empty path or URL"}), 400

    video_input = video_input.strip()
    parsed_url = urlsplit(video_input)
    
    is_url = parsed_url.scheme.lower() in {'http', 'https'} and parsed_url.netloc

    if not is_url:
        resolved_path = resolve_project_path(video_input)
        if not os.path.isfile(resolved_path):
            return jsonify({
                "error": f"Video file not found: {resolved_path}",
                "hint": "Use an existing absolute path, relative path, or a valid stream URL.",
            }), 400
        target_source = resolved_path
    else:
        target_source = video_input

    job = VideoJob(job_type='replay', source_path=target_source, status='queued')
    db.session.add(job)
    db.session.commit()

    app_context = current_app.app_context()
    thread = threading.Thread(target=run_pipeline_task, args=(app_context, job.id, "main.py", target_source, output_dir))
    thread.start()

    return jsonify({"message": "Replay job queued successfully.", "job": job.to_dict()}), 201


@main_bp.route('/api/model/train', methods=['POST'])
def train_model():
    """Phase 2: Train model parameters using ground-truth highlights."""
    data = request.get_json()
    if not data or 'match_name' not in data or 'ground_truth_path' not in data:
        return jsonify({"error": "Missing 'match_name' or 'ground_truth_path' in JSON body"}), 400

    match_name = data['match_name']
    gt_path = data['ground_truth_path']

    log_entry = TrainingLog(match_name=match_name, ground_truth_path=gt_path, status='training')
    db.session.add(log_entry)
    db.session.commit()

    app_context = current_app.app_context()
    thread = threading.Thread(target=run_training_task, args=(app_context, log_entry.id, match_name, gt_path))
    thread.start()

    return jsonify({"message": "Model training started in background.", "training_log": log_entry.to_dict()}), 201


@main_bp.route('/api/jobs/live', methods=['POST'])
def start_live_stream():
    """Phase 3: Extract highlights from an active live stream."""
    data = request.get_json()
    if not data or 'stream_url' not in data:
        return jsonify({"error": "Missing 'stream_url' in JSON body"}), 400

    stream_url = data['stream_url']
    output_dir = data.get('output_dir', './outputs')

    if not isinstance(stream_url, str) or not stream_url.strip():
        return jsonify({"error": "'stream_url' must be a non-empty HTTP(S) URL"}), 400
    stream_url = stream_url.strip()
    parsed_url = urlsplit(stream_url)
    if parsed_url.scheme.lower() not in {'http', 'https'} or not parsed_url.netloc:
        return jsonify({"error": "'stream_url' must be a valid HTTP(S) URL"}), 400

    job = VideoJob(job_type='live', source_path=stream_url, status='queued')
    db.session.add(job)
    db.session.commit()

    app_context = current_app.app_context()
    thread = threading.Thread(target=run_pipeline_task, args=(app_context, job.id, "main.py", stream_url, output_dir))
    thread.start()

    return jsonify({"message": "Live stream highlight extraction queued.", "job": job.to_dict()}), 201


@main_bp.route('/api/jobs', methods=['GET'])
def get_jobs():
    jobs = VideoJob.query.order_by(VideoJob.created_at.desc()).all()
    return jsonify([job.to_dict() for job in jobs]), 200


@main_bp.route('/api/training/logs', methods=['GET'])
def get_training_logs():
    logs = TrainingLog.query.order_by(TrainingLog.created_at.desc()).all()
    return jsonify([log.to_dict() for log in logs]), 200
