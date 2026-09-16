import sys
import os
import subprocess

def run_step(description, command):
    """Utility function to execute shell commands cleanly."""
    print(f"\n==========================================")
    print(f"▶ {description}")
    print(f"==========================================")
    result = subprocess.run(command)
    if result.returncode != 0:
        print(f" Error during: {description}")
        sys.exit(result.returncode)

def main():
    # 1. Validate Command Line Input
    if len(sys.argv) < 2:
        print("\n❌ Error: Missing video path argument.")
        print("Usage: python main.py \"<path_to_video.mp4>\"\n")
        sys.exit(1)

    video_path = sys.argv[1]

    if not os.path.exists(video_path):
        print(f"\n❌ Error: File '{video_path}' does not exist.")
        sys.exit(1)

    print(f"\n🚀 Starting Pipeline for video: {video_path}")

    # 2. Run Visual Channel Pipeline
    run_step(
        "Processing Visual Channel...",
        [sys.executable, "visual_channel/visual_channel_pipeline.py", video_path]
    )

    # 3. Run Audio Channel Pipeline
    run_step(
        "Processing Audio Channel...",
        [sys.executable, "audio_channel/audio_channel_pipeline.py", video_path]
    )

    # 4. Run Speech Channel Pipeline
    run_step(
        "Processing Speech Channel...",
        [sys.executable, "speech_channel/speech_channel_pipeline.py", video_path]
    )

    # 5. Run Fusion Pipeline to generate final highlights
    run_step(
        "Fusing Features and Generating Highlights...",
        [sys.executable, "fusion_pipeline.py", video_path]
    )

    print("\n==========================================")
    print("✅ Pipeline execution completed successfully!")
    print("📁 Check 'output_highlights/' for generated video clips.")
    print("==========================================\n")

if __name__ == "__main__":
    main()