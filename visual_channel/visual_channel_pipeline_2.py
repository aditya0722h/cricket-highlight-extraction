"""Compatibility entrypoint for the expected visual channel script name."""

from visual_channel_pipeline import run_visual_channel


if __name__ == "__main__":
    import os
    import sys

    if len(sys.argv) < 2:
        print("Usage: python visual_channel_pipeline_2.py <path_to_video>")
        sys.exit(1)

    video_path = sys.argv[1]
    output_filepath = os.path.join(
        os.path.dirname(video_path) or ".", "visual_output.json"
    )

    try:
        run_visual_channel(video_path, output_file=output_filepath)
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        sys.exit(1)
