from pathlib import Path
import shutil
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "test_data" / "synthetic_match.mp4"


def main() -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to generate the synthetic test video.")
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {OUTPUT_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=darkgreen:s=640x360:r=25:d=20",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=1000:sample_rate=44100:duration=20",
        "-vf",
        "drawbox=x='mod(t*100,iw-100)':y=130:w=100:h=100:color=white:t=fill",
        "-af",
        "volume='if(between(t,9,11),0.9,0)'",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(OUTPUT_PATH),
    ]
    subprocess.run(command, check=True)
    print(f"Created 20-second synthetic video with audio: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
