import json
import os

from fusion_channel.fusion_engine import render_highlights_ffmpeg, run_fusion_pipeline


def test_render_highlights_uses_new_filename_and_refuses_overwrite(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "output_highlights"
    output_dir.mkdir()
    existing_clip = output_dir / "highlight_01.mp4"
    existing_clip.write_bytes(b"previous clip")

    def fake_ffmpeg(command, **kwargs):
        assert "-n" in command
        assert "-y" not in command
        with open(command[-1], "wb") as clip:
            clip.write(b"new clip")

    monkeypatch.setattr("fusion_channel.fusion_engine.subprocess.run", fake_ffmpeg)

    rendered = render_highlights_ffmpeg(
        "match.mp4", [(1.0, 7.0)], str(output_dir)
    )

    assert existing_clip.read_bytes() == b"previous clip"
    assert os.path.basename(rendered[0]) == "highlight_01_2.mp4"
    assert open(rendered[0], "rb").read() == b"new clip"


def test_fusion_pipeline_preserves_existing_metadata(tmp_path):
    visual_path = tmp_path / "visual.json"
    audio_path = tmp_path / "audio.json"
    speech_path = tmp_path / "speech.json"
    output_path = tmp_path / "selected.json"
    old_metadata = {"existing": True}
    output_path.write_text(json.dumps(old_metadata), encoding="utf-8")

    visual_path.write_text(
        json.dumps([{"t": 0.0, "V_motion": 1.0}]), encoding="utf-8"
    )
    audio_path.write_text(json.dumps([{"t": 0.0, "STE": 1.0}]), encoding="utf-8")
    speech_path.write_text(
        json.dumps([{"t": 0.0, "sentiment_score": 1.0, "keyword_hit": False}]),
        encoding="utf-8",
    )

    run_fusion_pipeline(
        source_video=str(tmp_path / "missing.mp4"),
        vis_file=str(visual_path),
        aud_file=str(audio_path),
        speech_file=str(speech_path),
        out_json=str(output_path),
        top_k=1,
        window_sec=1.0,
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == old_metadata
    next_metadata = tmp_path / "selected_2.json"
    assert json.loads(next_metadata.read_text(encoding="utf-8"))["highlights"]
