import json

from auto_generated_ground_truth import locate_highlights


def visual_rows(timestamps, embeddings):
    return [
        {"t": timestamp, "V_spatial": embedding}
        for timestamp, embedding in zip(timestamps, embeddings)
    ]


def test_locates_matching_highlight_segment_and_reports_unmatched(tmp_path):
    full_path = tmp_path / "full_visual.json"
    highlight_path = tmp_path / "highlight_visual.json"
    basis = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    full_path.write_text(
        json.dumps(visual_rows([10.0, 11.0, 12.0], basis)),
        encoding="utf-8",
    )
    highlight_path.write_text(
        json.dumps(
            visual_rows(
                [0.0, 1.0, 2.0, 5.0, 6.0, 7.0],
                basis + [[0.0, 0.0, 0.0, 1.0]] * 3,
            )
        ),
        encoding="utf-8",
    )

    intervals, report = locate_highlights(
        str(full_path),
        str(highlight_path),
        threshold=0.92,
        min_frames=3,
    )

    assert intervals == [[10.0, 13.0]]
    assert report["matched_candidate_clip_count"] == 1
    assert report["unmatched_candidate_clip_count"] == 1
    assert report["label_type"] == "pseudo_ground_truth_from_official_highlights"
