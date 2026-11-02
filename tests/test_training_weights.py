import json

import pytest

from training_weights import candidate_weights, labels_for_intervals, make_report


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return str(path)


def channel_rows(values, field):
    rows = []
    for timestamp, value in enumerate(values):
        row = {"t": float(timestamp), field: value}
        if field == "sentiment_score":
            row["keyword_hit"] = value > 0.5
        rows.append(row)
    return rows


def test_candidate_weights_cover_simplex():
    weights = list(candidate_weights(4))

    assert len(weights) == 15
    assert all(abs(sum(candidate) - 1.0) < 1e-12 for candidate in weights)
    assert all(all(0.0 <= value <= 1.0 for value in candidate) for candidate in weights)


def test_labels_use_half_open_intervals():
    labels = labels_for_intervals(
        [0.0, 1.0, 2.0, 3.0, 4.0],
        [[1.0, 3.0]],
    )

    assert labels.tolist() == [0, 1, 1, 0, 0]


def test_make_report_trains_on_one_match_and_tests_another(tmp_path):
    train_visual = [0.1] * 20
    train_audio = [0.2] * 20
    train_speech = [0.1] * 20
    for index in range(4, 8):
        train_visual[index] = 0.9
        train_audio[index] = 0.8
        train_speech[index] = 0.9

    test_visual = [0.2] * 20
    test_audio = [0.1] * 20
    test_speech = [0.2] * 20
    for index in range(10, 14):
        test_visual[index] = 0.8
        test_audio[index] = 0.9
        test_speech[index] = 0.8

    paths = {}
    for prefix, visual, audio, speech, intervals in (
        ("train", train_visual, train_audio, train_speech, [[4, 8]]),
        ("test", test_visual, test_audio, test_speech, [[10, 14]]),
    ):
        paths[f"{prefix}_visual"] = write_json(
            tmp_path / f"{prefix}_visual.json",
            channel_rows(visual, "V_motion"),
        )
        paths[f"{prefix}_audio"] = write_json(
            tmp_path / f"{prefix}_audio.json",
            channel_rows(audio, "STE"),
        )
        paths[f"{prefix}_speech"] = write_json(
            tmp_path / f"{prefix}_speech.json",
            channel_rows(speech, "sentiment_score"),
        )
        paths[f"{prefix}_ground_truth"] = write_json(
            tmp_path / f"{prefix}_ground_truth.json",
            intervals,
        )

    from argparse import Namespace

    args = Namespace(
        sport="cricket",
        train_match_id="match-one",
        test_match_id="match-two",
        grid_resolution=4,
        top_k=1,
        window_sec=4.0,
        overlap_iou=0.5,
        **paths,
    )
    report = make_report(args)

    assert report["split_method"] == "entire matches; no within-match second-level split"
    assert set(report["methods"]) == {
        "hand_set_baseline",
        "simplex_grid_search",
        "logistic_regression",
    }
    assert report["methods"]["simplex_grid_search"]["candidate_count"] == 15
    assert len(report["methods"]["logistic_regression"]["coefficients"]) == 3
    assert report["methods"]["logistic_regression"]["test_metrics"]["f1"] is not None


def test_make_report_rejects_same_match_id():
    from argparse import Namespace

    args = Namespace(
        train_match_id="same-match",
        test_match_id="SAME-MATCH",
    )

    with pytest.raises(ValueError, match="different matches"):
        make_report(args)
