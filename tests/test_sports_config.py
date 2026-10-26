import unittest

from sports.config import detect_events, keyword_events, load_sport_config
from sports.evaluate_football import evaluate


class SportConfigTests(unittest.TestCase):
    def test_profiles_load_and_preserve_cricket_keyword_coverage(self):
        cricket = load_sport_config("cricket")
        football = load_sport_config("football")
        self.assertEqual(cricket["tuning_status"], "baseline")
        self.assertEqual(football["tuning_status"], "initial_guess")

        original_cricket_keywords = {
            "six", "four", "out", "wicket", "catch", "bowled", "huge",
            "shot", "boundary", "review", "appeal", "gone", "taken",
            "maximum", "stumped", "lbw", "dropped", "unbelievable", "wided",
        }
        configured = set(cricket["keyword_lists"]["general"])
        for words in cricket["keyword_lists"].values():
            configured.update(words)
        self.assertEqual(configured, original_cricket_keywords)
        self.assertEqual(keyword_events("That is a SIX!", cricket), ["six"])

    def test_football_requires_event_evidence_and_groups_adjacent_hits(self):
        football = load_sport_config("football")
        events = detect_events(
            [
                {"t": 1, "transcript_segment": "GOAL!"},
                {"t": 2, "transcript_segment": "He scores a goal"},
                {"t": 10, "STE": 0.03, "V_motion": 0.02},
                {"t": 20, "transcript_segment": "Red card"},
                {"t": 30, "STE": 0.03, "V_motion": 0.0},
            ],
            football,
        )
        self.assertEqual(
            [(event["event_type"], event["start"], event["end"]) for event in events],
            [("goal", 1.0, 2.0), ("near_miss", 10.0, 10.0), ("card", 20.0, 20.0)],
        )

    def test_event_evaluation_reports_precision_and_recall(self):
        precision, recall, matches = evaluate(
            [
                {"t": 10, "event_type": "goal"},
                {"t": 40, "event_type": "card"},
            ],
            [
                {"event_type": "goal", "start": 8, "end": 12},
                {"event_type": "near_miss", "start": 20, "end": 22},
            ],
            tolerance=2,
        )
        self.assertEqual(matches, 1)
        self.assertEqual(precision, 0.5)
        self.assertEqual(recall, 0.5)

    def test_football_crowd_and_motion_strength_separate_goal_from_near_miss(self):
        football = load_sport_config("football")
        events = detect_events(
            [
                {"t": 1, "STE": 0.03, "V_motion": 0.015},
                {"t": 2, "STE": 0.05, "V_motion": 0.025},
            ],
            football,
        )
        self.assertEqual(
            [(event["event_type"], event["t"]) for event in events],
            [("near_miss", 1.0), ("goal", 2.0)],
        )


if __name__ == "__main__":
    unittest.main()
