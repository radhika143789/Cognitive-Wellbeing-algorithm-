"""
tests/test_ml.py
----------------
Unit tests for the ML/NLP/resources layer.
These tests run WITHOUT the Flask server — pure Python logic.

Run with:
    python -m pytest tests/ -v
"""

import sys
import os
# Make sure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# questionnaire_extractor tests
# ─────────────────────────────────────────────────────────────────────────────

from questionnaire_extractor import extract_features, QUESTIONS, _normalize


class TestNormalize:
    def test_zero_gives_lo(self):
        assert _normalize(0, 2, 0.10, 1.00) == pytest.approx(0.10)

    def test_max_gives_hi(self):
        assert _normalize(8, 2, 0.10, 1.00) == pytest.approx(1.00)

    def test_midpoint(self):
        result = _normalize(4, 2, 0.0, 2.0)
        assert result == pytest.approx(1.0)

    def test_zero_max_possible_no_divide(self):
        # n_questions=0 → max_possible=0 → should return lo without ZeroDivision
        result = _normalize(0, 0, 0.5, 2.0)
        assert result == pytest.approx(0.5)


class TestExtractFeatures:
    """Test extract_features with controlled inputs."""

    def _all_zero_form(self):
        return {q["id"]: 0 for q in QUESTIONS}

    def _all_max_form(self):
        return {q["id"]: 4 for q in QUESTIONS}

    def test_all_zero_returns_lo_bounds(self):
        features = extract_features(self._all_zero_form(), journal_sentiment=0.0)
        assert features["Schizophrenia"]         == pytest.approx(0.10, abs=0.01)
        assert features["Bipolar_disorder"]      == pytest.approx(0.50, abs=0.01)
        assert features["Eating_disorders"]      == pytest.approx(0.10, abs=0.01)
        assert features["Drug_use_disorders"]    == pytest.approx(0.20, abs=0.01)

    def test_all_max_returns_hi_bounds(self):
        features = extract_features(self._all_max_form(), journal_sentiment=0.0)
        assert features["Schizophrenia"]         == pytest.approx(1.00, abs=0.01)
        assert features["Bipolar_disorder"]      == pytest.approx(2.00, abs=0.01)
        assert features["Eating_disorders"]      == pytest.approx(1.50, abs=0.01)

    def test_output_keys_present(self):
        features = extract_features(self._all_zero_form())
        expected_keys = {
            "Schizophrenia", "Bipolar_disorder", "Eating_disorders",
            "Anxiety_disorders", "Drug_use_disorders",
            "Depressive_disorders", "Alcohol_use_disorders"
        }
        assert set(features.keys()) == expected_keys

    def test_anxiety_increases_with_negative_sentiment(self):
        neutral    = extract_features(self._all_zero_form(), journal_sentiment=0.0)
        negative   = extract_features(self._all_zero_form(), journal_sentiment=-1.0)
        assert negative["Anxiety_disorders"] > neutral["Anxiety_disorders"]

    def test_anxiety_decreases_with_positive_sentiment(self):
        neutral   = extract_features(self._all_zero_form(), journal_sentiment=0.0)
        positive  = extract_features(self._all_zero_form(), journal_sentiment=1.0)
        assert positive["Anxiety_disorders"] <= neutral["Anxiety_disorders"]

    def test_depression_increases_with_negative_sentiment(self):
        neutral  = extract_features(self._all_zero_form(), journal_sentiment=0.0)
        negative = extract_features(self._all_zero_form(), journal_sentiment=-1.0)
        assert negative["Depressive_disorders"] > neutral["Depressive_disorders"]

    def test_clamping_prevents_out_of_range(self):
        features = extract_features(self._all_max_form(), journal_sentiment=-1.0)
        assert features["Anxiety_disorders"]    <= 8.00
        assert features["Depressive_disorders"] <= 7.00

    def test_missing_keys_default_to_zero(self):
        """If form data is missing question ids, should default to 0 (not crash)."""
        features = extract_features({}, journal_sentiment=0.0)
        assert "Anxiety_disorders" in features
        assert features["Anxiety_disorders"] >= 2.0  # lo bound

    def test_clamp_input_values(self):
        """Answers outside 0-4 should be clamped, not raise an error."""
        bad_form = {q["id"]: 99 for q in QUESTIONS}  # all 99 → clamped to 4
        features = extract_features(bad_form, journal_sentiment=0.0)
        assert features["Schizophrenia"] == pytest.approx(1.00, abs=0.01)


# ─────────────────────────────────────────────────────────────────────────────
# nlp_service tests
# ─────────────────────────────────────────────────────────────────────────────

from nlp_service import analyze_sentiment


class TestAnalyzeSentiment:
    def test_positive_text(self):
        result = analyze_sentiment("I feel absolutely wonderful and happy today!")
        assert result["label"] == "Positive"
        assert result["compound"] > 0.05
        assert result["emoji"] == "😊"

    def test_negative_text(self):
        result = analyze_sentiment("I feel terrible, hopeless and very sad.")
        assert result["label"] == "Negative"
        assert result["compound"] < -0.05
        assert result["emoji"] == "😔"

    def test_neutral_text(self):
        result = analyze_sentiment("The meeting is at 3pm.")
        assert result["label"] == "Neutral"
        assert result["emoji"] == "😐"

    def test_output_keys(self):
        result = analyze_sentiment("Test")
        required_keys = {"compound", "label", "emoji", "color", "display_score", "pos", "neg", "neu"}
        assert required_keys.issubset(set(result.keys()))

    def test_display_score_range(self):
        result = analyze_sentiment("This is great!")
        assert 0 <= result["display_score"] <= 100

    def test_component_scores_sum_to_one(self):
        result = analyze_sentiment("I am not sure about everything here")
        total = result["pos"] + result["neg"] + result["neu"]
        assert total == pytest.approx(1.0, abs=0.01)

    def test_empty_string_does_not_crash(self):
        result = analyze_sentiment("")
        assert "label" in result

    def test_very_long_text(self):
        text = ("I am feeling really great today. " * 200)
        result = analyze_sentiment(text)
        assert result["label"] in {"Positive", "Neutral", "Negative"}


# ─────────────────────────────────────────────────────────────────────────────
# resources tests
# ─────────────────────────────────────────────────────────────────────────────

from resources import get_recommendations, get_score_band


class TestGetScoreBand:
    def test_excellent(self):   assert get_score_band(100) == "excellent"
    def test_excellent_lo(self): assert get_score_band(75) == "excellent"
    def test_good(self):        assert get_score_band(74) == "good"
    def test_good_lo(self):     assert get_score_band(55) == "good"
    def test_moderate(self):    assert get_score_band(54) == "moderate"
    def test_moderate_lo(self): assert get_score_band(35) == "moderate"
    def test_low(self):         assert get_score_band(34) == "low"
    def test_zero(self):        assert get_score_band(0)  == "low"


class TestGetRecommendations:
    def _rec(self, score, features=None):
        return get_recommendations(score, features or {})

    def test_output_keys(self):
        rec = self._rec(60)
        required = {"score_band","headline","color","emoji","breathing","grounding","cbt_prompts","lifestyle","crisis_needed","triggered"}
        assert required.issubset(set(rec.keys()))

    def test_excellent_no_crisis(self):
        rec = self._rec(90)
        assert rec["score_band"] == "excellent"
        assert rec["crisis_needed"] is False

    def test_low_triggers_crisis(self):
        rec = self._rec(20)
        assert rec["crisis_needed"] is True

    def test_score_29_triggers_crisis(self):
        rec = self._rec(29)
        assert rec["crisis_needed"] is True

    def test_score_30_no_crisis_no_schizophrenia(self):
        rec = self._rec(30, {})
        assert rec["crisis_needed"] is False

    def test_schizophrenia_triggers_crisis(self):
        rec = self._rec(80, {"Schizophrenia": 0.6})
        assert rec["crisis_needed"] is True

    def test_breathing_always_present(self):
        rec = self._rec(50)
        assert len(rec["breathing"]) >= 1

    def test_grounding_always_present(self):
        rec = self._rec(50)
        assert len(rec["grounding"]) >= 1

    def test_cbt_prompts_max_6(self):
        rec = self._rec(20, {"Anxiety_disorders": 6.0, "Depressive_disorders": 5.0})
        assert len(rec["cbt_prompts"]) <= 6

    def test_anxiety_adds_cbt_prompts(self):
        base = self._rec(50, {})
        with_anxiety = self._rec(50, {"Anxiety_disorders": 6.0})
        assert len(with_anxiety["cbt_prompts"]) >= len(base["cbt_prompts"])

    def test_triggered_disorders_listed(self):
        rec = self._rec(50, {"Anxiety_disorders": 6.0})
        assert "Anxiety_disorders" in rec["triggered"]

    def test_empty_features_no_triggered(self):
        rec = self._rec(60, {})
        assert rec["triggered"] == []

    def test_alcohol_lifestyle_tip(self):
        rec = self._rec(50, {"Alcohol_use_disorders": 3.0})
        assert any("alcohol" in tip.lower() or "drink" in tip.lower() for tip in rec["lifestyle"])


# ─────────────────────────────────────────────────────────────────────────────
# gamification tests
# ─────────────────────────────────────────────────────────────────────────────

from datetime import date, timedelta
from gamification import update_streak, evaluate_badges, get_badge_details, BADGE_DEFINITIONS


class MockStats:
    """Minimal mock of UserStats for unit testing."""
    def __init__(self):
        self.streak_days       = 0
        self.longest_streak    = 0
        self.last_journal_date = None
        self.total_journals    = 0
        self.total_assessments = 0
        self.badges_earned     = ''

    def get_badges(self):
        return [b for b in self.badges_earned.split(',') if b]

    def award_badge(self, badge_id):
        current = self.get_badges()
        if badge_id not in current:
            current.append(badge_id)
            self.badges_earned = ','.join(current)
            return True
        return False


class TestUpdateStreak:
    def test_first_journal_sets_streak_1(self):
        stats = MockStats()
        today = date.today()
        update_streak(stats, today)
        assert stats.streak_days == 1
        assert stats.last_journal_date == today

    def test_consecutive_day_increments(self):
        stats = MockStats()
        yesterday = date.today() - timedelta(days=1)
        stats.last_journal_date = yesterday
        stats.streak_days = 1
        update_streak(stats, date.today())
        assert stats.streak_days == 2

    def test_gap_resets_streak(self):
        stats = MockStats()
        stats.last_journal_date = date.today() - timedelta(days=5)
        stats.streak_days = 10
        update_streak(stats, date.today())
        assert stats.streak_days == 1

    def test_same_day_is_idempotent(self):
        stats = MockStats()
        today = date.today()
        stats.last_journal_date = today
        stats.streak_days = 5
        result = update_streak(stats, today)
        assert result is False
        assert stats.streak_days == 5

    def test_longest_streak_updated(self):
        stats = MockStats()
        stats.last_journal_date = date.today() - timedelta(days=1)
        stats.streak_days = 6
        stats.longest_streak = 6
        update_streak(stats, date.today())
        assert stats.longest_streak == 7

    def test_longest_streak_not_lowered(self):
        stats = MockStats()
        stats.last_journal_date = date.today() - timedelta(days=5)
        stats.streak_days = 3
        stats.longest_streak = 20  # was 20 before
        update_streak(stats, date.today())
        assert stats.longest_streak == 20  # should stay at 20

    def test_total_journals_incremented(self):
        stats = MockStats()
        stats.total_journals = 4
        update_streak(stats, date.today())
        assert stats.total_journals == 5


class TestEvaluateBadges:
    def test_first_journal_badge(self):
        stats = MockStats()
        stats.total_journals = 1
        new = evaluate_badges(stats)
        assert "first_journal" in new
        assert "first_journal" in stats.get_badges()

    def test_streak_3_badge(self):
        stats = MockStats()
        stats.streak_days = 3
        new = evaluate_badges(stats)
        assert "streak_3" in new

    def test_streak_7_badge(self):
        stats = MockStats()
        stats.streak_days = 7
        new = evaluate_badges(stats)
        assert "streak_7" in new
        assert "streak_3" in new  # also qualifies for 3

    def test_no_duplicate_badge_award(self):
        stats = MockStats()
        stats.total_journals = 1
        evaluate_badges(stats)
        new_again = evaluate_badges(stats)  # second call
        assert "first_journal" not in new_again  # already awarded

    def test_assessment_badges(self):
        stats = MockStats()
        stats.total_assessments = 5
        new = evaluate_badges(stats)
        assert "first_assessment" in new
        assert "assessments_5" in new

    def test_score_badge_80(self):
        stats = MockStats()
        new = evaluate_badges(stats, latest_score=85.0)
        assert "score_80" in new

    def test_score_badge_95(self):
        stats = MockStats()
        new = evaluate_badges(stats, latest_score=97.0)
        assert "score_95" in new
        assert "score_80" in new

    def test_score_badge_not_awarded_below_threshold(self):
        stats = MockStats()
        new = evaluate_badges(stats, latest_score=79.9)
        assert "score_80" not in new

    def test_no_score_badge_without_score(self):
        stats = MockStats()
        new = evaluate_badges(stats, latest_score=None)
        assert "score_80" not in new


class TestGetBadgeDetails:
    def test_returns_full_detail(self):
        details = get_badge_details(["first_journal", "streak_7"])
        assert len(details) == 2
        ids = [d["id"] for d in details]
        assert "first_journal" in ids
        assert "streak_7" in ids

    def test_unknown_ids_skipped(self):
        details = get_badge_details(["nonexistent_badge"])
        assert len(details) == 0

    def test_sorted_by_tier_descending(self):
        # platinum > gold > silver > bronze
        details = get_badge_details(["first_journal", "score_80", "streak_30"])
        tiers = [d["tier"] for d in details]
        tier_order = {"platinum": 3, "gold": 2, "silver": 1, "bronze": 0}
        values = [tier_order[t] for t in tiers]
        assert values == sorted(values, reverse=True)

    def test_all_definitions_have_required_fields(self):
        for badge_id, defn in BADGE_DEFINITIONS.items():
            assert "name"        in defn, f"{badge_id} missing name"
            assert "emoji"       in defn, f"{badge_id} missing emoji"
            assert "description" in defn, f"{badge_id} missing description"
            assert "tier"        in defn, f"{badge_id} missing tier"
            assert defn["tier"]  in ("bronze","silver","gold","platinum"), f"{badge_id} bad tier"
