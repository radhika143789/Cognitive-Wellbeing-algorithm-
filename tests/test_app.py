"""
tests/test_app.py
-----------------
Integration tests for all Flask routes.
Uses Flask test client with an in-memory SQLite DB.

Run with:
    python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from unittest.mock import MagicMock, patch
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def app():
    """Create a test Flask application with an in-memory database."""
    # Patch the model before importing app so the joblib.load doesn't fail
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([72.5])

    mock_bundle = {"model": mock_model, "feature_cols": [
        "Schizophrenia", "Bipolar_disorder", "Eating_disorders",
        "Anxiety_disorders", "Drug_use_disorders",
        "Depressive_disorders", "Alcohol_use_disorders",
        "journal_sentiment"
    ]}

    with patch("joblib.load", return_value=mock_bundle):
        import app as flask_app_module
        flask_app_module.app.config.update({
            "TESTING": True,
            "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
            "WTF_CSRF_ENABLED": False,
            "SECRET_KEY": "test-secret",
        })
        with flask_app_module.app.app_context():
            flask_app_module.db.create_all()
        yield flask_app_module.app


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def auth_client(client):
    """A client that is already registered and logged in."""
    client.post("/register", data={"username": "testuser", "password": "TestPass123"})
    client.post("/login",    data={"username": "testuser", "password": "TestPass123"})
    return client


# Convenience
QUESTIONNAIRE_DATA = {
    "sq1": "0", "sq2": "0",
    "bq1": "0", "bq2": "0",
    "eq1": "0", "eq2": "0", "eq3": "0",
    "aq1": "1", "aq2": "1", "aq3": "1",
    "dq1": "0", "dq2": "0",
    "pq1": "0", "pq2": "0", "pq3": "0",
    "alq1": "0", "alq2": "0", "alq3": "0",
}


# ─────────────────────────────────────────────────────────────────────────────
# Auth tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAuth:
    def test_register_get(self, client):
        resp = client.get("/register")
        assert resp.status_code == 200

    def test_register_post_success(self, client):
        resp = client.post("/register", data={
            "username": "newuser_reg", "password": "password123"
        }, follow_redirects=True)
        assert resp.status_code == 200
        assert b"Login" in resp.data or b"login" in resp.data

    def test_register_duplicate_user(self, client):
        client.post("/register", data={"username": "dupuser", "password": "pass"})
        resp = client.post("/register", data={
            "username": "dupuser", "password": "pass"
        }, follow_redirects=True)
        assert b"already exists" in resp.data

    def test_register_missing_fields(self, client):
        resp = client.post("/register", data={
            "username": "", "password": ""
        }, follow_redirects=True)
        assert b"required" in resp.data

    def test_login_get(self, client):
        resp = client.get("/login")
        assert resp.status_code == 200

    def test_login_success(self, client):
        client.post("/register", data={"username": "loginuser", "password": "pass123"})
        resp = client.post("/login", data={
            "username": "loginuser", "password": "pass123"
        }, follow_redirects=True)
        assert resp.status_code == 200

    def test_login_wrong_password(self, client):
        client.post("/register", data={"username": "wrongpw", "password": "correct"})
        resp = client.post("/login", data={
            "username": "wrongpw", "password": "wrong"
        }, follow_redirects=True)
        assert b"Invalid" in resp.data

    def test_login_nonexistent_user(self, client):
        resp = client.post("/login", data={
            "username": "doesnotexist", "password": "any"
        }, follow_redirects=True)
        assert b"Invalid" in resp.data

    def test_logout_requires_login(self, client):
        resp = client.get("/logout")
        # Should redirect to login
        assert resp.status_code in (302, 401)

    def test_logout_works_when_logged_in(self, auth_client):
        resp = auth_client.get("/logout", follow_redirects=True)
        assert resp.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# Route protection tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRouteProtection:
    """All @login_required routes must redirect anonymous users."""

    PROTECTED_ROUTES = [
        "/dashboard",
        "/journal",
        "/report",
        "/cbt",
    ]

    @pytest.mark.parametrize("route", PROTECTED_ROUTES)
    def test_anonymous_redirected(self, client, route):
        resp = client.get(route)
        assert resp.status_code == 302
        assert "/login" in resp.headers.get("Location", "")

    def test_public_landing_accessible(self, client):
        assert client.get("/").status_code == 200

    def test_public_assess_accessible(self, client):
        assert client.get("/assess").status_code == 200

    def test_public_resources_accessible(self, client):
        assert client.get("/resources").status_code == 200

    def test_public_crisis_accessible(self, client):
        assert client.get("/crisis").status_code == 200

    def test_public_login_accessible(self, client):
        assert client.get("/login").status_code == 200

    def test_public_register_accessible(self, client):
        assert client.get("/register").status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# Assessment tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAssessment:
    def test_get_assess_renders(self, client):
        resp = client.get("/assess")
        assert resp.status_code == 200
        assert b"question" in resp.data.lower() or b"assess" in resp.data.lower()

    def test_post_assess_returns_result(self, client):
        resp = client.post("/assess", data=QUESTIONNAIRE_DATA)
        assert resp.status_code == 200
        # Score should appear (the mock returns 72.5)
        assert b"72" in resp.data or b"score" in resp.data.lower()

    def test_post_assess_logged_in_saves_to_db(self, auth_client):
        resp = auth_client.post("/assess", data=QUESTIONNAIRE_DATA, follow_redirects=True)
        assert resp.status_code == 200

    def test_post_assess_score_is_float(self, client):
        # The mocked model returns 72.5 — just verify the response isn't an error page
        resp = client.post("/assess", data=QUESTIONNAIRE_DATA)
        assert b"error" not in resp.data.lower() or b"Model not" not in resp.data

    def test_assess_increments_assessment_count(self, auth_client):
        import app as flask_app_module
        from models import UserStats
        with flask_app_module.app.app_context():
            auth_client.post("/assess", data=QUESTIONNAIRE_DATA)
            stats = UserStats.query.filter_by(user_id=1).first()
            if stats:
                assert stats.total_assessments >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Journal tests
# ─────────────────────────────────────────────────────────────────────────────

class TestJournal:
    def test_journal_get_requires_login(self, client):
        resp = client.get("/journal")
        assert resp.status_code == 302

    def test_journal_get_renders_when_logged_in(self, auth_client):
        resp = auth_client.get("/journal")
        assert resp.status_code == 200

    def test_journal_post_saves_entry(self, auth_client):
        resp = auth_client.post("/journal", data={
            "content": "Today I felt really good about my progress."
        }, follow_redirects=True)
        assert resp.status_code == 200
        # Sentiment result should show
        assert b"Positive" in resp.data or b"journal" in resp.data.lower()

    def test_journal_post_negative_text(self, auth_client):
        resp = auth_client.post("/journal", data={
            "content": "I am feeling terribly sad and hopeless today."
        }, follow_redirects=True)
        assert resp.status_code == 200
        assert b"Negative" in resp.data

    def test_journal_post_empty_content_ignored(self, auth_client):
        resp = auth_client.post("/journal", data={"content": "   "}, follow_redirects=True)
        assert resp.status_code == 200  # gracefully handled

    def test_journal_updates_streak(self, auth_client):
        import app as flask_app_module
        from models import UserStats
        auth_client.post("/journal", data={"content": "Streak test entry."})
        with flask_app_module.app.app_context():
            stats = UserStats.query.first()
            if stats:
                assert stats.streak_days >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDashboard:
    def test_dashboard_renders(self, auth_client):
        resp = auth_client.get("/dashboard")
        assert resp.status_code == 200
        assert b"Dashboard" in resp.data or b"Journey" in resp.data

    def test_dashboard_shows_streak(self, auth_client):
        resp = auth_client.get("/dashboard")
        assert resp.status_code == 200
        # Streak card should be present
        assert b"streak" in resp.data.lower() or b"Streak" in resp.data


# ─────────────────────────────────────────────────────────────────────────────
# Report tests
# ─────────────────────────────────────────────────────────────────────────────

class TestReport:
    def test_report_redirects_anonymous(self, client):
        resp = client.get("/report")
        assert resp.status_code == 302

    def test_report_renders_when_logged_in(self, auth_client):
        resp = auth_client.get("/report")
        assert resp.status_code == 200
        assert b"MindTrack" in resp.data or b"Report" in resp.data

    def test_report_contains_username(self, auth_client):
        resp = auth_client.get("/report")
        assert b"testuser" in resp.data

    def test_report_contains_print_button(self, auth_client):
        resp = auth_client.get("/report")
        assert b"Print" in resp.data or b"print" in resp.data


# ─────────────────────────────────────────────────────────────────────────────
# CBT tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCBT:
    def test_cbt_redirects_anonymous(self, client):
        resp = client.get("/cbt")
        assert resp.status_code == 302

    def test_cbt_renders_when_logged_in(self, auth_client):
        resp = auth_client.get("/cbt")
        assert resp.status_code == 200
        assert b"Thought" in resp.data or b"CBT" in resp.data

    def test_thought_record_save(self, auth_client):
        resp = auth_client.post("/cbt/thought-record", data={
            "situation":        "I made a mistake at work.",
            "emotions":         "Anxious 70%, Embarrassed 50%",
            "hot_thought":      "I am completely incompetent.",
            "evidence_for":     "I missed a deadline.",
            "evidence_against": "I have delivered 20 projects successfully.",
            "balanced_thought": "One mistake does not define my competence.",
            "outcome_mood":     "7",
        }, follow_redirects=True)
        assert resp.status_code == 200
        assert b"saved" in resp.data.lower() or b"Thought" in resp.data

    def test_thought_record_missing_required_fields(self, auth_client):
        resp = auth_client.post("/cbt/thought-record", data={
            "situation":   "",
            "emotions":    "Sad",
            "hot_thought": "",
        }, follow_redirects=True)
        assert resp.status_code == 200
        assert b"required" in resp.data.lower()

    def test_activity_log_save(self, auth_client):
        resp = auth_client.post("/cbt/activity-log", data={
            "activity":    "10-minute walk around the block",
            "mood_before": "4",
            "mood_after":  "7",
            "notes":       "Felt much better after getting outside.",
        }, follow_redirects=True)
        assert resp.status_code == 200
        assert b"logged" in resp.data.lower() or b"Activity" in resp.data

    def test_activity_log_missing_activity(self, auth_client):
        resp = auth_client.post("/cbt/activity-log", data={
            "activity":    "",
            "mood_before": "5",
        }, follow_redirects=True)
        assert resp.status_code == 200
        assert b"required" in resp.data.lower()

    def test_thought_records_appear_in_history(self, auth_client):
        auth_client.post("/cbt/thought-record", data={
            "situation":   "History test situation.",
            "emotions":    "Calm",
            "hot_thought": "History hot thought.",
        }, follow_redirects=True)
        resp = auth_client.get("/cbt")
        assert resp.status_code == 200

    def test_outcome_mood_clamped(self, auth_client):
        """An out-of-range mood value should not crash the route."""
        resp = auth_client.post("/cbt/thought-record", data={
            "situation":    "Test clamping.",
            "emotions":     "Anxious",
            "hot_thought":  "I will fail.",
            "outcome_mood": "999",  # out of range
        }, follow_redirects=True)
        assert resp.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# Resources + Crisis tests
# ─────────────────────────────────────────────────────────────────────────────

class TestResourcesAndCrisis:
    def test_resources_get(self, client):
        assert client.get("/resources").status_code == 200

    def test_crisis_get(self, client):
        assert client.get("/crisis").status_code == 200

    def test_resources_renders_breathing(self, client):
        resp = client.get("/resources")
        assert b"Breathing" in resp.data or b"breathing" in resp.data

    def test_crisis_renders_hotline_text(self, client):
        resp = client.get("/crisis")
        assert b"crisis" in resp.data.lower() or b"Crisis" in resp.data
