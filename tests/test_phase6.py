"""
test_phase6.py — Unit & Integration Test Suite for Phase 6
Covers AI MindCompanion conversation engine, clinician sharing token security, and analytics routes.
"""

import pytest
from werkzeug.security import generate_password_hash
from app import app, db
from models import User, ChatMessage, ShareToken
from chat_service import generate_ai_response


@pytest.fixture
def client():
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    app.config['WTF_CSRF_ENABLED'] = False

    with app.test_client() as client:
        with app.app_context():
            db.create_all()
            user = User.query.filter_by(username='phase6user').first()
            if not user:
                user = User(
                    username='phase6user',
                    password=generate_password_hash('testpass123', method='pbkdf2:sha256')
                )
                db.session.add(user)
                db.session.commit()
        yield client
        with app.app_context():
            db.session.remove()
            db.drop_all()


def login(client):
    """Helper to log in test user."""
    return client.post('/login', data={'username': 'phase6user', 'password': 'testpass123'}, follow_redirects=True)


# ── 1. Unit Tests for Chat Service ────────────────────────────────────────────

def test_chat_service_empty():
    res = generate_ai_response("")
    assert "didn't catch that" in res['response']
    assert res['sentiment_label'] == 'Neutral'
    assert res['is_crisis'] is False


def test_chat_service_crisis():
    res = generate_ai_response("I feel hopeless and want to end my life")
    assert res['is_crisis'] is True
    assert "988 Lifeline" in res['response'] or "iCall" in res['response']


def test_chat_service_anxiety():
    res = generate_ai_response("I am having panic attacks and breathing stress")
    assert "anxiety" in res['response'].lower() or "breath" in res['response'].lower()
    assert res['is_crisis'] is False


def test_chat_service_positive():
    res = generate_ai_response("I feel so happy and excited today!")
    assert res['sentiment_label'] == 'Positive'


# ── 2. Integration Tests for Phase 6 Routes ───────────────────────────────────

def test_chat_page_requires_auth(client):
    res = client.get('/chat')
    assert res.status_code == 302


def test_chat_page_authenticated(client):
    login(client)
    res = client.get('/chat')
    assert res.status_code == 200
    assert b'MindCompanion' in res.data


def test_chat_api_endpoint(client):
    login(client)
    res = client.post('/chat/api', json={'message': 'I am feeling calm today'})
    assert res.status_code == 200
    data = res.get_json()
    assert 'response' in data
    assert data['sentiment_label'] == 'Positive'


def test_share_page_and_generate(client):
    login(client)
    res = client.get('/share')
    assert res.status_code == 200

    # Generate token
    gen_res = client.post('/share/generate', data={'duration_days': 7, 'pin': '1234'}, follow_redirects=True)
    assert gen_res.status_code == 200
    assert b'Share link created' in gen_res.data


def test_analytics_page(client):
    login(client)
    res = client.get('/analytics')
    assert res.status_code == 200
    assert b'Advanced Multi-Metric' in res.data
