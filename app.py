import os
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from models import db, User, AssessmentResult, JournalEntry
from nlp_service import analyze_sentiment
from questionnaire_extractor import extract_features, QUESTIONS
from resources import get_recommendations
import joblib
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'super-secret-key-replace-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Expose Python's enumerate to Jinja2 templates
app.jinja_env.globals.update(enumerate=enumerate)

db.init_app(app)

login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)


# SQLAlchemy 2.x compatible user loader
@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


# Create database tables on startup
with app.app_context():
    db.create_all()


# Load the trained model bundle
try:
    bundle = joblib.load('model.pkl')
    model = bundle['model']
    feature_cols = bundle['feature_cols']
except Exception as e:
    model = None
    feature_cols = []
    print(f"Warning: Could not load model.pkl — {e}. Run train_model.py first.")


# ─────────────────────────── PUBLIC PAGES ───────────────────────────

@app.route('/')
def landing():
    """Public landing / marketing page."""
    return render_template('landing.html')


# ─────────────────────────── AUTH ───────────────────────────

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        if not username or not password:
            flash('Username and password are required.', 'error')
            return redirect(url_for('register'))
        if User.query.filter_by(username=username).first():
            flash('Username already exists.', 'error')
            return redirect(url_for('register'))
        new_user = User(
            username=username,
            password=generate_password_hash(password, method='pbkdf2:sha256')
        )
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user, remember=True)
            return redirect(url_for('dashboard'))
        flash('Invalid username or password.', 'error')
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('landing'))


# ─────────────────────────── DASHBOARD ───────────────────────────

@app.route('/dashboard')
@login_required
def dashboard():
    assessments = (
        AssessmentResult.query
        .filter_by(user_id=current_user.id)
        .order_by(AssessmentResult.timestamp.asc())
        .all()
    )
    dates  = [a.timestamp.strftime('%Y-%m-%d %H:%M') for a in assessments]
    scores = [a.score for a in assessments]
    return render_template('dashboard.html', dates=dates, scores=scores)


# ─────────────────────────── ASSESSMENT ───────────────────────────

@app.route('/assess', methods=['GET', 'POST'])
def assess():
    """
    GET  → render multi-step questionnaire
    POST → extract features, run ML prediction, save result, render result page
    """
    if request.method == 'GET':
        return render_template('assess.html', questions=QUESTIONS, result=None)

    # ── POST: process questionnaire submission ──
    if model is None:
        flash('Model not loaded. Please run train_model.py first.', 'error')
        return redirect(url_for('assess'))

    # Fetch average journal sentiment for logged-in users (last 7 entries)
    journal_sentiment = 0.0
    if current_user.is_authenticated:
        recent_entries = (
            JournalEntry.query
            .filter_by(user_id=current_user.id)
            .order_by(JournalEntry.timestamp.desc())
            .limit(7)
            .all()
        )
        if recent_entries:
            journal_sentiment = float(
                np.mean([e.sentiment_compound for e in recent_entries])
            )

    # Map raw form answers → disorder prevalence features
    form_data = {key: request.form.get(key, 0) for key in request.form}
    features  = extract_features(form_data, journal_sentiment)

    # Build ordered feature vector matching the model's training columns
    vec = [
        journal_sentiment if col == 'journal_sentiment' else features.get(col, 0)
        for col in feature_cols
    ]

    # Run prediction
    prediction = model.predict([vec])[0]
    score      = round(float(np.clip(prediction, 0, 100)), 2)

    # Persist to DB for logged-in users
    if current_user.is_authenticated:
        db.session.add(AssessmentResult(score=score, user_id=current_user.id))
        db.session.commit()

    return render_template(
        'assess.html',
        questions=QUESTIONS,
        result=score,
        features=features,
        journal_sentiment=round(journal_sentiment, 3),
    )


# ─────────────────────────── JOURNAL ───────────────────────────

@app.route('/journal', methods=['GET', 'POST'])
@login_required
def journal():
    result = None
    if request.method == 'POST':
        content = request.form.get('content', '').strip()
        if content:
            sentiment = analyze_sentiment(content)
            db.session.add(JournalEntry(
                content=content,
                sentiment_compound=sentiment['compound'],
                sentiment_label=sentiment['label'],
                user_id=current_user.id
            ))
            db.session.commit()
            result = sentiment

    entries = (
        JournalEntry.query
        .filter_by(user_id=current_user.id)
        .order_by(JournalEntry.timestamp.desc())
        .limit(10)
        .all()
    )
    return render_template('journal.html', result=result, entries=entries)


# ─────────────────────────── RESOURCES & CRISIS ───────────────────────────

@app.route('/resources')
def resources():
    """
    Personalized recommendations based on the most recent assessment.
    Falls back to a neutral score if no assessment exists.
    """
    score    = 50.0
    features = {}
    if current_user.is_authenticated:
        latest = (
            AssessmentResult.query
            .filter_by(user_id=current_user.id)
            .order_by(AssessmentResult.timestamp.desc())
            .first()
        )
        if latest:
            score = latest.score
        # Pull the most recently stored feature snapshot if available
        # (For now we derive from score band — features are stateless)
    rec = get_recommendations(score, features)
    return render_template('resources.html', rec=rec, score=round(score, 1))


@app.route('/crisis')
def crisis():
    """Public crisis support page — no login required."""
    return render_template('crisis.html')


if __name__ == '__main__':
    app.run(debug=True, port=5000)
