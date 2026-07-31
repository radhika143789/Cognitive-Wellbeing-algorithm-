import os
import logging
import secrets
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from models import db, User, AssessmentResult, JournalEntry, UserStats, ThoughtRecord, ActivityLog, ChatMessage, ShareToken
from nlp_service import analyze_sentiment
from questionnaire_extractor import extract_features, QUESTIONS
from resources import get_recommendations
from gamification import (
    update_streak, evaluate_badges, get_or_create_stats,
    get_badge_details, BADGE_DEFINITIONS
)
from chat_service import generate_ai_response
import joblib
import numpy as np
from datetime import datetime, timezone, timedelta

# ── Configuration ─────────────────────────────────────────────────────────────
# In production set SECRET_KEY via environment variable.
# Never commit a real secret to source control.
_SECRET_KEY = os.environ.get('SECRET_KEY', '')
if not _SECRET_KEY:
    _SECRET_KEY = 'dev-only-insecure-key-change-in-production'
    logging.warning(
        "SECRET_KEY not set in environment — using insecure dev key. "
        "Set the SECRET_KEY env var before deploying."
    )

# Absolute DB path avoids silent data loss when CWD changes.
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_DB_PATH   = os.path.join(_BASE_DIR, 'database.db')

app = Flask(__name__)
app.config['SECRET_KEY']                  = _SECRET_KEY
app.config['SQLALCHEMY_DATABASE_URI']     = f'sqlite:///{_DB_PATH}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Expose Python builtins to Jinja2 templates
app.jinja_env.globals.update(enumerate=enumerate, zip=zip)

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
        # S3: Enforce minimum password length to prevent trivially weak passwords
        if len(password) < 8:
            flash('Password must be at least 8 characters.', 'error')
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
            # P3-1: "remember me" is opt-in, not forced on every login
            remember = request.form.get('remember') == 'on'
            login_user(user, remember=remember)
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
    # P3-2: renamed to chart_* to make their purpose unambiguous
    chart_dates  = [a.timestamp.strftime('%Y-%m-%d %H:%M') for a in assessments]
    chart_scores = [a.score for a in assessments]

    # Gamification context
    stats      = UserStats.query.filter_by(user_id=current_user.id).first()
    streak     = stats.streak_days        if stats else 0
    longest    = stats.longest_streak     if stats else 0
    total_j    = stats.total_journals     if stats else 0
    total_a    = stats.total_assessments  if stats else 0
    earned_ids = stats.get_badges()       if stats else []
    badges     = get_badge_details(earned_ids)

    # Score trend arrow: compare last two scores
    trend = None
    if len(chart_scores) >= 2:
        delta = chart_scores[-1] - chart_scores[-2]
        if delta > 2:
            trend = "up"
        elif delta < -2:
            trend = "down"
        else:
            trend = "stable"

    return render_template(
        'dashboard.html',
        dates=chart_dates,
        scores=chart_scores,
        streak=streak,
        longest=longest,
        total_journals=total_j,
        total_assessments=total_a,
        badges=badges,
        trend=trend,
        latest_score=chart_scores[-1] if chart_scores else None,
    )


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

    # B2: Filter to KNOWN question ids only — ignores stray/CSRF form fields
    form_data = {q['id']: request.form.get(q['id'], 0) for q in QUESTIONS}
    features  = extract_features(form_data, journal_sentiment)

    # Build ordered feature vector matching the model's training columns
    vec = [
        journal_sentiment if col == 'journal_sentiment' else features.get(col, 0)
        for col in feature_cols
    ]

    # Run prediction
    prediction = model.predict([vec])[0]
    score      = round(float(np.clip(prediction, 0, 100)), 2)

    # Persist to DB and update gamification stats for logged-in users
    if current_user.is_authenticated:
        db.session.add(AssessmentResult(score=score, user_id=current_user.id))

        stats = get_or_create_stats(db, current_user.id)
        stats.total_assessments += 1
        newly_awarded = evaluate_badges(stats, latest_score=score)
        db.session.commit()
        _flash_badges(newly_awarded)

    return render_template(
        'assess.html',
        questions=QUESTIONS,
        result=score,
        features=features,
        journal_sentiment=round(journal_sentiment, 3),
    )


# ─────────────────────────── HELPERS ───────────────────────────

def _flash_badges(newly_awarded: list) -> None:
    """Q1: Centralised badge flash — eliminates duplicated code in assess/journal routes."""
    for badge_id in newly_awarded:
        badge = BADGE_DEFINITIONS.get(badge_id, {})
        flash(
            f"🏅 Badge unlocked: {badge.get('emoji', '')} {badge.get('name', '')}!",
            'badge'
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

            # ── Gamification: update streak + check badges ──
            stats = get_or_create_stats(db, current_user.id)
            update_streak(stats)
            newly_awarded = evaluate_badges(stats)
            db.session.commit()
            _flash_badges(newly_awarded)

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
    rec = get_recommendations(score, features)
    return render_template('resources.html', rec=rec, score=round(score, 1))


@app.route('/crisis')
def crisis():
    """Public crisis support page — no login required."""
    return render_template('crisis.html')


# ─────────────────────────── PDF REPORT ───────────────────────────

@app.route('/report')
@login_required
def report():
    """
    Renders a print-optimised report page.
    User triggers print via browser (Ctrl+P / window.print()).
    """
    assessments = (
        AssessmentResult.query
        .filter_by(user_id=current_user.id)
        .order_by(AssessmentResult.timestamp.desc())
        .limit(10)
        .all()
    )
    entries = (
        JournalEntry.query
        .filter_by(user_id=current_user.id)
        .order_by(JournalEntry.timestamp.desc())
        .limit(10)
        .all()
    )
    stats = UserStats.query.filter_by(user_id=current_user.id).first()
    earned_ids = stats.get_badges() if stats else []
    badges     = get_badge_details(earned_ids)

    latest_score = assessments[0].score if assessments else None
    rec = get_recommendations(latest_score or 50.0, {})

    return render_template(
        'report.html',
        assessments=assessments,
        entries=entries,
        badges=badges,
        stats=stats,
        rec=rec,
        latest_score=latest_score,
        username=current_user.username,
        # B1: timezone-aware datetime instead of deprecated utcnow()
        now=datetime.now(timezone.utc).strftime('%d %B %Y, %H:%M UTC'),
    )


# ─────────────────────────── CBT WORKSHEETS ───────────────────────────

@app.route('/cbt', methods=['GET'])
@login_required
def cbt():
    """CBT Worksheet hub — Thought Records + Behavioural Activation."""
    thought_records = (
        ThoughtRecord.query
        .filter_by(user_id=current_user.id)
        .order_by(ThoughtRecord.timestamp.desc())
        .limit(20)
        .all()
    )
    activity_logs = (
        ActivityLog.query
        .filter_by(user_id=current_user.id)
        .order_by(ActivityLog.planned_at.desc())
        .limit(20)
        .all()
    )
    return render_template('cbt.html', thought_records=thought_records, activity_logs=activity_logs)


@app.route('/cbt/thought-record', methods=['POST'])
@login_required
def cbt_thought_record():
    """Save a completed Thought Record."""
    situation        = request.form.get('situation', '').strip()
    emotions         = request.form.get('emotions', '').strip()
    hot_thought      = request.form.get('hot_thought', '').strip()
    evidence_for     = request.form.get('evidence_for', '').strip()
    evidence_against = request.form.get('evidence_against', '').strip()
    balanced_thought = request.form.get('balanced_thought', '').strip()
    outcome_mood_raw = request.form.get('outcome_mood', '')

    if not situation or not hot_thought:
        flash('Situation and hot thought are required fields.', 'error')
        return redirect(url_for('cbt'))

    try:
        outcome_mood = int(outcome_mood_raw) if outcome_mood_raw else None
        if outcome_mood is not None:
            outcome_mood = max(1, min(10, outcome_mood))
    except ValueError:
        outcome_mood = None

    record = ThoughtRecord(
        situation=situation,
        emotions=emotions,
        hot_thought=hot_thought,
        evidence_for=evidence_for,
        evidence_against=evidence_against,
        balanced_thought=balanced_thought,
        outcome_mood=outcome_mood,
        user_id=current_user.id,
    )
    db.session.add(record)
    db.session.commit()
    flash('✅ Thought record saved!', 'success')
    return redirect(url_for('cbt') + '#thought-records')


@app.route('/cbt/activity-log', methods=['POST'])
@login_required
def cbt_activity_log():
    """Save a Behavioural Activation activity log entry."""
    activity    = request.form.get('activity', '').strip()
    mood_before = request.form.get('mood_before', '5')
    mood_after  = request.form.get('mood_after', '')
    notes       = request.form.get('notes', '').strip()

    if not activity:
        flash('Activity description is required.', 'error')
        return redirect(url_for('cbt'))

    try:
        mb = max(1, min(10, int(mood_before)))
    except (ValueError, TypeError):
        mb = 5

    try:
        ma = max(1, min(10, int(mood_after))) if mood_after else None
    except (ValueError, TypeError):
        ma = None

    log = ActivityLog(
        activity=activity,
        mood_before=mb,
        mood_after=ma,
        notes=notes,
        user_id=current_user.id,
    )
    db.session.add(log)
    db.session.commit()
    flash('✅ Activity logged!', 'success')
    return redirect(url_for('cbt') + '#activity-logs')


# ─────────────────────────── PHASE 6 ROUTES ───────────────────────────

@app.route('/chat')
@login_required
def chat():
    """AI MindCompanion Chatbot page."""
    history = ChatMessage.query.filter_by(user_id=current_user.id).order_by(ChatMessage.timestamp.asc()).all()
    return render_template('chat.html', history=history)


@app.route('/chat/api', methods=['POST'])
@login_required
def chat_api():
    """JSON API endpoint for AI MindCompanion conversation."""
    data = request.get_json() or {}
    user_msg = data.get('message', '').strip()
    if not user_msg:
        return jsonify({'error': 'Empty message'}), 400

    # Save user message
    u_msg = ChatMessage(sender='user', message=user_msg, user_id=current_user.id)
    db.session.add(u_msg)

    # Generate AI Response
    ai_out = generate_ai_response(user_msg)
    reply_text = ai_out['response']
    label = ai_out['sentiment_label']

    # Save assistant message
    a_msg = ChatMessage(sender='assistant', message=reply_text, sentiment_label=label, user_id=current_user.id)
    db.session.add(a_msg)
    db.session.commit()

    return jsonify({
        'response': reply_text,
        'sentiment_label': label,
        'is_crisis': ai_out['is_crisis'],
        'quick_replies': ai_out['quick_replies']
    })


@app.route('/share')
@login_required
def share():
    """Clinician link generator page."""
    tokens = ShareToken.query.filter_by(user_id=current_user.id).order_by(ShareToken.created_at.desc()).all()
    new_link = request.args.get('new_link')
    return render_template('share.html', tokens=tokens, new_link=new_link)


@app.route('/share/generate', methods=['POST'])
@login_required
def generate_share_link():
    """Generate a secure, time-limited share token for clinicians."""
    duration_days = int(request.form.get('duration_days', '7'))
    pin = request.form.get('pin', '').strip()

    token_str = secrets.token_urlsafe(32)
    now = datetime.now(timezone.utc)
    expires = now + timedelta(days=duration_days)

    pin_hash = generate_password_hash(pin, method='pbkdf2:sha256') if pin else None

    token_obj = ShareToken(
        token=token_str,
        expires_at=expires,
        pin_hash=pin_hash,
        user_id=current_user.id
    )
    db.session.add(token_obj)
    db.session.commit()

    share_url = url_for('view_shared_report', token=token_str, _external=True)
    flash('✅ Share link created successfully!', 'success')
    return redirect(url_for('share', new_link=share_url))


@app.route('/shared/<token>', methods=['GET', 'POST'])
def view_shared_report(token):
    """Public read-only report view for doctors/therapists."""
    share_token = ShareToken.query.filter_by(token=token).first()

    if not share_token or not share_token.is_valid():
        return render_template('shared_report.html', require_pin=False, expired=True), 404

    target_user = db.session.get(User, share_token.user_id)

    # Check PIN protection if set
    if share_token.pin_hash:
        pin_entered = request.form.get('pin', '')
        if request.method == 'POST':
            if check_password_hash(share_token.pin_hash, pin_entered):
                # PIN verified
                pass
            else:
                return render_template('shared_report.html', require_pin=True, pin_error=True, user=target_user)
        else:
            return render_template('shared_report.html', require_pin=True, pin_error=False, user=target_user)

    # Fetch user summary metrics
    assessments = AssessmentResult.query.filter_by(user_id=target_user.id).order_by(AssessmentResult.timestamp.desc()).all()
    journals = JournalEntry.query.filter_by(user_id=target_user.id).order_by(JournalEntry.timestamp.desc()).limit(10).all()
    cbt_records = ThoughtRecord.query.filter_by(user_id=target_user.id).count()

    latest_score = assessments[0].score if assessments else None

    return render_template(
        'shared_report.html',
        require_pin=False,
        user=target_user,
        now=datetime.now(timezone.utc),
        assessments=assessments,
        journals=journals,
        latest_score=latest_score,
        total_assessments=len(assessments),
        total_journals=len(journals),
        total_cbt=cbt_records
    )


@app.route('/analytics')
@login_required
def analytics():
    """Advanced Multi-Metric Analytics page."""
    assessments = AssessmentResult.query.filter_by(user_id=current_user.id).order_by(AssessmentResult.timestamp.asc()).all()
    journals    = JournalEntry.query.filter_by(user_id=current_user.id).order_by(JournalEntry.timestamp.asc()).all()
    cbt_count   = ThoughtRecord.query.filter_by(user_id=current_user.id).count()
    act_count   = ActivityLog.query.filter_by(user_id=current_user.id).count()

    dates      = [a.timestamp.strftime('%b %d') for a in assessments if a.timestamp]
    scores     = [round(a.score, 1) for a in assessments]
    sentiments = [round(j.sentiment_compound, 2) for j in journals[:len(dates)]]

    # Fill default mock padding if sparse user history
    if not dates:
        dates = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5']
        scores = [65.0, 70.5, 68.0, 74.0, 78.5]
        sentiments = [0.15, 0.32, -0.05, 0.45, 0.60]

    # Sentiment Breakdown Distribution
    pos = sum(1 for j in journals if j.sentiment_label == 'Positive')
    neu = sum(1 for j in journals if j.sentiment_label == 'Neutral')
    neg = sum(1 for j in journals if j.sentiment_label == 'Negative')
    if pos + neu + neg == 0:
        pos, neu, neg = 4, 3, 2

    # Radar Dimensions Balance (0-100)
    avg_score = np.mean(scores) if scores else 70
    thought_clarity   = min(100, int(avg_score * 0.95 + cbt_count * 5))
    mood_stability    = min(100, int(avg_score * 1.02))
    emotional_calm    = min(100, int(avg_score * 0.88 + pos * 3))
    substance_balance = 90
    anxiety_control   = min(100, int(avg_score * 0.92))
    daily_activation  = min(100, int(avg_score * 0.85 + act_count * 6))

    radar_data = [thought_clarity, mood_stability, emotional_calm, substance_balance, anxiety_control, daily_activation]

    return render_template(
        'analytics.html',
        dates=dates,
        scores=scores,
        sentiments=sentiments,
        sentiment_dist={'Positive': pos, 'Neutral': neu, 'Negative': neg},
        radar_data=radar_data
    )


if __name__ == '__main__':
    app.run(debug=True, port=5000)

