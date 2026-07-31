from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime, timezone


def _utcnow():
    """Timezone-aware UTC now. Drop-in replacement for deprecated datetime.utcnow."""
    return datetime.now(timezone.utc)

db = SQLAlchemy()


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    # P3-3: String(255) is more future-proof for longer hash algorithms
    password = db.Column(db.String(255), nullable=False)
    assessments    = db.relationship('AssessmentResult', backref='user', lazy=True)
    journal_entries = db.relationship('JournalEntry',    backref='user', lazy=True)
    thought_records = db.relationship('ThoughtRecord',   backref='user', lazy=True)
    activity_logs   = db.relationship('ActivityLog',     backref='user', lazy=True)
    chat_messages   = db.relationship('ChatMessage',     backref='user', lazy=True)
    share_tokens    = db.relationship('ShareToken',      backref='user', lazy=True)
    stats           = db.relationship('UserStats', backref='user', uselist=False, lazy=True)


class AssessmentResult(db.Model):
    id        = db.Column(db.Integer, primary_key=True)
    score     = db.Column(db.Float, nullable=False)
    # B3: use timezone-aware callable — datetime.utcnow is deprecated in Python 3.12+
    timestamp = db.Column(db.DateTime(timezone=True), default=_utcnow)
    user_id   = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)


class JournalEntry(db.Model):
    id                 = db.Column(db.Integer, primary_key=True)
    content            = db.Column(db.Text, nullable=False)
    # Sentiment scores from VADER: -1 (most negative) to +1 (most positive)
    sentiment_compound = db.Column(db.Float, nullable=False, default=0.0)
    sentiment_label    = db.Column(db.String(20), nullable=False, default='Neutral')
    timestamp          = db.Column(db.DateTime(timezone=True), default=_utcnow)
    user_id            = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)


# ── Phase 5: Gamification ─────────────────────────────────────────────────────

class UserStats(db.Model):
    """Tracks streaks, totals, and earned badges per user."""
    id                = db.Column(db.Integer, primary_key=True)
    user_id           = db.Column(db.Integer, db.ForeignKey('user.id'), unique=True, nullable=False)
    streak_days       = db.Column(db.Integer, default=0, nullable=False)
    longest_streak    = db.Column(db.Integer, default=0, nullable=False)
    last_journal_date = db.Column(db.Date, nullable=True)  # date of most recent journal
    total_journals    = db.Column(db.Integer, default=0, nullable=False)
    total_assessments = db.Column(db.Integer, default=0, nullable=False)
    # Comma-separated list of earned badge ids, e.g. "first_journal,streak_3"
    badges_earned     = db.Column(db.Text, default='', nullable=False)

    def get_badges(self):
        """Return list of earned badge ids."""
        if not self.badges_earned:
            return []
        return [b for b in self.badges_earned.split(',') if b]

    def award_badge(self, badge_id: str):
        """Idempotently add a badge id."""
        current = self.get_badges()
        if badge_id not in current:
            current.append(badge_id)
            self.badges_earned = ','.join(current)
            return True
        return False


# ── Phase 5: CBT Worksheets ───────────────────────────────────────────────────

class ThoughtRecord(db.Model):
    """7-column CBT Thought Record."""
    id               = db.Column(db.Integer, primary_key=True)
    situation        = db.Column(db.Text, nullable=False)      # What happened?
    emotions         = db.Column(db.String(200), nullable=False)  # e.g. "Anxious 80%, Sad 60%"
    hot_thought      = db.Column(db.Text, nullable=False)      # The most distressing thought
    evidence_for     = db.Column(db.Text, nullable=True)       # Evidence that supports it
    evidence_against = db.Column(db.Text, nullable=True)       # Evidence that challenges it
    balanced_thought = db.Column(db.Text, nullable=True)       # Reframed, balanced thought
    outcome_mood     = db.Column(db.Integer, nullable=True)    # Mood after (1-10)
    timestamp        = db.Column(db.DateTime(timezone=True), default=_utcnow)
    user_id          = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)


class ActivityLog(db.Model):
    """Behavioural Activation log — tracks mood before/after an activity."""
    id          = db.Column(db.Integer, primary_key=True)
    activity    = db.Column(db.String(300), nullable=False)
    mood_before = db.Column(db.Integer, nullable=False)   # 1-10
    mood_after  = db.Column(db.Integer, nullable=True)    # 1-10 (filled in later optionally)
    notes       = db.Column(db.Text, nullable=True)
    planned_at  = db.Column(db.DateTime(timezone=True), default=_utcnow)
    user_id     = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)


# ── Phase 6: AI Companion & Clinician Sharing ─────────────────────────────────

class ChatMessage(db.Model):
    """Stores AI MindCompanion chatbot conversation history."""
    id              = db.Column(db.Integer, primary_key=True)
    sender          = db.Column(db.String(10), nullable=False)  # 'user' or 'assistant'
    message         = db.Column(db.Text, nullable=False)
    sentiment_label = db.Column(db.String(20), nullable=True, default='Neutral')
    timestamp       = db.Column(db.DateTime(timezone=True), default=_utcnow)
    user_id         = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)


class ShareToken(db.Model):
    """Secure, time-limited token for sharing clinician summaries."""
    id         = db.Column(db.Integer, primary_key=True)
    token      = db.Column(db.String(64), unique=True, nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), default=_utcnow)
    expires_at = db.Column(db.DateTime(timezone=True), nullable=False)
    pin_hash   = db.Column(db.String(255), nullable=True)  # Optional PIN protection
    user_id    = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    def is_valid(self):
        """Check if token is not expired."""
        now = _utcnow()
        # Handle offset-naive vs offset-aware comparison gracefully
        expires = self.expires_at
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        return now < expires

