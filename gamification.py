"""
gamification.py — Badge Definitions & Streak Engine
----------------------------------------------------
Defines all achievement badges and the logic to evaluate which
ones a user has earned based on their UserStats record.
"""

from datetime import date


# ── Badge Definitions ─────────────────────────────────────────────────────────
# Each badge: id → {name, emoji, description, tier}
# tier: 'bronze' | 'silver' | 'gold' | 'platinum'

BADGE_DEFINITIONS = {
    # Journal streaks
    "first_journal": {
        "name": "First Entry",
        "emoji": "📓",
        "description": "Wrote your very first journal entry.",
        "tier": "bronze",
    },
    "streak_3": {
        "name": "3-Day Streak",
        "emoji": "🔥",
        "description": "Journaled 3 days in a row.",
        "tier": "bronze",
    },
    "streak_7": {
        "name": "Week Warrior",
        "emoji": "⚡",
        "description": "Maintained a 7-day journaling streak.",
        "tier": "silver",
    },
    "streak_30": {
        "name": "Iron Mind",
        "emoji": "💎",
        "description": "Achieved a 30-day journaling streak.",
        "tier": "platinum",
    },
    # Assessment milestones
    "first_assessment": {
        "name": "Self-Aware",
        "emoji": "🎯",
        "description": "Completed your first mental fitness assessment.",
        "tier": "bronze",
    },
    "assessments_5": {
        "name": "Committed",
        "emoji": "📈",
        "description": "Completed 5 mental fitness assessments.",
        "tier": "silver",
    },
    "assessments_10": {
        "name": "Dedicated",
        "emoji": "🏆",
        "description": "Completed 10 mental fitness assessments.",
        "tier": "gold",
    },
    # Score-based
    "score_80": {
        "name": "High Performer",
        "emoji": "🌟",
        "description": "Scored 80 or above on a mental fitness assessment.",
        "tier": "gold",
    },
    "score_95": {
        "name": "Peak Wellness",
        "emoji": "🚀",
        "description": "Scored 95 or above — exceptional mental wellness!",
        "tier": "platinum",
    },
    # Journaling volume
    "journals_10": {
        "name": "Reflective",
        "emoji": "✍️",
        "description": "Wrote 10 journal entries.",
        "tier": "silver",
    },
    "journals_30": {
        "name": "Deep Thinker",
        "emoji": "🧠",
        "description": "Wrote 30 journal entries.",
        "tier": "gold",
    },
}


TIER_ORDER = {"bronze": 0, "silver": 1, "gold": 2, "platinum": 3}
TIER_COLORS = {
    "bronze":   "#cd7f32",
    "silver":   "#c0c0c0",
    "gold":     "#ffd700",
    "platinum": "#e8e8f0",
}


# ── Streak Engine ─────────────────────────────────────────────────────────────

def update_streak(stats, today: date = None) -> bool:
    """
    Update streak_days for a UserStats object given a journal submission today.
    Returns True if a new streak milestone badge should be re-evaluated.

    Rules:
      - If last_journal_date is today → already counted, no change
      - If last_journal_date is yesterday → increment streak
      - Otherwise → reset streak to 1
    """
    if today is None:
        today = date.today()

    if stats.last_journal_date == today:
        # Already journaled today; no change
        return False

    from datetime import timedelta
    yesterday = today - timedelta(days=1)

    if stats.last_journal_date == yesterday:
        stats.streak_days += 1
    else:
        stats.streak_days = 1

    # Track longest streak ever
    if stats.streak_days > stats.longest_streak:
        stats.longest_streak = stats.streak_days

    stats.last_journal_date = today
    stats.total_journals += 1
    return True


def evaluate_badges(stats, latest_score: float = None) -> list:
    """
    Check all badge conditions against the given UserStats.
    Awards any newly earned badges (idempotent via UserStats.award_badge).
    Returns list of newly awarded badge ids (for flash messages).
    """
    newly_awarded = []

    def check(badge_id, condition: bool):
        if condition:
            if stats.award_badge(badge_id):
                newly_awarded.append(badge_id)

    # Journal streaks
    check("first_journal",    stats.total_journals >= 1)
    check("streak_3",         stats.streak_days >= 3)
    check("streak_7",         stats.streak_days >= 7)
    check("streak_30",        stats.streak_days >= 30)

    # Assessment milestones
    check("first_assessment", stats.total_assessments >= 1)
    check("assessments_5",    stats.total_assessments >= 5)
    check("assessments_10",   stats.total_assessments >= 10)

    # Score-based (only if a score was just computed)
    if latest_score is not None:
        check("score_80", latest_score >= 80)
        check("score_95", latest_score >= 95)

    # Volume
    check("journals_10", stats.total_journals >= 10)
    check("journals_30", stats.total_journals >= 30)

    return newly_awarded


def get_or_create_stats(db, user_id):
    """Fetch UserStats for a user, creating a blank record if needed."""
    from models import UserStats
    stats = UserStats.query.filter_by(user_id=user_id).first()
    if stats is None:
        stats = UserStats(user_id=user_id)
        db.session.add(stats)
        db.session.flush()  # get an id without committing
    return stats


def get_badge_details(badge_ids: list) -> list:
    """
    Given a list of earned badge ids, return full badge detail dicts
    sorted by tier (platinum first) for display.
    """
    details = []
    for bid in badge_ids:
        if bid in BADGE_DEFINITIONS:
            detail = dict(BADGE_DEFINITIONS[bid])
            detail["id"]    = bid
            detail["color"] = TIER_COLORS[detail["tier"]]
            details.append(detail)
    # Sort platinum → gold → silver → bronze
    details.sort(key=lambda b: -TIER_ORDER[b["tier"]])
    return details
