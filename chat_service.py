"""
chat_service.py — AI MindCompanion Conversation Engine

Provides empathetic, CBT-grounded AI conversational logic for MindTrack.
Integrates VADER sentiment analysis, safety detection for crisis terms,
and personalized cognitive reframing prompts.
"""

import re
import random
from nlp_service import analyze_sentiment

# Crisis keywords that trigger immediate compassionate safety escalation
CRISIS_KEYWORDS = [
    r'\bsuicide\b', r'\bsuicidal\b', r'\bkill myself\b', r'\bend my life\b',
    r'\bself-harm\b', r'\bwant to die\b', r'\bno reason to live\b',
    r'\bhurt myself\b', r'\bhopeless\b'
]

# Maximum accepted message length (chars) — prevents DoS via large inputs
MAX_MESSAGE_LENGTH = 1000

# Quick response options based on conversation state
DEFAULT_QUICK_RESPONSES = [
    "Tell me more about how you're feeling",
    "Can you give me a CBT grounding exercise?",
    "How can I improve my mental fitness score?",
    "Help me reframe a negative thought"
]

CBT_REFRAMING_PROMPTS = [
    "When you notice that thought, ask yourself: *What is the concrete evidence for and against this idea?*",
    "Is there a more balanced perspective you can take right now?",
    "If a dear friend were in your situation thinking this, what compassionate advice would you give them?",
    "Notice how this thought makes you feel physically. Let's take a slow 4-7-8 breath together."
]

POSITIVE_RESPONSES = [
    "I'm glad to hear you're feeling positive today! What contributed most to your good mood?",
    "That's wonderful! Celebrating small wins and moments of joy builds long-term mental resilience.",
    "Great energy! Writing a quick journal entry about this positive moment can help lock in those positive feelings."
]

NEUTRAL_RESPONSES = [
    "Thank you for sharing that with me. How are you holding up overall today?",
    "I'm here with you. Would you like to explore your thoughts in a journal entry or try a quick breathing session?",
    "Checking in with yourself regularly is a great habit. Is there anything specific on your mind right now?"
]

ANXIETY_RESPONSES = [
    "It sounds like you're experiencing some anxiety or stress right now. Remember that feelings come and go like waves.",
    "When anxiety feels overwhelming, grounding yourself in the present helps. Try noticing 5 things you can see around you right now.",
    "Take a deep breath with me. Inhale slowly for 4 seconds, hold for 7, and exhale for 8."
]

SADNESS_RESPONSES = [
    "I hear you, and it's completely okay to feel down sometimes. Your feelings are valid.",
    "When we feel low, even taking one tiny positive step — like taking a brief walk or sipping water — can help.",
    "Would you like to try a CBT Thought Record worksheet to break down what's weighing on you?"
]


def generate_ai_response(user_message: str, history: list | None = None) -> dict:
    """
    Generates an empathetic, CBT-focused response to the user's message.

    Parameters
    ----------
    user_message : str
        The user's input text.
    history : list[dict] | None
        Optional list of previous {'sender', 'message'} dicts for context-aware
        responses. Used to detect repeated distress signals across turns.

    Returns
    -------
    dict with keys:
        'response'        : str  — the AI reply text
        'sentiment_label' : str  — 'Positive', 'Neutral', or 'Negative'
        'is_crisis'       : bool — True if crisis keywords detected
        'quick_replies'   : list[str]
    """
    # Truncate input to guard against oversized payloads
    clean_msg = user_message.strip()[:MAX_MESSAGE_LENGTH]

    if not clean_msg:
        return {
            'response': "I didn't catch that. How are you feeling right now?",
            'sentiment_label': 'Neutral',
            'is_crisis': False,
            'quick_replies': DEFAULT_QUICK_RESPONSES
        }

    # 1. Safety check for crisis keywords (before any other processing)
    for pattern in CRISIS_KEYWORDS:
        if re.search(pattern, clean_msg, re.IGNORECASE):
            return {
                'response': (
                    "💙 **You matter, and you don't have to carry this alone.**\n\n"
                    "If you are feeling overwhelmed or having thoughts of self-harm, please reach out to a trusted professional immediately:\n"
                    "• **iCall India**: 9152987821\n"
                    "• **Vandrevala Foundation**: 1860-2662-345\n"
                    "• **988 Lifeline (US/Canada)**: Call or text 988\n\n"
                    "You can also visit our [Crisis Support Page](/crisis) anytime for immediate grounding exercises and 24/7 helplines."
                ),
                'sentiment_label': 'Negative',
                'is_crisis': True,
                'quick_replies': ["Go to Crisis Support", "Try 4-7-8 Breathing", "I want to talk more"]
            }

    # 2. NLP Sentiment Analysis
    sentiment = analyze_sentiment(clean_msg)
    label     = sentiment['label']
    compound  = sentiment['compound']  # -1.0 to +1.0

    # 3. Context-awareness: check if user has expressed distress in recent history
    recent_negative = False
    if history:
        recent_msgs = [h['message'] for h in history[-4:] if h.get('sender') == 'user']
        recent_negative = any(
            re.search(p, m, re.IGNORECASE)
            for m in recent_msgs
            for p in [r'\bsad\b', r'\bdepressed\b', r'\banxious\b', r'\bworried\b']
        )

    # 4. Keyword / Intent Matching (priority ordered)
    msg_lower = clean_msg.lower()

    if any(k in msg_lower for k in ['breath', 'breathing', 'anxious', 'panic', 'stress', 'overwhelmed']):
        reply   = random.choice(ANXIETY_RESPONSES) + " " + random.choice(CBT_REFRAMING_PROMPTS)
        replies = ["Start 4-7-8 Breathing", "Try Box Breathing", "Write a Thought Record"]

    elif any(k in msg_lower for k in ['sad', 'depressed', 'lonely', 'unhappy', 'tired', 'crying']):
        reply   = random.choice(SADNESS_RESPONSES) + " " + random.choice(CBT_REFRAMING_PROMPTS)
        replies = ["Write in Journal", "CBT Behavioural Planner", "Take Assessment"]

    elif any(k in msg_lower for k in ['cbt', 'thought', 'reframe', 'cognitive', 'worksheet']):
        reply = (
            "Cognitive Behavioral Therapy (CBT) helps us identify and reframe unhelpful thoughts. "
            "Our [CBT Worksheets Hub](/cbt) features a 7-column Thought Record and a Behavioural Activation Planner!"
        )
        replies = ["Open CBT Worksheets", "How do I reframe a thought?", "Take Assessment"]

    elif any(k in msg_lower for k in ['score', 'fitness', 'assessment', 'predict']):
        reply = (
            "Your Mental Fitness Score (0-100) combines your clinical questionnaire responses and recent journal sentiment. "
            "You can complete a quick 3-minute assessment at any time on our [Assessment Page](/assess)!"
        )
        replies = ["Take Assessment", "View My Dashboard", "Write in Journal"]

    elif label == 'Positive' and not recent_negative:
        reply   = random.choice(POSITIVE_RESPONSES)
        replies = ["Write in Journal", "Check Achievements", "Take Assessment"]

    elif label == 'Negative' or compound < -0.2 or recent_negative:
        # Use compound threshold for more nuanced detection
        reply   = random.choice(SADNESS_RESPONSES)
        replies = ["Try 4-7-8 Breathing", "Write a Thought Record", "Talk to MindCompanion"]

    else:
        reply   = random.choice(NEUTRAL_RESPONSES)
        replies = DEFAULT_QUICK_RESPONSES

    return {
        'response': reply,
        'sentiment_label': label,
        'is_crisis': False,
        'quick_replies': replies
    }
