"""
resources.py  —  Personalized Insights & Recommendation Engine
---------------------------------------------------------------
Given a mental fitness score (0-100) and the extracted feature dict,
returns tailored recommendations across breathing, grounding, CBT,
and lifestyle categories.
"""

# ── Disorder thresholds that trigger specific recommendations ──
# (feature_key, threshold_percent_above_which_to_trigger)
DISORDER_THRESHOLDS = {
    "Anxiety_disorders":     5.0,
    "Depressive_disorders":  4.0,
    "Bipolar_disorder":      1.2,
    "Schizophrenia":         0.5,
    "Eating_disorders":      0.8,
    "Drug_use_disorders":    1.5,
    "Alcohol_use_disorders": 2.0,
}


def get_score_band(score: float) -> str:
    if score >= 75:
        return "excellent"
    elif score >= 55:
        return "good"
    elif score >= 35:
        return "moderate"
    else:
        return "low"


def get_recommendations(score: float, features: dict) -> dict:
    """
    Returns a structured recommendations dict.

    Fields:
      score_band    : 'excellent' | 'good' | 'moderate' | 'low'
      headline      : str  — top-level message
      color         : hex string
      emoji         : str
      breathing     : list[dict]  — breathing exercises
      grounding     : list[dict]  — grounding techniques
      cbt_prompts   : list[str]   — journaling / reflection prompts
      lifestyle     : list[str]   — general lifestyle tips
      crisis_needed : bool        — show crisis banner if True
      triggered     : list[str]   — which disorders are elevated
    """
    band  = get_score_band(score)
    triggered = [
        k for k, thresh in DISORDER_THRESHOLDS.items()
        if features.get(k, 0) >= thresh
    ]

    # ── Top-level message ──
    headlines = {
        "excellent": "You're thriving! Keep nurturing your mental wellness.",
        "good":      "You're doing well. A few mindful habits can take you further.",
        "moderate":  "Your mind needs some extra care. These exercises can help.",
        "low":       "You're going through a tough time. You're not alone — please try these and seek support.",
    }
    colors = {
        "excellent": "#10b981",
        "good":      "#3b82f6",
        "moderate":  "#f59e0b",
        "low":       "#ef4444",
    }
    emojis = {"excellent": "🌟", "good": "💙", "moderate": "🌤️", "low": "💙"}

    # ── Breathing exercises ──
    breathing = [
        {
            "name":        "Box Breathing",
            "description": "Inhale 4s → Hold 4s → Exhale 4s → Hold 4s. Repeat 4 cycles.",
            "duration":    "4 minutes",
            "best_for":    "Stress & anxiety relief",
            "icon":        "⬛",
            "phases":      [4, 4, 4, 4],
            "labels":      ["Inhale", "Hold", "Exhale", "Hold"],
        },
        {
            "name":        "4-7-8 Breathing",
            "description": "Inhale 4s → Hold 7s → Exhale 8s. Calms the nervous system.",
            "duration":    "3 minutes",
            "best_for":    "Sleep & panic reduction",
            "icon":        "🌙",
            "phases":      [4, 7, 8, 0],
            "labels":      ["Inhale", "Hold", "Exhale", ""],
        },
        {
            "name":        "Coherent Breathing",
            "description": "Inhale 5s → Exhale 5s. Balances heart rate variability.",
            "duration":    "5 minutes",
            "best_for":    "General calm & focus",
            "icon":        "🌊",
            "phases":      [5, 0, 5, 0],
            "labels":      ["Inhale", "", "Exhale", ""],
        },
    ]

    # ── Grounding techniques (context-sensitive) ──
    grounding_all = [
        {
            "name":        "5-4-3-2-1 Grounding",
            "description": "Name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste.",
            "icon":        "👁️",
            "tags":        ["anxiety", "panic", "dissociation"],
        },
        {
            "name":        "Cold Water Reset",
            "description": "Splash cold water on your face or hold an ice cube. Activates the dive reflex and rapidly lowers heart rate.",
            "icon":        "💧",
            "tags":        ["panic", "anxiety", "anger"],
        },
        {
            "name":        "Body Scan Meditation",
            "description": "Close your eyes and slowly bring attention from your feet to the top of your head, releasing tension as you go.",
            "icon":        "🧘",
            "tags":        ["depression", "anxiety", "stress"],
        },
        {
            "name":        "Safe Place Visualization",
            "description": "Close your eyes and vividly imagine a calm, safe place. Engage all senses — what do you see, hear, smell?",
            "icon":        "🏡",
            "tags":        ["trauma", "depression", "anxiety"],
        },
        {
            "name":        "Progressive Muscle Relaxation",
            "description": "Tense each muscle group for 5 seconds, then release. Work from feet to face. Releases physical stress.",
            "icon":        "💪",
            "tags":        ["stress", "anxiety", "sleep"],
        },
    ]

    # Build a tag set from triggered disorders for clean set-intersection matching
    triggered_tags = set()
    for t in triggered:
        # e.g. "Anxiety_disorders" → "anxiety", "Drug_use_disorders" → "drug use"
        base = t.lower().replace("_disorders", "").replace("_", " ").strip()
        triggered_tags.add(base)
        # Also add individual words so "drug use" matches tag "drug"
        triggered_tags.update(base.split())

    grounding = []
    seen = set()
    for g in grounding_all:
        if g['name'] in seen:
            continue
        tag_match = bool(triggered_tags.intersection(g["tags"]))
        if len(grounding) < 3 or tag_match:
            grounding.append(g)
            seen.add(g['name'])

    # ── CBT Journal Prompts ──
    cbt_base = [
        "What is one thought that has been weighing on you? Write it down, then challenge it: is it a fact or an assumption?",
        "List 3 things you are grateful for today, no matter how small.",
        "What would you tell a close friend who was going through exactly what you're experiencing right now?",
        "Describe a recent situation that triggered a strong emotion. What were you thinking, and is there another way to interpret it?",
        "What is one small action you can take today to move toward feeling better?",
    ]
    cbt_anxiety = [
        "Write down your biggest worry. Now write the realistic best case, worst case, and most likely outcome.",
        "What physical sensations do you notice when you feel anxious? Where in your body do they live?",
    ]
    cbt_depression = [
        "List one activity that used to bring you joy. What is stopping you from doing it today? How could you take a 5-minute version of it?",
        "Write about a time when you overcame something difficult. What strengths did you use?",
    ]

    prompts = list(cbt_base)
    if "Anxiety_disorders" in triggered:
        prompts = cbt_anxiety + prompts
    if "Depressive_disorders" in triggered:
        prompts = cbt_depression + prompts

    # ── Lifestyle tips ──
    lifestyle_base = [
        "🌅 Aim for 7–9 hours of sleep at a consistent bedtime.",
        "🚶 Take a 20-minute walk outdoors — sunlight and movement both reduce cortisol.",
        "📵 Schedule one phone-free hour per day, especially before bed.",
        "🥗 Eat at least one nutrient-rich meal per day with vegetables and protein.",
        "👥 Reach out to one person you trust today, even just to say hello.",
    ]
    lifestyle_extra = []
    if "Alcohol_use_disorders" in triggered:
        lifestyle_extra.append("🍶 Try replacing one alcoholic drink with sparkling water or herbal tea this week.")
    if "Drug_use_disorders" in triggered:
        lifestyle_extra.append("💊 Consider speaking to a GP or counsellor about substance use — help is available without judgment.")
    if "Eating_disorders" in triggered:
        lifestyle_extra.append("🍽️ Consider working with a registered dietitian — eating disorders are medical conditions, not willpower issues.")

    lifestyle = lifestyle_extra + lifestyle_base

    return {
        "score_band":    band,
        "headline":      headlines[band],
        "color":         colors[band],
        "emoji":         emojis[band],
        "breathing":     breathing,
        "grounding":     grounding,
        "cbt_prompts":   prompts[:6],
        "lifestyle":     lifestyle,
        "crisis_needed": score < 30 or "Schizophrenia" in triggered,
        "triggered":     triggered,
    }
