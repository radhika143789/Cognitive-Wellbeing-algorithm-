"""
questionnaire_extractor.py
--------------------------
Maps clinical questionnaire answers (0-4 scale) into the 7 disorder
prevalence estimates expected by the Random Forest model, optionally
enriched by a journal sentiment score.

Answer scale: 0=Never, 1=Rarely, 2=Sometimes, 3=Often, 4=Always

Feature output (mirrors training data ranges):
  - Schizophrenia       : 0.10 – 1.00 %
  - Bipolar_disorder    : 0.50 – 2.00 %
  - Eating_disorders    : 0.10 – 1.50 %
  - Anxiety_disorders   : 2.00 – 8.00 %
  - Drug_use_disorders  : 0.20 – 3.00 %
  - Depressive_disorders: 2.00 – 7.00 %
  - Alcohol_use_disorders: 0.50 – 5.00 %
"""


QUESTIONS = [
    # --- SCHIZOPHRENIA indicators (Q1–Q2) ---
    {
        "id": "sq1",
        "group": "schizophrenia",
        "text": "Do you ever hear voices or see things that others around you do not seem to notice?",
        "options": ["Never", "Rarely", "Sometimes", "Often", "Always"],
    },
    {
        "id": "sq2",
        "group": "schizophrenia",
        "text": "Do you feel that your thoughts are being controlled by an outside force, or that others can read your mind?",
        "options": ["Never", "Rarely", "Sometimes", "Often", "Always"],
    },

    # --- BIPOLAR DISORDER indicators (Q3–Q4) ---
    {
        "id": "bq1",
        "group": "bipolar",
        "text": "Do you experience extreme swings between very high energy/euphoria and very low mood within the same week or month?",
        "options": ["Never", "Rarely", "Sometimes", "Often", "Always"],
    },
    {
        "id": "bq2",
        "group": "bipolar",
        "text": "Do you have periods where you need very little sleep but still feel fully energized and unusually productive?",
        "options": ["Never", "Rarely", "Sometimes", "Often", "Always"],
    },

    # --- EATING DISORDER indicators (Q5–Q7) ---
    {
        "id": "eq1",
        "group": "eating",
        "text": "Do you severely restrict your food intake due to fear of weight gain?",
        "options": ["Never", "Rarely", "Sometimes", "Often", "Always"],
    },
    {
        "id": "eq2",
        "group": "eating",
        "text": "Do you experience episodes of eating large amounts of food in a short time and feeling out of control?",
        "options": ["Never", "Rarely", "Sometimes", "Often", "Always"],
    },
    {
        "id": "eq3",
        "group": "eating",
        "text": "Does your body image significantly affect how you feel about yourself overall?",
        "options": ["Never", "Rarely", "Sometimes", "Often", "Always"],
    },

    # --- ANXIETY DISORDER indicators (Q8–Q10, GAD-7 inspired) ---
    {
        "id": "aq1",
        "group": "anxiety",
        "text": "How often do you feel excessive, uncontrollable worry about everyday things?",
        "options": ["Never", "Rarely", "Sometimes", "Often", "Always"],
    },
    {
        "id": "aq2",
        "group": "anxiety",
        "text": "Do you feel restless, keyed up, or on edge?",
        "options": ["Never", "Rarely", "Sometimes", "Often", "Always"],
    },
    {
        "id": "aq3",
        "group": "anxiety",
        "text": "Does anxiety make it hard for you to concentrate or remember things?",
        "options": ["Never", "Rarely", "Sometimes", "Often", "Always"],
    },

    # --- DRUG USE DISORDER indicators (Q11–Q12) ---
    {
        "id": "dq1",
        "group": "drug",
        "text": "How frequently do you use recreational drugs (cannabis, stimulants, opioids, etc.)?",
        "options": ["Never", "Rarely (a few times a year)", "Sometimes (monthly)", "Often (weekly)", "Almost daily"],
    },
    {
        "id": "dq2",
        "group": "drug",
        "text": "Do you feel a strong craving or urge to use drugs that is difficult to control?",
        "options": ["Never", "Rarely", "Sometimes", "Often", "Always"],
    },

    # --- DEPRESSIVE DISORDER indicators (Q13–Q15, PHQ-9 inspired) ---
    {
        "id": "pq1",
        "group": "depressive",
        "text": "How often do you feel persistently sad, empty, or hopeless?",
        "options": ["Never", "Rarely", "Sometimes", "Often", "Always"],
    },
    {
        "id": "pq2",
        "group": "depressive",
        "text": "Have you lost interest or pleasure in activities you once enjoyed?",
        "options": ["Never", "Rarely", "Sometimes", "Often", "Always"],
    },
    {
        "id": "pq3",
        "group": "depressive",
        "text": "Do you struggle with low energy, fatigue, or feeling slowed down nearly every day?",
        "options": ["Never", "Rarely", "Sometimes", "Often", "Always"],
    },

    # --- ALCOHOL USE DISORDER indicators (Q16–Q18, AUDIT inspired) ---
    {
        "id": "alq1",
        "group": "alcohol",
        "text": "How often do you have a drink containing alcohol?",
        "options": ["Never", "Monthly or less", "2-4 times/month", "2-3 times/week", "4+ times/week"],
    },
    {
        "id": "alq2",
        "group": "alcohol",
        "text": "How often during the last year have you been unable to stop drinking once you started?",
        "options": ["Never", "Less than monthly", "Monthly", "Weekly", "Daily or almost daily"],
    },
    {
        "id": "alq3",
        "group": "alcohol",
        "text": "How often have you felt guilt or remorse after drinking?",
        "options": ["Never", "Less than monthly", "Monthly", "Weekly", "Daily or almost daily"],
    },
]


def _normalize(raw_sum: float, n_questions: int, lo: float, hi: float) -> float:
    """
    Converts a raw question score sum into a prevalence estimate
    within the expected model range [lo, hi].
    """
    max_possible = n_questions * 4.0
    ratio = raw_sum / max_possible if max_possible > 0 else 0.0
    return lo + ratio * (hi - lo)


def extract_features(form_data: dict, journal_sentiment: float = 0.0) -> dict:
    """
    Parameters
    ----------
    form_data : dict
        {question_id: int_answer_0_to_4}
    journal_sentiment : float
        Mean VADER compound score from recent journal entries (-1 to +1).
        Positive values dampen anxiety/depression; negative values amplify them.

    Returns
    -------
    dict of feature_name → float (prevalence %)
    """
    groups = {
        "schizophrenia": [],
        "bipolar": [],
        "eating": [],
        "anxiety": [],
        "drug": [],
        "depressive": [],
        "alcohol": [],
    }

    for q in QUESTIONS:
        try:
            val = int(form_data.get(q["id"], 0))
        except (ValueError, TypeError):
            # Q6: Tampered / non-numeric form field — default to 0 (no symptom)
            val = 0
        val = max(0, min(4, val))        # clamp to [0, 4]
        groups[q["group"]].append(val)

    # Map each group to a prevalence estimate
    schizophrenia  = _normalize(sum(groups["schizophrenia"]),  2, 0.10, 1.00)
    bipolar        = _normalize(sum(groups["bipolar"]),         2, 0.50, 2.00)
    eating         = _normalize(sum(groups["eating"]),          3, 0.10, 1.50)
    anxiety        = _normalize(sum(groups["anxiety"]),         3, 2.00, 8.00)
    drug           = _normalize(sum(groups["drug"]),            2, 0.20, 3.00)
    depressive     = _normalize(sum(groups["depressive"]),      3, 2.00, 7.00)
    alcohol        = _normalize(sum(groups["alcohol"]),         3, 0.50, 5.00)

    # --- Journal Sentiment Enrichment ---
    # Compound in [-1, +1]. Negative mood → higher anxiety & depression.
    # Weight: up to ±20% adjustment on anxiety and depression.
    sentiment_factor = -journal_sentiment  # flip sign: negative journal → higher disorder
    anxiety    = min(8.00, max(2.00, anxiety    * (1.0 + 0.20 * sentiment_factor)))
    depressive = min(7.00, max(2.00, depressive * (1.0 + 0.20 * sentiment_factor)))

    return {
        "Schizophrenia":        round(schizophrenia, 4),
        "Bipolar_disorder":     round(bipolar,        4),
        "Eating_disorders":     round(eating,         4),
        "Anxiety_disorders":    round(anxiety,        4),
        "Drug_use_disorders":   round(drug,           4),
        "Depressive_disorders": round(depressive,     4),
        "Alcohol_use_disorders":round(alcohol,        4),
    }
