from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text: str) -> dict:
    """
    Analyzes sentiment of a given text using VADER.
    
    Returns:
        dict with keys:
          - compound: float from -1.0 (most negative) to +1.0 (most positive)
          - label: str 'Positive', 'Negative', or 'Neutral'
          - pos, neg, neu: component scores (0.0 to 1.0)
          - emoji: a relevant emoji for display
    """
    scores = _analyzer.polarity_scores(text[:2000])  # guard: truncate to prevent DoS on huge inputs
    compound = scores['compound']

    if compound >= 0.05:
        label = 'Positive'
        emoji = '😊'
        color = '#10b981'  # green
    elif compound <= -0.05:
        label = 'Negative'
        emoji = '😔'
        color = '#ef4444'  # red
    else:
        label = 'Neutral'
        emoji = '😐'
        color = '#f59e0b'  # yellow

    # Normalize compound score from [-1, 1] to [0, 100] for display
    display_score = round((compound + 1) / 2 * 100, 1)

    return {
        'compound': compound,
        'label': label,
        'emoji': emoji,
        'color': color,
        'display_score': display_score,
        'pos': scores['pos'],
        'neg': scores['neg'],
        'neu': scores['neu'],
    }
