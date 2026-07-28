# 🧠 MindTrack — AI-Powered Mental Wellness Platform

> A full-stack Flask web application that uses clinical questionnaires, NLP sentiment analysis, and machine learning to provide personalized mental fitness scores, daily mood journaling, and evidence-based wellness resources.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-000000?style=flat&logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-F7931E?style=flat&logo=scikit-learn)
![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?style=flat&logo=sqlite)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## 📸 Features at a Glance

| Feature | Description |
|---|---|
| 🎯 **Clinical Questionnaire** | 18 PHQ-9/GAD-7/AUDIT-inspired questions extract 7 disorder prevalence scores |
| 🤖 **AI Fitness Score** | Random Forest model (R²=0.83) predicts mental fitness 0–100 |
| 📓 **Mood Journal + NLP** | VADER sentiment analysis enriches ML predictions via journal mood |
| 📈 **Progress Dashboard** | Chart.js interactive line graph tracks score history over time |
| 🌬️ **Breathing Exercises** | Interactive 4-7-8, Box, and Coherent breathing timers |
| 📝 **CBT Prompts** | Evidence-based cognitive behavioural therapy journal prompts |
| 🆘 **Crisis Support** | 24/7 hotlines, grounding exercises, always one click away |
| 🔐 **Secure Auth** | Flask-Login sessions with Werkzeug bcrypt password hashing |

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/radhika143789/Cognitive-Wellbeing-algorithm-.git
cd Cognitive-Wellbeing-algorithm-
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the ML model
```bash
python train_model.py
```
This generates 5,000 synthetic respondents and trains the Random Forest.
Expected output:
```
Generating questionnaire-derived synthetic dataset ...
Training on 4000 samples ...
Test MSE  : ~27
Test R²   : ~0.83
model.pkl saved successfully [OK]
```

### 4. Run the application
```bash
python app.py
```
Open your browser at **http://127.0.0.1:5000**

---

## 📁 Project Structure

```
Cognitive-Wellbeing-algorithm-/
│
├── app.py                      # Flask application & all routes
├── models.py                   # SQLAlchemy DB models (User, AssessmentResult, JournalEntry)
├── nlp_service.py              # VADER NLP sentiment analysis helper
├── questionnaire_extractor.py  # Maps 18 clinical questions → 7 disorder scores
├── resources.py                # Insights engine: score → personalized recommendations
├── train_model.py              # ML training pipeline (Random Forest)
├── requirements.txt            # Python dependencies
├── .gitignore                  # Excludes model.pkl, *.db, __pycache__
│
├── static/
│   ├── css/
│   │   ├── style.css           # Global glassmorphism design system
│   │   ├── landing.css         # Landing page styles
│   │   ├── assess.css          # Questionnaire wizard & result styles
│   │   └── resources.css       # Resources & crisis page styles
│   └── js/
│       └── main.js             # Frontend utility scripts
│
└── templates/
    ├── landing.html            # Public marketing/landing page
    ├── assess.html             # Multi-step clinical questionnaire + result
    ├── journal.html            # Daily mood journal with NLP analysis
    ├── dashboard.html          # Progress chart (Chart.js)
    ├── resources.html          # Personalized wellness exercises & CBT prompts
    ├── crisis.html             # Crisis hotlines + grounding exercises
    ├── login.html              # Login page
    └── register.html           # Registration page
```

---

## 🧬 ML Architecture

### Feature Extraction Pipeline
```
18 Clinical Questions
        ↓
questionnaire_extractor.py
        ↓
7 Disorder Prevalence Estimates (%)
  • Schizophrenia         (0.10 – 1.00%)
  • Bipolar Disorder      (0.50 – 2.00%)
  • Eating Disorders      (0.10 – 1.50%)
  • Anxiety Disorders     (2.00 – 8.00%)
  • Drug Use Disorders    (0.20 – 3.00%)
  • Depressive Disorders  (2.00 – 7.00%)
  • Alcohol Use Disorders (0.50 – 5.00%)
        +
Journal Sentiment Score (VADER compound, –1 to +1)
        ↓
Random Forest Regressor (200 trees)
        ↓
Mental Fitness Score (0 – 100)
```

### Training Data
- **5,000 synthetic respondents** generated via `train_model.py`
- Severity sampled from `Beta(2, 5)` distribution (skewed healthy)
- Journal sentiment correlated inversely with severity + Gaussian noise
- **Test R² = 0.83**, MSE ≈ 27

---

## 🗂️ Phase Breakdown

### Phase 1 — User Authentication & Progress Tracking
- SQLite database with Flask-SQLAlchemy
- Secure registration/login (pbkdf2:sha256 hashing)
- `AssessmentResult` model — stores score + timestamp per user
- Dashboard with Chart.js line graph of historical scores

### Phase 2 — Daily Mood Journal & NLP Sentiment Analysis
- `JournalEntry` model — stores text, VADER compound score, label
- VADER NLP engine (`nlp_service.py`) — Positive / Neutral / Negative
- Animated sentiment result card with breakdown bars (pos/neu/neg %)
- Past entries list with color-coded sentiment badges

### Phase 3 — Clinical Questionnaire + ML Pipeline
- Replaced raw prevalence inputs with 18 guided clinical questions
- `questionnaire_extractor.py` maps answers to 7 WHO disorder categories
- Journal sentiment enriches the anxiety & depression feature estimates
- Random Forest retrained on questionnaire-derived synthetic data
- Multi-step wizard UI with progress bar, animated result ring
- Feature breakdown table on result page

### Phase 4 — Personalized Resources & Crisis Management
- `resources.py` insights engine — score band + triggered disorders → recommendations
- Interactive breathing exercise timers (Box, 4-7-8, Coherent)
- 5-4-3-2-1 grounding techniques (context-matched to triggered disorders)
- CBT reflection prompts linked directly to the journal
- Dedicated `/crisis` page with global hotlines + live box breathing widget
- Crisis banner auto-shown on result page when score < 30

### Phase 5 — Gamification, PDF Reports & Advanced CBT Worksheets
- **Gamification engine** (`gamification.py`) — 11 achievement badges (bronze/silver/gold/platinum)
- **Streak tracking** — daily journal streak, longest streak, total stats all persisted in `UserStats` DB model
- **Enhanced Dashboard** — streak card 🔥, badges showcase, score trend arrow (↑↓→), quick action buttons
- **PDF Report** (`/report`) — print-optimized A4 layout with score history, journal excerpts, badges, and recommendations. Uses browser `window.print()` — zero extra dependencies
- **CBT Worksheet Hub** (`/cbt`) — tabbed interface with:
  - 7-column Thought Record (situation, emotions, hot thought, evidence for/against, balanced thought, outcome mood slider)
  - Behavioural Activation Planner (activity + mood before/after sliders)
  - Past Records history (collapsible detail cards)
- **New DB models**: `UserStats`, `ThoughtRecord`, `ActivityLog`
- **Full QA test suite** (`tests/`) — 70+ tests covering unit (ML, NLP, gamification, resources) and integration (all Flask routes, auth, CBT, report)

---

## 🌐 Routes

| Route | Auth | Description |
|---|---|---|
| `GET /` | Public | Landing / marketing page |
| `GET /assess` | Public | Multi-step questionnaire |
| `POST /assess` | Public | Process answers, run ML, show result |
| `GET /journal` | 🔒 Login | View journal + past entries |
| `POST /journal` | 🔒 Login | Submit entry, run NLP analysis |
| `GET /dashboard` | 🔒 Login | View historical score chart |
| `GET /resources` | Public | Personalized wellness plan |
| `GET /crisis` | Public | Crisis hotlines & grounding |
| `GET /login` | Public | Login form |
| `POST /login` | Public | Authenticate |
| `GET /register` | Public | Registration form |
| `POST /register` | Public | Create account |
| `GET /logout` | 🔒 Login | End session |

---

## ⚙️ Requirements

```
flask
flask-sqlalchemy
flask-login
scikit-learn
pandas
numpy
joblib
vaderSentiment
```

Install with: `pip install -r requirements.txt`

---

## ⚠️ Disclaimer

> MindTrack is an **informational and self-care tool only**. It is **not** a substitute for professional medical advice, diagnosis, or treatment. If you are experiencing a mental health crisis, please contact a qualified healthcare professional or a crisis hotline immediately.

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built with ❤️ as a full-stack ML + Flask project.*
