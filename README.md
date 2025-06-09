# Cognitive-Wellbeing-algorithm


A data-driven project that leverages machine learning to track and analyze mental fitness based on global mental health disorder data. This project includes data cleaning, visualization, and predictive modeling to assess cognitive well-being trends worldwide.

---

## 📌 Problem Statement

Mental health is a vital component of overall well-being. With rising cases of disorders like anxiety, depression, and substance abuse globally, it is essential to analyze trends and understand their impact. This project aims to:

- Analyze the prevalence of mental and substance use disorders.
- Explore correlations with overall mental fitness.
- Build machine learning models to predict mental fitness based on health indicators.

---

## 📂 Datasets Used

1. [`prevalence-by-mental-and-substance-use-disorder.csv`](https://ourworldindata.org/)
2. [`mental-and-substance-use-as-share-of-disease.csv`](https://ourworldindata.org/)

These datasets include:
- Mental disorder prevalence (schizophrenia, depression, etc.)
- Substance use data (alcohol, drug use)
- Year-wise and country-wise records
- Combined indicators of cognitive and mental fitness

---

## 🛠️ Libraries and Tools

- `numpy`, `pandas` – Data manipulation
- `matplotlib`, `seaborn`, `plotly` – Data visualization
- `sklearn` – Machine learning (regression models)
- `LabelEncoder`, `train_test_split`, `LinearRegression`, `RandomForestRegressor`

---

## 📊 Exploratory Data Analysis

- Merged datasets for unified analysis.
- Performed null checks and cleaned data.
- Visualized patterns using:
  - Correlation heatmaps
  - Pair plots
  - Time-series line charts
  - Pie charts for year-wise distribution

### 🔍 Key Insight:
> Eating disorders and anxiety are strongly correlated with mental fitness, indicating dietary and psychological habits significantly affect well-being.

---

## 🤖 Machine Learning Models

**Target Variable:** `mental_fitness`

### Models Used:
- **Linear Regression**
- **Random Forest Regressor**

### Model Performance:

| Model                  | Dataset | R² Score | RMSE      |
|------------------------|---------|----------|-----------|
| Linear Regression      | Train   | ~0.90    | High      |
| Linear Regression      | Test    | ~0.89    | Medium    |
| Random Forest Regressor| Train   | ~0.99    | Very Low  |
| Random Forest Regressor| Test    | ~0.95    | Low       |

---

## 🧾 Results & Conclusion

- Random Forest outperformed Linear Regression, showing better accuracy in mental fitness predictions.
- Strong correlations observed between eating habits, anxiety, and overall cognitive health.
- This model can form the basis for building future mental health support tools and dashboards.

---

## 📁 Project Structure

```

.
├── Cognitive\_Wellbeing\_Monitoring.ipynb  # Main notebook
├── data/
│   ├── prevalence-by-mental-and-substance-use-disorder.csv
│   └── mental-and-substance-use-as-share-of-disease.csv
├── README.md
└── AI\_Mental\_Fitness\_Tracker\_Report.pdf

```

---

## 🚀 Future Improvements

- Deploy model using Streamlit or Flask.
- Integrate real-time data collection via surveys or APIs.
- Expand to classification (mental risk levels) and time-series forecasting.

---

## 👩‍💻 Author

**Radhika Khattar**  
💼 MCA Graduate | Full Stack Developer | Cybersecurity & AI Intern  
🔗 [LinkedIn](https://www.linkedin.com/in/radhika-khattar)

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).

