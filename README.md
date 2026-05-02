# Cardiovascular Risk Prediction & The Fairness Trap

A reproducible experiment demonstrating how standard machine learning 
practices can silently disadvantage women in cardiovascular disease (CVD) 
prediction — and how purpose-built, female-aware features can close that gap.

Read the full write-up on Substack: [YOUR_SUBSTACK_POST_URL]

---

## The Problem

The Framingham Heart Study is the gold standard dataset for 10-year coronary 
heart disease (CHD) prediction. But when you train a model on it the 
conventional way — even one that includes sex as a feature — something quietly 
goes wrong: the model learns to miss women.

Four experiments expose this fairness trap and propose a better path forward.

---

## Experiments

### Experiment 1 — The Fairness Trap

Three models compared on the same data:

| Model | Female Recall | Male Recall | Recall Gap |
|---|---|---|---|
| Baseline (no sex feature) | 63.3% | 72.8% | 9.5pp |
| Naive (sex as feature) | 52.2% | 80.6% | **28.4pp** |
| Fairness-constrained (sex-specific thresholds) | 63.3% | 70.9% | 7.5pp |

Adding sex as a feature tripled the recall gap. The fairness-constrained 
approach — sex-specific decision thresholds — recovered performance without 
retraining the model.

### Experiment 2 — Sex-Stratified Models

Separate logistic regression models trained independently on male and female 
subsets:

| Model | Recall | AUC |
|---|---|---|
| Male baseline | 67.0% | 0.728 |
| Female baseline | 64.4% | 0.708 |

This establishes the ceiling for what a female-specific model can achieve 
using only standard clinical features — and motivates the next two experiments.

### Experiment 3 — Female-Specific Feature Engineering

15 features engineered around female physiology, grouped into four domains:

- **Menopause proxies** — perimenopause flag (45–54), post-menopause flag 
  (55+), years since age 55
- **Cholesterol-age interactions** — cholesterol × post-menopause status, 
  cholesterol/age ratio
- **Metabolic syndrome indicators** — obesity flag, prediabetic glucose range, 
  hypertension stage 2, composite metabolic score
- **Risk amplification** — diabetes + post-menopause, smoking + post-menopause, 
  triple-risk composite

| Feature set | Recall | AUC |
|---|---|---|
| Female baseline (12 features) | 64.4% | 0.708 |
| Full engineered (27 features) | 63.3% | 0.720 |
| Lean engineered (17 features) | 65.6% | 0.723 |

More features introduced collinearity and hurt recall. A lean subset of 5 
non-redundant engineered features outperformed the full set.

### Experiment 4 — Automated Feature Selection (RFECV)

Recursive Feature Elimination with Cross-Validation (scoring by recall) 
selected 8 features from the 27-feature pool:

| Feature | Domain |
|---|---|
| `cholesterol_total` | Lipid panel |
| `systolic_bp` | Blood pressure |
| `cigarettes_per_day` | Lifestyle |
| `prevalent_hypertension` | Medical history |
| `age_postmenopause` | Menopause proxy *(engineered)* |
| `chol_postmeno_interaction` | Cholesterol-menopause interaction *(engineered)* |
| `chol_age_ratio` | Cholesterol-age interaction *(engineered)* |
| `hypertension_stage2` | Metabolic syndrome *(engineered)* |

**Result: 68.9% recall, AUC 0.729** — the strongest result across all 
experiments, using fewer features than the 12-feature baseline. Nearly 7 in 
10 women correctly identified as future CVD cases, approaching the 72.8% male 
recall from Experiment 1.

---

## Repository Structure
```
cardiovascular_demo/
├── framingham_experiments.ipynb   # Main experiment notebook
├── framingham_eda.ipynb           # Exploratory data analysis
├── framingham_heart_study.csv     # Framingham dataset (4,240 participants)
└── cvd_fairness/                  # Python library
├── data_prep.py               # Loading, imputation, standardization
├── feature_engineering.py     # Feature sets and female-specific features
├── modeling.py                # Experiment runner, threshold optimization, RFECV
└── results.py                 # Metrics, comparison tables, fairness visualizations
```
---

## Quickstart

```bash
git clone <this-repo>
cd cardiovascular_demo
pip install pandas numpy scikit-learn matplotlib seaborn
jupyter notebook framingham_experiments.ipynb
```

---

## Dataset

**Framingham Heart Study** — a longitudinal cardiovascular cohort study 
initiated in 1948 in Framingham, Massachusetts.

- 4,240 participants (1,820 male / 2,420 female)
- 15 clinical variables: age, cholesterol, blood pressure, smoking, BMI, 
  glucose, diabetes, hypertension, and more
- Outcome: 10-year coronary heart disease incidence (15.2% prevalence)
- Source: publicly available via 
  [Kaggle](https://www.kaggle.com/datasets/amanajmera1/framingham-heart-study-dataset)

---

## Key Takeaways

- **Adding sex as a raw feature backfires** — it allows the model to take 
  shortcuts based on prevalence differences, not biology
- **Stratification helps, but isn't enough** — sex-stratified models still 
  rely on features that don't capture female-specific risk
- **Female physiology needs female-aware features** — menopause transitions, 
  cholesterol-age dynamics, and metabolic patterns matter
- **Fewer, better features beat more features** — RFECV with 8 features 
  outperformed a 27-feature set
- **The proxy has a ceiling** — age-derived menopause features improve recall, 
  but direct hormonal measurements (FSH, estradiol, AMH) would represent a 
  fundamentally different class of model

---

## Connect

- Full write-up and analysis: [YOUR_SUBSTACK_URL]
- Personal website: [[Personal website](https://www.mhd-ai.com/)]  
- LinkedIn: [[LinkedIn](https://www.linkedin.com/in/marie-humbert-droz/)]

Feedback, questions, or collaboration ideas are welcome — open an issue or 
reach out directly.
