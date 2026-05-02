"""
Data loading, imputation, and column standardization for Framingham Heart Study.
"""

import pandas as pd

OUTCOME_COL = "TenYearCHD"

# Framingham CSV column -> standardized name
COLUMN_RENAME_MAP = {
    "male": "sex",
    "totChol": "cholesterol_total",
    "sysBP": "systolic_bp",
    "diaBP": "diastolic_bp",
    "currentSmoker": "smoker",
    "cigsPerDay": "cigarettes_per_day",
    "BPMeds": "bp_meds",
    "BMI": "bmi",
    "heartRate": "heart_rate",
    "prevalentStroke": "prevalent_stroke",
    "prevalentHyp": "prevalent_hypertension",
}

CATEGORICAL_COLS = [
    "education",
    "currentSmoker",
    "BPMeds",
    "diabetes",
    "prevalentStroke",
    "prevalentHyp",
]

CONTINUOUS_COLS = [
    "age",
    "totChol",
    "sysBP",
    "diaBP",
    "BMI",
    "heartRate",
    "glucose",
]


def load_framingham(path: str = "framingham_heart_study.csv") -> pd.DataFrame:
    """Load raw Framingham CSV."""
    return pd.read_csv(path)


def impute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply imputation strategy (operates on original column names):
    - Drop rows with missing outcome
    - Mode for categorical columns
    - Median for continuous columns
    - Conditional for cigsPerDay (0 for non-smokers, median-of-smokers for smokers)
    """
    df = df.copy()

    # Drop missing outcome
    if OUTCOME_COL in df.columns and df[OUTCOME_COL].isnull().any():
        df = df.dropna(subset=[OUTCOME_COL])

    # Categorical: mode imputation
    for col in CATEGORICAL_COLS:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # Continuous: median imputation
    for col in CONTINUOUS_COLS:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Special case: cigsPerDay conditional on smoking status
    if "cigsPerDay" in df.columns and "currentSmoker" in df.columns:
        if df["cigsPerDay"].isnull().any():
            # Non-smokers -> 0
            non_smoker_mask = (df["currentSmoker"] == 0) & df["cigsPerDay"].isnull()
            df.loc[non_smoker_mask, "cigsPerDay"] = 0

            # Smokers -> median among smokers
            smoker_median = df[df["currentSmoker"] == 1]["cigsPerDay"].median()
            smoker_mask = (df["currentSmoker"] == 1) & df["cigsPerDay"].isnull()
            df.loc[smoker_mask, "cigsPerDay"] = smoker_median

    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to standardized names and ensure sex is 1=male, 0=female."""
    df = df.rename(columns=COLUMN_RENAME_MAP)

    # Lowercase any remaining uppercase column names (e.g. BMI handled by map,
    # but catch stragglers)
    df.columns = [c if c in df.columns else c for c in df.columns]

    # Drop rows with missing sex
    if "sex" in df.columns:
        df = df.dropna(subset=["sex"])

    return df


def prepare_data(path: str = "framingham_heart_study.csv") -> pd.DataFrame:
    """Full pipeline: load -> impute -> standardize. Returns analysis-ready DataFrame."""
    df = load_framingham(path)
    df = impute(df)
    df = standardize_columns(df)
    return df
