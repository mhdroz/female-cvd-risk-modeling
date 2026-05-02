"""
Feature set definitions and female-specific feature engineering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Feature list constants
# ---------------------------------------------------------------------------

BASE_FEATURES_NO_SEX = [
    "age",
    "education",
    "smoker",
    "cigarettes_per_day",
    "bp_meds",
    "prevalent_stroke",
    "prevalent_hypertension",
    "diabetes",
    "cholesterol_total",
    "systolic_bp",
    "diastolic_bp",
    "bmi",
    "heart_rate",
    "glucose",
]

BASE_FEATURES_WITH_SEX = BASE_FEATURES_NO_SEX + ["sex"]

# Base features used in female-only models (no education, no sex)
FEMALE_BASE_FEATURES = [
    "age",
    "cholesterol_total",
    "systolic_bp",
    "diastolic_bp",
    "glucose",
    "diabetes",
    "bmi",
    "smoker",
    "cigarettes_per_day",
    "bp_meds",
    "prevalent_hypertension",
    "prevalent_stroke",
]

# All 15 engineered female-specific feature names
FEMALE_ENGINEERED_FULL = [
    # Age transitions
    "age_perimenopause",
    "age_postmenopause",
    "age_early_postmeno",
    "years_post_menopause",
    # Cholesterol interactions
    "chol_postmeno_interaction",
    "high_chol_postmeno",
    "chol_age_ratio",
    # Metabolic syndrome
    "obesity",
    "prediabetic_range",
    "hypertension_stage2",
    "metabolic_risk_score",
    # Risk amplification
    "diabetes_postmeno",
    "smoking_postmeno",
    "triple_risk_postmeno",
    "bp_age_interaction",
]

# Lean subset (5 features)
FEMALE_ENGINEERED_LEAN = [
    "age_postmenopause",
    "chol_postmeno_interaction",
    "metabolic_risk_score",
    "diabetes_postmeno",
    "bp_age_interaction",
]


# ---------------------------------------------------------------------------
# FeatureSet dataclass
# ---------------------------------------------------------------------------


@dataclass
class FeatureSet:
    """
    A named collection of features for an experiment.

    Attributes:
        name: Human-readable name for this feature set.
        features: List of column names to use as model inputs.
        engineer_fn: Optional function that adds engineered columns to a DataFrame.
        sex_filter: If set, filter to this sex value (0=female, 1=male) before modeling.
        outcome: Name of the outcome column.
        description: Optional description for reporting.
    """

    name: str
    features: list[str]
    engineer_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None
    sex_filter: Optional[int] = None
    outcome: str = "TenYearCHD"
    description: str = ""


# ---------------------------------------------------------------------------
# Female-specific feature engineering
# ---------------------------------------------------------------------------


def engineer_female_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create all 15 female-specific features.

    Expects a DataFrame (typically already filtered to females).
    Returns a copy with new columns added.
    """
    df = df.copy()

    # 1. Age-based risk transitions (menopause proxy)
    df["age_perimenopause"] = ((df["age"] >= 45) & (df["age"] < 55)).astype(int)
    df["age_postmenopause"] = (df["age"] >= 55).astype(int)
    df["age_early_postmeno"] = ((df["age"] >= 55) & (df["age"] < 65)).astype(int)
    df["years_post_menopause"] = np.maximum(0, df["age"] - 55)

    # 2. Cholesterol-age interactions
    df["chol_postmeno_interaction"] = df["cholesterol_total"] * df["age_postmenopause"]
    df["high_chol_postmeno"] = (
        (df["cholesterol_total"] > 240) & (df["age_postmenopause"] == 1)
    ).astype(int)
    df["chol_age_ratio"] = df["cholesterol_total"] / (df["age"] + 1)

    # 3. Metabolic syndrome indicators
    df["obesity"] = (df["bmi"] > 30).astype(int)
    df["prediabetic_range"] = ((df["glucose"] >= 100) & (df["glucose"] < 126)).astype(
        int
    )
    df["hypertension_stage2"] = (df["systolic_bp"] >= 140).astype(int)
    df["metabolic_risk_score"] = (
        df["obesity"]
        + (df["glucose"] > 100).astype(int)
        + df["hypertension_stage2"]
        + (df["cholesterol_total"] > 200).astype(int)
    )

    # 4. Risk amplification patterns
    df["diabetes_postmeno"] = (
        (df["diabetes"] == 1) & (df["age_postmenopause"] == 1)
    ).astype(int)
    df["smoking_postmeno"] = (
        (df["smoker"] == 1) & (df["age_postmenopause"] == 1)
    ).astype(int)
    df["triple_risk_postmeno"] = (
        (df["cholesterol_total"] > 240)
        & (df["systolic_bp"] >= 140)
        & (df["age_postmenopause"] == 1)
    ).astype(int)
    df["bp_age_interaction"] = df["systolic_bp"] * (df["age"] / 100)

    return df


# ---------------------------------------------------------------------------
# Preset feature sets
# ---------------------------------------------------------------------------


def get_preset_feature_sets() -> dict[str, FeatureSet]:
    """Return a dictionary of pre-defined feature sets, keyed by name."""
    return {
        "baseline_no_sex": FeatureSet(
            name="Baseline (no sex)",
            features=BASE_FEATURES_NO_SEX,
            description="Standard CVD features, sex excluded",
        ),
        "baseline_with_sex": FeatureSet(
            name="Naive (sex as feature)",
            features=BASE_FEATURES_WITH_SEX,
            description="Standard CVD features with sex included naively",
        ),
        "male_only": FeatureSet(
            name="Male-only baseline",
            features=FEMALE_BASE_FEATURES,
            sex_filter=1,
            description="Base features, trained on males only",
        ),
        "female_base": FeatureSet(
            name="Female baseline",
            features=FEMALE_BASE_FEATURES,
            sex_filter=0,
            description="Base features, trained on females only",
        ),
        "female_engineered_full": FeatureSet(
            name="Female engineered (full)",
            features=FEMALE_BASE_FEATURES + FEMALE_ENGINEERED_FULL,
            engineer_fn=engineer_female_features,
            sex_filter=0,
            description="Base + all 15 female-specific engineered features",
        ),
        "female_engineered_lean": FeatureSet(
            name="Female engineered (lean)",
            features=FEMALE_BASE_FEATURES + FEMALE_ENGINEERED_LEAN,
            engineer_fn=engineer_female_features,
            sex_filter=0,
            description="Base + 5 highest-impact engineered features",
        ),
    }
