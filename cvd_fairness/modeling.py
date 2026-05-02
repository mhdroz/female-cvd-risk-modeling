"""
Experiment runner: train/test split, scaling, model fitting, prediction.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV, SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, make_scorer, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from .feature_engineering import FeatureSet


@dataclass
class TrainedExperiment:
    """All artifacts from a single experiment run."""

    name: str
    feature_set: FeatureSet
    model: object
    scaler: StandardScaler
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    sex_train: Optional[pd.Series]
    sex_test: Optional[pd.Series]
    y_pred: np.ndarray
    y_prob: np.ndarray
    thresholds: Optional[dict] = None


def run_experiment(
    df: pd.DataFrame,
    feature_set: FeatureSet,
    test_size: float = 0.3,
    random_state: int = 42,
    model_class=None,
    model_kwargs: Optional[dict] = None,
) -> TrainedExperiment:
    """
    End-to-end experiment: filter -> engineer -> split -> scale -> train -> predict.

    Parameters
    ----------
    df : DataFrame with standardized column names (output of prepare_data).
    feature_set : FeatureSet defining which features, engineering, and filters to use.
    test_size : Fraction of data for testing.
    random_state : Random seed for reproducibility.
    model_class : Sklearn-compatible classifier class (default LogisticRegression).
    model_kwargs : Keyword arguments for the model constructor.
    """
    df = df.copy()

    # 1. Filter by sex if requested
    if feature_set.sex_filter is not None:
        df = df[df["sex"] == feature_set.sex_filter].copy()

    # 2. Apply feature engineering if needed
    if feature_set.engineer_fn is not None:
        df = feature_set.engineer_fn(df)

    # 3. Extract X, y, sex
    X = df[feature_set.features]
    y = df[feature_set.outcome]
    sex = df["sex"] if feature_set.sex_filter is None else None

    # 4. Stratified train/test split
    if sex is not None:
        stratify_var = sex.astype(str) + "_" + y.astype(str)
        X_train, X_test, y_train, y_test, sex_train, sex_test = train_test_split(
            X,
            y,
            sex,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_var,
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
        sex_train, sex_test = None, None

    # 5. Scale
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    # 6. Fit model
    if model_class is None:
        model_class = LogisticRegression
    if model_kwargs is None:
        model_kwargs = dict(random_state=42, max_iter=1000, class_weight="balanced")

    model = model_class(**model_kwargs)
    model.fit(X_train_scaled, y_train)

    # 7. Predict
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    return TrainedExperiment(
        name=feature_set.name,
        feature_set=feature_set,
        model=model,
        scaler=scaler,
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=y_train,
        y_test=y_test,
        sex_train=sex_train,
        sex_test=sex_test,
        y_pred=y_pred,
        y_prob=y_prob,
    )


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_acc: float,
) -> float:
    """Find the decision threshold that achieves closest to target accuracy."""
    best_thresh = 0.5
    best_diff = float("inf")

    for thresh in np.linspace(0, 1, 100):
        y_pred_t = (y_prob >= thresh).astype(int)
        acc = accuracy_score(y_true, y_pred_t)
        diff = abs(acc - target_acc)
        if diff < best_diff:
            best_diff = diff
            best_thresh = thresh

    return best_thresh


def apply_sex_specific_thresholds(
    experiment: TrainedExperiment,
    target_accuracy: Optional[float] = None,
) -> TrainedExperiment:
    """
    Create a new TrainedExperiment with sex-specific decision thresholds.

    Requires experiment.sex_test to be available (full-population experiment).
    If target_accuracy is None, uses the average of male/female accuracy at default threshold.
    """
    if experiment.sex_test is None:
        raise ValueError(
            "Cannot apply sex-specific thresholds to a sex-filtered experiment."
        )

    sex_test = experiment.sex_test
    y_test = experiment.y_test
    y_prob = experiment.y_prob
    y_pred = experiment.y_pred

    # Compute target accuracy if not provided
    if target_accuracy is None:
        male_acc = accuracy_score(
            y_test[sex_test == 1], y_pred[sex_test == 1]
        )
        female_acc = accuracy_score(
            y_test[sex_test == 0], y_pred[sex_test == 0]
        )
        target_accuracy = (male_acc + female_acc) / 2

    # Find sex-specific thresholds
    thresh_male = find_optimal_threshold(
        y_test[sex_test == 1].values,
        y_prob[sex_test == 1],
        target_accuracy,
    )
    thresh_female = find_optimal_threshold(
        y_test[sex_test == 0].values,
        y_prob[sex_test == 0],
        target_accuracy,
    )

    # Apply thresholds
    y_pred_new = np.zeros_like(y_pred, dtype=int)
    y_pred_new[sex_test == 1] = (y_prob[sex_test == 1] >= thresh_male).astype(int)
    y_pred_new[sex_test == 0] = (y_prob[sex_test == 0] >= thresh_female).astype(int)

    # Build new experiment with updated predictions
    new_exp = deepcopy(experiment)
    new_exp.name = f"{experiment.name} (fairness-constrained)"
    new_exp.y_pred = y_pred_new
    new_exp.thresholds = {"Male": thresh_male, "Female": thresh_female}
    return new_exp


@dataclass
class FeatureSelectionResult:
    """Results from feature selection."""

    selected_features: list[str]
    feature_set: FeatureSet
    cv_scores: dict[str, list[float]]  # {n_features: [cv_scores]}
    ranking: pd.DataFrame  # columns: feature, rank (1 = selected)
    method: str


def select_features(
    df: pd.DataFrame,
    feature_set: FeatureSet,
    scoring: str = "recall",
    method: str = "rfecv",
    cv: int = 5,
    min_features: int = 1,
    random_state: int = 42,
    model_class=None,
    model_kwargs: Optional[dict] = None,
) -> FeatureSelectionResult:
    """
    Find the optimal subset of features that maximizes a scoring metric.

    Parameters
    ----------
    df : DataFrame with standardized column names (output of prepare_data).
    feature_set : FeatureSet to select features from.
    scoring : Metric to maximize. Default "recall". Also supports "precision",
              "f1", "accuracy", "roc_auc".
    method : "rfecv" (Recursive Feature Elimination with CV) or
             "sequential" (Sequential Feature Selection, forward).
    cv : Number of cross-validation folds.
    min_features : Minimum number of features to select (rfecv only).
    random_state : Random seed.
    model_class : Sklearn classifier class (default LogisticRegression).
    model_kwargs : Keyword arguments for the model constructor.

    Returns
    -------
    FeatureSelectionResult with the selected features and a new FeatureSet
    ready to pass to run_experiment.
    """
    df = df.copy()

    # Filter and engineer
    if feature_set.sex_filter is not None:
        df = df[df["sex"] == feature_set.sex_filter].copy()
    if feature_set.engineer_fn is not None:
        df = feature_set.engineer_fn(df)

    X = df[feature_set.features]
    y = df[feature_set.outcome]

    # Scale
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    # Model
    if model_class is None:
        model_class = LogisticRegression
    if model_kwargs is None:
        model_kwargs = dict(random_state=random_state, max_iter=1000, class_weight="balanced")
    estimator = model_class(**model_kwargs)

    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    if method == "rfecv":
        selector = RFECV(
            estimator=estimator,
            step=1,
            cv=cv_splitter,
            scoring=scoring,
            min_features_to_select=min_features,
        )
        selector.fit(X_scaled, y)

        selected_mask = selector.support_
        selected_features = [f for f, s in zip(feature_set.features, selected_mask) if s]
        rankings = selector.ranking_

        # Collect CV scores per number of features
        cv_scores = {}
        n_features_range = range(min_features, len(feature_set.features) + 1)
        for i, n in enumerate(n_features_range):
            if i < len(selector.cv_results_["mean_test_score"]):
                cv_scores[n] = selector.cv_results_["mean_test_score"][i]

        ranking_df = (
            pd.DataFrame(
                {"feature": feature_set.features, "rank": rankings}
            )
            .sort_values("rank")
            .reset_index(drop=True)
        )

    elif method == "sequential":
        # Forward selection — try all sizes, pick the one maximizing scoring
        best_score = -1.0
        best_features = None
        all_cv_scores = {}

        for n in range(min_features, len(feature_set.features) + 1):
            sfs = SequentialFeatureSelector(
                estimator=estimator,
                n_features_to_select=n,
                direction="forward",
                scoring=scoring,
                cv=cv_splitter,
            )
            sfs.fit(X_scaled, y)
            mask = sfs.get_support()
            feats = [f for f, s in zip(feature_set.features, mask) if s]

            # Cross-validate the selected subset to get the score
            from sklearn.model_selection import cross_val_score

            scores = cross_val_score(
                estimator, X_scaled[feats], y, cv=cv_splitter, scoring=scoring
            )
            mean_score = scores.mean()
            all_cv_scores[n] = mean_score

            if mean_score > best_score:
                best_score = mean_score
                best_features = feats

        selected_features = best_features
        cv_scores = all_cv_scores

        # Build ranking: selected = 1, rest = 2
        ranking_df = pd.DataFrame(
            {
                "feature": feature_set.features,
                "rank": [1 if f in selected_features else 2 for f in feature_set.features],
            }
        ).sort_values("rank").reset_index(drop=True)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'rfecv' or 'sequential'.")

    # Build a new FeatureSet with the selected features
    new_feature_set = FeatureSet(
        name=f"{feature_set.name} (selected {len(selected_features)})",
        features=selected_features,
        engineer_fn=feature_set.engineer_fn,
        sex_filter=feature_set.sex_filter,
        outcome=feature_set.outcome,
        description=f"Feature selection ({method}, scoring={scoring}) from {feature_set.name}",
    )

    return FeatureSelectionResult(
        selected_features=selected_features,
        feature_set=new_feature_set,
        cv_scores=cv_scores,
        ranking=ranking_df,
        method=method,
    )
