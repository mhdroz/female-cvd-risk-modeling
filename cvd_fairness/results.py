"""
Metrics computation, experiment comparison, and fairness visualizations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .modeling import TrainedExperiment

# ---------------------------------------------------------------------------
# Color palette (clinical / publication style)
# ---------------------------------------------------------------------------

COLORS = {
    "male": "#3498db",
    "female": "#e74c3c",
    "baseline": "#95a5a6",
    "naive": "#e67e22",
    "constrained": "#27ae60",
    "danger": "#c0392b",
    "warning": "#f39c12",
    "success": "#2ecc71",
}

# Rotating palette for arbitrary number of experiments
EXPERIMENT_PALETTE = [
    "#95a5a6",
    "#e67e22",
    "#27ae60",
    "#3498db",
    "#9b59b6",
    "#1abc9c",
    "#e74c3c",
    "#f1c40f",
]


def _setup_style():
    sns.set_style("white")
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica"]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GroupMetrics:
    """Metrics for a single subgroup (Male, Female, or Overall)."""

    accuracy: float
    precision: float
    recall: float
    auc: Optional[float]
    fpr: Optional[float]
    n_samples: int


@dataclass
class ExperimentResult:
    """Complete results for one experiment."""

    name: str
    description: str
    overall: GroupMetrics
    by_sex: dict[str, GroupMetrics]  # {'Male': ..., 'Female': ...} or empty
    feature_importances: pd.DataFrame  # columns: feature, coefficient
    thresholds: Optional[dict]

    @property
    def recall_gap(self) -> Optional[float]:
        if "Male" in self.by_sex and "Female" in self.by_sex:
            return abs(self.by_sex["Male"].recall - self.by_sex["Female"].recall)
        return None

    @property
    def accuracy_gap(self) -> Optional[float]:
        if "Male" in self.by_sex and "Female" in self.by_sex:
            return abs(self.by_sex["Male"].accuracy - self.by_sex["Female"].accuracy)
        return None

    @property
    def fpr_gap(self) -> Optional[float]:
        if "Male" in self.by_sex and "Female" in self.by_sex:
            m = self.by_sex["Male"].fpr
            f = self.by_sex["Female"].fpr
            if m is not None and f is not None:
                return abs(m - f)
        return None


# ---------------------------------------------------------------------------
# Metric computation helpers
# ---------------------------------------------------------------------------


def _compute_fpr(y_true, y_pred) -> Optional[float]:
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        return fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return None


def _compute_group_metrics(y_true, y_pred, y_prob=None) -> GroupMetrics:
    auc = None
    if y_prob is not None and len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_prob)

    return GroupMetrics(
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        auc=auc,
        fpr=_compute_fpr(y_true, y_pred),
        n_samples=len(y_true),
    )


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------


def evaluate_experiment(experiment: TrainedExperiment) -> ExperimentResult:
    """Compute all metrics from a TrainedExperiment."""
    y_test = experiment.y_test
    y_pred = experiment.y_pred
    y_prob = experiment.y_prob

    overall = _compute_group_metrics(y_test, y_pred, y_prob)

    # Sex-stratified metrics (only when we have sex_test)
    by_sex: dict[str, GroupMetrics] = {}
    if experiment.sex_test is not None:
        for sex_val, sex_name in [(1, "Male"), (0, "Female")]:
            mask = experiment.sex_test == sex_val
            if mask.sum() > 0:
                by_sex[sex_name] = _compute_group_metrics(
                    y_test[mask], y_pred[mask], y_prob[mask]
                )

    # Feature importances (for linear models with coef_)
    if hasattr(experiment.model, "coef_"):
        feature_importances = (
            pd.DataFrame(
                {
                    "feature": experiment.feature_set.features,
                    "coefficient": experiment.model.coef_[0],
                }
            )
            .sort_values("coefficient", key=abs, ascending=False)
            .reset_index(drop=True)
        )
    else:
        feature_importances = pd.DataFrame(columns=["feature", "coefficient"])

    return ExperimentResult(
        name=experiment.name,
        description=experiment.feature_set.description,
        overall=overall,
        by_sex=by_sex,
        feature_importances=feature_importances,
        thresholds=experiment.thresholds,
    )


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------


def compare_experiments(results: list[ExperimentResult]) -> pd.DataFrame:
    """Build a comparison DataFrame across experiments."""
    rows = []
    for r in results:
        row = {
            "name": r.name,
            "overall_accuracy": r.overall.accuracy,
            "overall_precision": r.overall.precision,
            "overall_recall": r.overall.recall,
            "overall_auc": r.overall.auc,
        }
        if "Male" in r.by_sex:
            row["male_accuracy"] = r.by_sex["Male"].accuracy
            row["male_recall"] = r.by_sex["Male"].recall
            row["male_fpr"] = r.by_sex["Male"].fpr
        if "Female" in r.by_sex:
            row["female_accuracy"] = r.by_sex["Female"].accuracy
            row["female_recall"] = r.by_sex["Female"].recall
            row["female_fpr"] = r.by_sex["Female"].fpr
        row["accuracy_gap"] = r.accuracy_gap
        row["recall_gap"] = r.recall_gap
        row["fpr_gap"] = r.fpr_gap
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Visualization functions
# ---------------------------------------------------------------------------


def plot_recall_gap(
    results: list[ExperimentResult], save_path: Optional[str] = None
):
    """Bar chart showing recall gap across experiments."""
    _setup_style()

    results_with_gap = [r for r in results if r.recall_gap is not None]
    if not results_with_gap:
        print("No experiments with sex-stratified results to plot.")
        return

    fig, ax = plt.subplots(figsize=(max(8, len(results_with_gap) * 2.5), 6))

    names = [r.name for r in results_with_gap]
    gaps = [r.recall_gap * 100 for r in results_with_gap]
    colors = [EXPERIMENT_PALETTE[i % len(EXPERIMENT_PALETTE)] for i in range(len(gaps))]
    x = np.arange(len(names))

    bars = ax.bar(x, gaps, color=colors, alpha=0.9, edgecolor="black", linewidth=2, width=0.6)

    for bar, gap in zip(bars, gaps):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{gap:.1f}pp",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_ylabel("Recall Gap (percentage points)", fontsize=13, fontweight="bold")
    ax.set_title("Recall Gap by Sex Across Experiments", fontsize=15, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11, rotation=15, ha="right")
    ax.set_ylim(0, max(gaps) * 1.35)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_fpr_gap(
    results: list[ExperimentResult], save_path: Optional[str] = None
):
    """Bar chart showing false positive rate gap across experiments."""
    _setup_style()

    results_with_gap = [r for r in results if r.fpr_gap is not None]
    if not results_with_gap:
        print("No experiments with FPR gap data to plot.")
        return

    fig, ax = plt.subplots(figsize=(max(8, len(results_with_gap) * 2.5), 6))

    names = [r.name for r in results_with_gap]
    gaps = [r.fpr_gap * 100 for r in results_with_gap]
    colors = [EXPERIMENT_PALETTE[i % len(EXPERIMENT_PALETTE)] for i in range(len(gaps))]
    x = np.arange(len(names))

    bars = ax.bar(x, gaps, color=colors, alpha=0.9, edgecolor="black", linewidth=2, width=0.6)

    for bar, gap in zip(bars, gaps):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{gap:.1f}pp",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_ylabel("FPR Gap (percentage points)", fontsize=13, fontweight="bold")
    ax.set_title("False Positive Rate Gap Across Experiments", fontsize=15, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11, rotation=15, ha="right")
    ax.set_ylim(0, max(gaps) * 1.35)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_fairness_dashboard(
    results: list[ExperimentResult], save_path: Optional[str] = None
):
    """
    Comprehensive fairness dashboard: accuracy & recall by sex for each experiment,
    plus gap comparison.
    """
    _setup_style()

    results_with_sex = [r for r in results if r.by_sex]
    if not results_with_sex:
        print("No experiments with sex-stratified results to plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # --- Top-left: Accuracy by sex ---
    ax = axes[0, 0]
    _grouped_bar_by_sex(ax, results_with_sex, "accuracy", "Accuracy (%)")

    # --- Top-right: Recall by sex ---
    ax = axes[0, 1]
    _grouped_bar_by_sex(ax, results_with_sex, "recall", "Recall / Sensitivity (%)")

    # --- Bottom-left: Recall gap ---
    ax = axes[1, 0]
    names = [r.name for r in results_with_sex]
    gaps = [r.recall_gap * 100 for r in results_with_sex]
    colors = [EXPERIMENT_PALETTE[i % len(EXPERIMENT_PALETTE)] for i in range(len(gaps))]
    bars = ax.bar(range(len(names)), gaps, color=colors, alpha=0.9, edgecolor="black", linewidth=1.5, width=0.6)
    for bar, gap in zip(bars, gaps):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{gap:.1f}pp", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Gap (pp)", fontsize=11, fontweight="bold")
    ax.set_title("Recall Gap (Male - Female)", fontsize=12, fontweight="bold")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9, rotation=20, ha="right")
    ax.set_ylim(0, max(gaps) * 1.4)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Bottom-right: FPR gap ---
    ax = axes[1, 1]
    fpr_results = [r for r in results_with_sex if r.fpr_gap is not None]
    if fpr_results:
        names_fpr = [r.name for r in fpr_results]
        fpr_gaps = [r.fpr_gap * 100 for r in fpr_results]
        colors_fpr = [EXPERIMENT_PALETTE[i % len(EXPERIMENT_PALETTE)] for i in range(len(fpr_gaps))]
        bars = ax.bar(range(len(names_fpr)), fpr_gaps, color=colors_fpr, alpha=0.9, edgecolor="black", linewidth=1.5, width=0.6)
        for bar, gap in zip(bars, fpr_gaps):
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{gap:.1f}pp", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_xticks(range(len(names_fpr)))
        ax.set_xticklabels(names_fpr, fontsize=9, rotation=20, ha="right")
        ax.set_ylim(0, max(fpr_gaps) * 1.4)
    ax.set_ylabel("Gap (pp)", fontsize=11, fontweight="bold")
    ax.set_title("FPR Gap (Male - Female)", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.suptitle("Fairness Dashboard", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()
    plt.close()


def _grouped_bar_by_sex(ax, results: list[ExperimentResult], metric: str, ylabel: str):
    """Helper: grouped bar chart with Male/Female bars for each experiment."""
    n = len(results)
    x = np.arange(n)
    width = 0.35

    male_vals = [getattr(r.by_sex["Male"], metric) * 100 for r in results]
    female_vals = [getattr(r.by_sex["Female"], metric) * 100 for r in results]

    bars_m = ax.bar(x - width / 2, male_vals, width, label="Male", color=COLORS["male"], alpha=0.8, edgecolor="black", linewidth=1.5)
    bars_f = ax.bar(x + width / 2, female_vals, width, label="Female", color=COLORS["female"], alpha=0.8, edgecolor="black", linewidth=1.5)

    for bars in [bars_m, bars_f]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, h, f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
    ax.set_title(f"{ylabel.split('(')[0].strip()} by Sex", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([r.name for r in results], fontsize=9, rotation=15, ha="right")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10, frameon=True)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_feature_importance(
    results: list[ExperimentResult],
    top_n: int = 10,
    save_path: Optional[str] = None,
):
    """Compare top feature importances across experiments."""
    _setup_style()

    n = len(results)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), squeeze=False)

    for i, r in enumerate(results):
        ax = axes[0, i]
        top = r.feature_importances.head(top_n)
        if top.empty:
            ax.set_title(r.name)
            continue

        colors = ["#e74c3c" if c > 0 else "#3498db" for c in top["coefficient"]]
        ax.barh(range(len(top)), top["coefficient"], color=colors, edgecolor="black", linewidth=0.5)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top["feature"], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Coefficient", fontsize=10)
        ax.set_title(r.name, fontsize=11, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle(f"Top {top_n} Feature Importances", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()
    plt.close()
