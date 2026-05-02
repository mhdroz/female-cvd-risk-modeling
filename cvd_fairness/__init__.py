"""
cvd_fairness: Cardiovascular risk prediction fairness analysis library.
"""

from .data_prep import (
    OUTCOME_COL,
    impute,
    load_framingham,
    prepare_data,
    standardize_columns,
)
from .feature_engineering import (
    BASE_FEATURES_NO_SEX,
    BASE_FEATURES_WITH_SEX,
    FEMALE_BASE_FEATURES,
    FEMALE_ENGINEERED_FULL,
    FEMALE_ENGINEERED_LEAN,
    FeatureSet,
    engineer_female_features,
    get_preset_feature_sets,
)
from .modeling import (
    FeatureSelectionResult,
    TrainedExperiment,
    apply_sex_specific_thresholds,
    find_optimal_threshold,
    run_experiment,
    select_features,
)
from .results import (
    ExperimentResult,
    GroupMetrics,
    compare_experiments,
    evaluate_experiment,
    plot_fairness_dashboard,
    plot_feature_importance,
    plot_fpr_gap,
    plot_recall_gap,
)
