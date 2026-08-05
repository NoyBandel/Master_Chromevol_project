from pathlib import Path
from typing import Dict, List, Tuple, Set

from source_code.constants import *

# =========================================================
# BASELINE MODELS
# =========================================================

BASELINE_MODELS_LABEL: str = "baseline_models"

BASELINE_FUNC_LABELS_ORDERED: Tuple[str, ...] = (
    LABEL_CONSTANT,
    LABEL_LINEAR,
    LABEL_IGNORE,
)
BASELINE_FUNC_LABELS_NO_IGNORE: Set = {LABEL_CONSTANT, LABEL_LINEAR}

# ---- core column names ----
NUM_OF_EVENTS_COL: str = "tested_transition_num_of_events"

LIN_SLOPE_P2_COL: str = "lin_slope_p2"
LIN_P1_COL: str = "lin_p1"
CONST_VAL_COL: str = "constant_value"

CHOSEN_FUNCTION_LABEL_COL: str = "chosen_function_label"
CHOSEN_CONFIG_COL: str = "chosen_configuration"

PARAMS_COL: str = "parameters"
CHOSEN_MODEL_PARAMS_COL: str = "chosen_model_parameters"

SECOND_BEST_AICC_COL: str = "second_best_aicc"
DELTA_BEST_VS_SECOND_COL: str = "delta_best_vs_second"
BEST_AKAIKE_WEIGHT_COL: str = "best_akaike_weight"

DELTA_SUPPORT_COL: str = "delta_support"
DELTA_SUPPORT_CLASS_COL: str = "delta_support_class"

WEIGHT_SUPPORT_COL: str = "weight_support"
WEIGHT_SUPPORT_CLASS_COL: str = "weight_support_class"

FUNCTION_LABEL_COL: str = "function_label"
CHOSEN_MODEL_COL: str = "chosen_model"

AKAIKE_WEIGHT_COL: str = "akaike_weight"
DELTA_AICC_COL: str = "delta_aicc"


# =========================================================
# SUPPORT LABELS & CLASSES
# =========================================================

# ---- delta AICc labels ----
DELTA_SUPPORT_MISSING_LABEL: str = "missing"
DELTA_SUPPORT_LE_2_LABEL: str = "<=2"
DELTA_SUPPORT_2_TO_4_LABEL: str = "2-4"
DELTA_SUPPORT_4_TO_10_LABEL: str = "4-10"
DELTA_SUPPORT_GT_10_LABEL: str = ">10"

# ---- delta classes (0–4) ----
DELTA_SUPPORT_CLASS_MISSING: int = 0
DELTA_SUPPORT_CLASS_LE_2: int = 1
DELTA_SUPPORT_CLASS_2_TO_4: int = 2
DELTA_SUPPORT_CLASS_4_TO_10: int = 3
DELTA_SUPPORT_CLASS_GT_10: int = 4


# ---- Akaike weight labels ----
WEIGHT_SUPPORT_MISSING_LABEL: str = "missing"
WEIGHT_SUPPORT_LT_06_LABEL: str = "<0.6"
WEIGHT_SUPPORT_06_TO_08_LABEL: str = "0.6-0.8"
WEIGHT_SUPPORT_08_TO_095_LABEL: str = "0.8-0.95"
WEIGHT_SUPPORT_GT_095_LABEL: str = ">0.95"

# ---- weight classes (0–4) ----
WEIGHT_SUPPORT_CLASS_MISSING: int = 0
WEIGHT_SUPPORT_CLASS_LT_06: int = 1
WEIGHT_SUPPORT_CLASS_06_TO_08: int = 2
WEIGHT_SUPPORT_CLASS_08_TO_095: int = 3
WEIGHT_SUPPORT_CLASS_GT_095: int = 4


# =========================================================
# TABLE SCHEMAS
# =========================================================

BASELINE_SUMMARY_TABLE_COLS: List[str] = [
    FAMILY_NAME_COL,
    CONFIG_COL,
    LABEL_TESTED_TRANSITION_COL,
    LABEL_FUNC_TYPE_COL,
    LIKELIHOOD_COL,
    AICC_COL,
    BASE_CHROM_NUM_COL,
    ROOT_CHROM_NUM_COL,
    NUM_OF_EVENTS_COL,
    PARAMS_COL,
]

BASELINE_CHOSEN_MODEL_TABLE_COLS: List[str] = [
    FAMILY_NAME_COL,
    LABEL_TESTED_TRANSITION_COL,
    CHOSEN_FUNCTION_LABEL_COL,
    CHOSEN_CONFIG_COL,
    LIKELIHOOD_COL,
    AICC_COL,
    SECOND_BEST_AICC_COL,
    DELTA_BEST_VS_SECOND_COL,
    BEST_AKAIKE_WEIGHT_COL,
    DELTA_SUPPORT_COL,
    DELTA_SUPPORT_CLASS_COL,
    WEIGHT_SUPPORT_COL,
    WEIGHT_SUPPORT_CLASS_COL,
    CHOSEN_MODEL_PARAMS_COL,
    CONST_VAL_COL,
    LIN_SLOPE_P2_COL,
    LIN_P1_COL,
]

FEATURES_SUMMARY_TABLE_COLS: List[str] = [
    FAMILY_NAME_COL,
    LABEL_TESTED_TRANSITION_COL,
    FUNCTION_LABEL_COL,
    BASE_CHROM_NUM_COL,
    ROOT_CHROM_NUM_COL,
    NUM_OF_EVENTS_COL,
    FAMILY_SIZE_COL,
    MIN_CHROM_COL,
    MAX_CHROM_COL,
    DIFF_COL,
    STD_CHROM_COL,
    PARAMS_COL,
    CHOSEN_MODEL_COL,
]


# =========================================================
# FILE NAMING
# =========================================================

MODELS_SUMMARY_SUFFIX: str = "models_summary_table.csv"
CHOSEN_MODEL_SUFFIX: str = "chosen_model_table.csv"
FEATURES_SUMMARY_SUFFIX: str = "features_summary.csv"


# =========================================================
# PATHS
# =========================================================

BASELINE_ANALYSIS_DIR: Path = ANALYSIS_DIR / BASELINE_MODELS_LABEL


# =========================================================
# STATISTICAL POWER (USED LATER)
# =========================================================

FEATURE_EVENTS: str = NUM_OF_EVENTS_COL
FEATURE_ROOT_NUM: str = ROOT_CHROM_NUM_COL
FEATURE_FAMILY_SIZE: str = FAMILY_SIZE_COL
FEATURE_MIN_CHROM: str = MIN_CHROM_COL
FEATURE_MAX_CHROM: str = MAX_CHROM_COL
FEATURE_CHROM_RANGE: str = DIFF_COL

SUPPORTED_POWER_FEATURES: Set[str] = {
    FEATURE_EVENTS,
    FEATURE_ROOT_NUM,
    FEATURE_FAMILY_SIZE,
    FEATURE_MIN_CHROM,
    FEATURE_MAX_CHROM,
    FEATURE_CHROM_RANGE,
}

RUN_DEPENDENT_FEATURES: Set[str] = {
    FEATURE_EVENTS,
    FEATURE_ROOT_NUM,
}

METADATA_FEATURE_COLS: Set[str] = {
    FEATURE_FAMILY_SIZE,
    FEATURE_MIN_CHROM,
    FEATURE_MAX_CHROM,
    FEATURE_CHROM_RANGE,
}


# =========================================================
# OUTPUT CONSTANTS
# =========================================================

STATISTICAL_POWER_DIR_NAME: str = "statistical_power"
MODEL_SELECTION_SUBDIR: str = "model_selection"



# -------- shared plotting colors --------
POSITIVE_SLOPE_LABEL: str = "positive"
NEGATIVE_SLOPE_LABEL: str = "negative"

MODEL_COLOR_MAP: Dict[str, str] = {
    LABEL_CONSTANT: "#4C72B0",   # muted blue
    LABEL_LINEAR: "#DD8452",     # muted orange
    LABEL_EXP: "#55A868",        # muted green
    LABEL_IGNORE: "#C7C7C7",     # light grey
}

SLOPE_SIGN_COLOR_MAP = {
    NEGATIVE_SLOPE_LABEL: "#7A5195",  # purple
    POSITIVE_SLOPE_LABEL: "#2A9D8F",  # teal
}

DELTA_SUPPORT_CLASS_COLOR_MAP: Dict[int, str] = {
    0: "#BDBDBD",
    1: "#D73027",
    2: "#FC8D59",
    3: "#FEE08B",
    4: "#1A9850",
}

WEIGHT_SUPPORT_CLASS_COLOR_MAP: Dict[int, str] = {
    0: "#BDBDBD",
    1: "#D73027",
    2: "#FC8D59",
    3: "#FEE08B",
    4: "#1A9850",
}


FEATURE_ANALYSIS_SUBDIR: str = "feature_analysis"

FEATURE_ANALYSIS_COLS: List[str] = [
    NUM_OF_EVENTS_COL,
    ROOT_CHROM_NUM_COL,
    FAMILY_SIZE_COL,
    MIN_CHROM_COL,
    MAX_CHROM_COL,
    DIFF_COL,
    STD_CHROM_COL,
]

# =========================================================
# FEATURE ANALYSIS
# =========================================================

FEATURE_ANALYSIS_SUBDIR: str = "feature_analysis"

FEATURE_ANALYSIS_INFERRED_COLS: List[str] = [
    NUM_OF_EVENTS_COL,
    ROOT_CHROM_NUM_COL,
    BASE_CHROM_NUM_COL,
]

FEATURE_ANALYSIS_METADATA_COLS: List[str] = [
    FAMILY_SIZE_COL,
    MIN_CHROM_COL,
    MAX_CHROM_COL,
    DIFF_COL,
    STD_CHROM_COL,
]

FEATURE_ANALYSIS_COLS: List[str] = (
    FEATURE_ANALYSIS_INFERRED_COLS
    + FEATURE_ANALYSIS_METADATA_COLS
)

# ---- direct H0-vs-H1 model comparison ----

H0_AICC_COL: str = "h0_aicc"
H1_AICC_COL: str = "h1_aicc"

SIGNED_DELTA_AICC_H0_MINUS_H1_COL: str = (
    "signed_delta_aicc_h0_minus_h1"
)
PAIRWISE_DELTA_AICC_COL: str = "pairwise_delta_aicc"

H0_PAIRWISE_AKAIKE_WEIGHT_COL: str = "h0_pairwise_akaike_weight"
H1_PAIRWISE_AKAIKE_WEIGHT_COL: str = "h1_pairwise_akaike_weight"
PAIRWISE_BEST_AKAIKE_WEIGHT_COL: str = "pairwise_best_akaike_weight"

# ---- threshold scans ----

THRESHOLD_AT_OR_ABOVE: str = "at_or_above"
THRESHOLD_AT_OR_BELOW: str = "at_or_below"

DEFAULT_MIN_THRESHOLD_FAMILIES: int = 5
STRONG_DELTA_AICC_THRESHOLD: float = 4.0

# <= threshold saturation
SATURATION_TOLERANCE_PP: float = 5.0
SATURATION_MIN_POINTS: int = 3

# >= threshold empirical candidate-filter criteria
CANDIDATE_MIN_H1_GAIN_PP: float = 10.0
CANDIDATE_MIN_STRONG_H1_GAIN_PP: float = 5.0
CANDIDATE_MAX_FUTURE_GAIN_PP: float = 5.0
CANDIDATE_MIN_RETAINED_PCT: float = 25.0