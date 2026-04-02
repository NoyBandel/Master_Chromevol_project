from pathlib import Path
from typing import Dict, List, Tuple

from source_code.constants import *

# -------- baseline models --------
# constant (M0_all_const) vs linear (M1) vs ignore (M1)

BASELINE_MODELS_LABEL: str = "baseline_models"

BASELINE_FUNC_LABELS_ORDERED: Tuple[str, ...] = (LABEL_CONSTANT, LABEL_LINEAR, LABEL_IGNORE)
BASELINE_FUNC_LABELS_NO_IGNORE: Tuple[str, ...] = (LABEL_CONSTANT, LABEL_LINEAR)

NUM_OF_EVENTS_COL: str = "tested_transition_num_of_events"
LIN_SLOPE_P2_COL: str = "lin_slope_p2"
LIN_P1_COL: str = "lin_p1"
CONST_VAL_COL: str = "constant_value"
CHOSEN_FUNCTION_LABEL_COL: str = "chosen_function_label"
CHOSEN_CONFIG_COL: str = "chosen_configuration"
PARAMS_COL: str = "parameters"
CHOSEN_MODEL_PARAMS_COL: str = "chosen_model_parameters"

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
    BASE_CHROM_NUM_COL,
    ROOT_CHROM_NUM_COL,
    NUM_OF_EVENTS_COL,
    CHOSEN_MODEL_PARAMS_COL,
    CONST_VAL_COL,
    LIN_SLOPE_P2_COL,
    LIN_P1_COL,
    FAMILY_SIZE_COL,
    MIN_CHROM_COL,
    MAX_CHROM_COL,
    DIFF_COL,
]

TRANSITION_TO_EVENTS_COL: Dict[str, str] = {
    LABEL_GAIN: EXP_GAIN_COL,
    LABEL_LOSS: EXP_LOSS_COL,
    LABEL_DUPL: EXP_DUPL_COL,
    LABEL_DEMI: EXP_DEMI_COL,
    LABEL_BASE_NUM: EXP_BASE_NUM_COL,
}

BASELINE_ANALYSIS_DIR: Path = ANALYSIS_DIR / BASELINE_MODELS_LABEL

# -------- derived analysis columns --------
CONST_EVENTS_COL: str = "constant_expected_events"
LINEAR_EVENTS_COL: str = "linear_expected_events"
EVENTS_DIFF_COL: str = "linear_minus_constant_events"
EVENTS_REL_DIFF_COL: str = "relative_linear_minus_constant_events"

POSITIVE_SLOPE_LABEL: str = "positive"
NEGATIVE_SLOPE_LABEL: str = "negative"
ZERO_SLOPE_LABEL: str = "zero"

# -------- shared plotting colors --------
MODEL_COLOR_MAP: Dict[str, str] = {
    LABEL_CONSTANT: "#4C78A8",
    LABEL_LINEAR: "#F58518",
    LABEL_IGNORE: "#54A24B",
}

SLOPE_SIGN_COLOR_MAP: Dict[str, str] = {
    NEGATIVE_SLOPE_LABEL: "#E45756",
    ZERO_SLOPE_LABEL: "#B8B8B8",
    POSITIVE_SLOPE_LABEL: "#72B7B2",
}