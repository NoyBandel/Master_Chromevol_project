from source_code.analysis.analysis_constants import *


# =========================================================
# Analysis directories
# =========================================================

EXPONENTIAL_VS_LINEAR_SUBDIR: str = "exponential_vs_linear"

SLOPE_ANALYSIS_SUBDIR: str = "slope_analysis"
TABLES_SUBDIR: str = "tables"
PLOTS_SUBDIR: str = "plots"

# Retained for the later per-family function-curve analysis.
FUNCTION_CURVES_SUBDIR: str = "function_curves"
PER_FAMILY_SUBDIR: str = "per_family"


# =========================================================
# Core EXP-vs-LINEAR comparison columns
# =========================================================

M0_VS_M1_EXP_COL: str = "M0_vs_M1_exp"
M0_VS_M1_LINEAR_COL: str = "M0_vs_M1_linear"
M1_LINEAR_VS_M1_EXP_COL: str = "M1_linear_vs_M1_exp"

M0_LIKELIHOOD_COL: str = "M0_likelihood"
M1_EXP_LIKELIHOOD_COL: str = "M1_exp_likelihood"
M1_LINEAR_LIKELIHOOD_COL: str = "M1_linear_likelihood"

M0_AICC_COL: str = "M0_AICc"
M1_EXP_AICC_COL: str = "M1_exp_AICc"
M1_LINEAR_AICC_COL: str = "M1_linear_AICc"

EXP_P1_COL: str = "exp_p1"
EXP_P2_COL: str = "exp_slope_p2"
EXP_SIGN_COL: str = "exp_sign"

LINEAR_P1_COL: str = LIN_P1_COL
LINEAR_P2_COL: str = LIN_SLOPE_P2_COL
LINEAR_SIGN_COL: str = "lin_sign"

LINEAR_EXP_SLOPE_SIGN_AGREEMENT_COL: str = "linear_exp_slope_sign_agreement"


CORE_COMPARISON_TABLE_COLS: list[str] = [
    FAMILY_NAME_COL,
    M0_VS_M1_EXP_COL,
    M0_VS_M1_LINEAR_COL,
    M1_LINEAR_VS_M1_EXP_COL,
    EXP_P2_COL,
    EXP_SIGN_COL,
    LINEAR_P2_COL,
    LINEAR_SIGN_COL,
    LINEAR_EXP_SLOPE_SIGN_AGREEMENT_COL,
    M0_LIKELIHOOD_COL,
    M1_EXP_LIKELIHOOD_COL,
    M1_LINEAR_LIKELIHOOD_COL,
    M0_AICC_COL,
    M1_EXP_AICC_COL,
    M1_LINEAR_AICC_COL,
    EXP_P1_COL,
    LINEAR_P1_COL,
]


# =========================================================
# Model-selection constants
# =========================================================

MODEL_LABELS_WITH_EXP: tuple[str, ...] = (
    LABEL_CONSTANT,
    LABEL_LINEAR,
    LABEL_EXP,
    LABEL_IGNORE,
)

EXP_ONLY_STATUS: str = "exp_only"
EXP_AND_LINEAR_STATUS: str = "exp_and_linear"
LINEAR_ONLY_STATUS: str = "linear_only"
CONSTANT_BEATS_BOTH_STATUS: str = "constant_beats_both"

DEPENDENCE_DETECTION_STATUS_COL: str = "dependence_detection_status"

CHOSEN_MODEL_WITHOUT_EXP_COL: str = "chosen_model_without_exp"
CHOSEN_MODEL_WITHOUT_EXP_AICC_COL: str = "chosen_model_without_exp_AICc"
CHOSEN_MODEL_WITH_EXP_COL: str = "chosen_model_with_exp"
CHOSEN_MODEL_WITH_EXP_AICC_COL: str = "chosen_model_with_exp_AICc"

OVERALL_CHOSEN_MODEL_COL: str = "overall_chosen_model"


# =========================================================
# Integrated AICc columns
# =========================================================

CONSTANT_AICC_COL: str = "constant_AICc"
LINEAR_AICC_COL: str = "linear_AICc"
EXP_AICC_COL: str = "exponential_AICc"
IGNORE_AICC_COL: str = "ignore_AICc"

AICC_COL_BY_MODEL: dict[str, str] = {
    LABEL_CONSTANT: CONSTANT_AICC_COL,
    LABEL_LINEAR: LINEAR_AICC_COL,
    LABEL_EXP: EXP_AICC_COL,
    LABEL_IGNORE: IGNORE_AICC_COL,
}


# =========================================================
# Slope-analysis columns and labels
# =========================================================

EXP_EFFECTIVE_SLOPE_COL: str = "exp_effective_slope"
LINEAR_EFFECTIVE_SLOPE_COL: str = "linear_effective_slope"

SIGN_AGREEMENT_STATUS_COL: str = "slope_sign_agreement_status"

SIGN_AGREEMENT_LABEL: str = "agreement"
SIGN_DISAGREEMENT_LABEL: str = "disagreement"
SIGN_MISSING_LABEL: str = "missing"
ZERO_SLOPE_LABEL: str = "zero"

SIGN_AGREEMENT_ORDER: tuple[str, ...] = (
    SIGN_AGREEMENT_LABEL,
    SIGN_DISAGREEMENT_LABEL,
    SIGN_MISSING_LABEL,
)

SIGN_AGREEMENT_COLOR_MAP: dict[str, str] = {
    SIGN_AGREEMENT_LABEL: "#2A9D8F",
    SIGN_DISAGREEMENT_LABEL: "#E76F51",
    SIGN_MISSING_LABEL: "#9E9E9E",
}