import argparse
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from source_code.analysis.analysis_constants import *
from source_code.logger import log_run
from exp_lin_analysis_constants import *

def build_model_specific_summary_df(parsed_results_file: Path, extra_cols: List[str] | None = None) -> pd.DataFrame:
    parsed_df: pd.DataFrame = pd.read_csv(parsed_results_file)

    required_cols: List[str] = [FAMILY_NAME_COL, CONFIG_COL, LABEL_FUNC_TYPE_COL, LIKELIHOOD_COL, AICC_COL, ROOT_CHROM_NUM_COL, PARAM_0_COL]
    if extra_cols is not None:
        required_cols.extend(extra_cols)
    missing_cols: List[str] = [
        col for col in required_cols
        if col not in parsed_df.columns
    ]
    if missing_cols:
        raise ValueError(f"Missing required columns in {parsed_results_file}: {missing_cols}")

    summary_df: pd.DataFrame = parsed_df[required_cols].copy()
    summary_df[LIKELIHOOD_COL] = pd.to_numeric(summary_df[LIKELIHOOD_COL], errors="coerce")
    summary_df[AICC_COL] = pd.to_numeric(summary_df[AICC_COL], errors="coerce")

    return summary_df

def build_raw_summary_df_from_input_files( m0_parsed_results_file: Path, linear_parsed_results_file: Path, exponential_parsed_results_file: Path) -> pd.DataFrame:
    m0_df: pd.DataFrame = build_model_specific_summary_df(m0_parsed_results_file)
    linear_df: pd.DataFrame = build_model_specific_summary_df(linear_parsed_results_file, [PARAM_1_COL])
    exponential_df: pd.DataFrame = build_model_specific_summary_df(exponential_parsed_results_file, extra_cols=[PARAM_1_COL])

    summary_df: pd.DataFrame = pd.concat([m0_df, linear_df, exponential_df], ignore_index=True)
    summary_df = summary_df.sort_values(by=[FAMILY_NAME_COL, LABEL_FUNC_TYPE_COL], ascending=[True, True]).reset_index(drop=True)

    return summary_df

# =========================================================
# COMPARISON HELPERS
# =========================================================
def choose_pairwise_winner(left_score: object, right_score: object, left_label: str, right_label: str) -> str:
    left_value = pd.to_numeric(left_score, errors="coerce")
    right_value = pd.to_numeric(right_score, errors="coerce")
    if float(left_value - right_value) < 0:
        return left_label
    return right_label

def get_slope_sign(value: object) -> str:
    numeric_value = pd.to_numeric(value, errors="coerce")
    if numeric_value > 0:
        return POSITIVE_SLOPE_LABEL
    else:
        return NEGATIVE_SLOPE_LABEL

# =========================================================
# CORE TABLE HELPERS
# =========================================================

def build_family_core_comparison_row(family_df: pd.DataFrame) -> Dict[str, object]:
    family_name: str = family_df[FAMILY_NAME_COL].iloc[0]

    m0_row: Optional[pd.Series] = family_df[family_df[LABEL_FUNC_TYPE_COL] == LABEL_CONSTANT].iloc[0]
    linear_row: Optional[pd.Series] = family_df[family_df[LABEL_FUNC_TYPE_COL] == LABEL_LINEAR].iloc[0]
    exp_row: Optional[pd.Series] = family_df[family_df[LABEL_FUNC_TYPE_COL] == LABEL_EXP].iloc[0]

    m0_likelihood: object = m0_row[LIKELIHOOD_COL]
    linear_likelihood: object = linear_row[LIKELIHOOD_COL]
    exp_likelihood: object = exp_row[LIKELIHOOD_COL]

    m0_aicc: object = m0_row[AICC_COL]
    linear_aicc: object = linear_row[AICC_COL]
    exp_aicc: object = exp_row[AICC_COL]

    linear_p1: object = linear_row[PARAM_0_COL]
    linear_p2: object = linear_row[PARAM_1_COL]

    exp_p1: object = exp_row[PARAM_0_COL]
    exp_p2: object = exp_row[PARAM_1_COL]

    linear_sign: str = get_slope_sign(linear_p2)
    exp_sign: str = get_slope_sign(exp_p2)

    family_models_summary_row_dict: Dict[str, object] = {
        FAMILY_NAME_COL: family_name,

        M0_VS_M1_EXP_COL: choose_pairwise_winner(m0_aicc, exp_aicc, LABEL_CONSTANT,LABEL_EXP),
        M0_VS_M1_LINEAR_COL: choose_pairwise_winner(m0_aicc, linear_aicc, LABEL_CONSTANT, LABEL_LINEAR),
        M1_LINEAR_VS_M1_EXP_COL: choose_pairwise_winner(linear_aicc, exp_aicc,LABEL_LINEAR, LABEL_EXP),

        EXP_P2_COL: exp_p2,
        EXP_SIGN_COL: exp_sign,
        LINEAR_P2_COL: linear_p2,
        LINEAR_SIGN_COL: linear_sign,
        LINEAR_EXP_SLOPE_SIGN_AGREEMENT_COL: linear_sign == exp_sign,

        M0_LIKELIHOOD_COL: m0_likelihood,
        M1_EXP_LIKELIHOOD_COL: exp_likelihood,
        M1_LINEAR_LIKELIHOOD_COL: linear_likelihood,

        M0_AICC_COL: m0_aicc,
        M1_EXP_AICC_COL: exp_aicc,
        M1_LINEAR_AICC_COL: linear_aicc,

        EXP_P1_COL: exp_p1,
        LINEAR_P1_COL: linear_p1,
    }

    return family_models_summary_row_dict


def build_core_comparison_table(m0_parsed_results_file: Path, linear_parsed_results_file: Path, exponential_parsed_results_file: Path) -> pd.DataFrame:
    summary_df: pd.DataFrame = build_raw_summary_df_from_input_files(m0_parsed_results_file,  linear_parsed_results_file, exponential_parsed_results_file)
    rows: List[Dict[str, object]] = []

    for _, family_df in summary_df.groupby(FAMILY_NAME_COL, sort=True):
        family_row: Dict[str, object] = build_family_core_comparison_row(family_df)
        rows.append(family_row)

    core_df: pd.DataFrame = pd.DataFrame(rows)

    core_df = core_df.reindex(columns=CORE_COMPARISON_TABLE_COLS)
    core_df = core_df.sort_values(by=FAMILY_NAME_COL, ascending=True).reset_index(drop=True)

    return core_df

# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build  M0-linear-exponential comparison table.")

    parser.add_argument("--transition_label", type=str, required=True, choices=[LABEL_GAIN, LABEL_LOSS, LABEL_DUPL], help="Transition to analyze: gain, loss, or dupl.")
    parser.add_argument("--M0_parsed_results_file", type=Path, required=True, help="Path to parsed_results_M0_all_const.csv.")
    parser.add_argument("--linear_parsed_results_file", type=Path, required=True, help="Path to parsed_results_M1_linear_<transition>.csv.")
    parser.add_argument("--exponential_parsed_results_file", type=Path, required=True, help="Path to parsed_results_M1_exponential_<transition>.csv.")
    return parser.parse_args()


def main() -> None:
    args: argparse.Namespace = parse_args()

    output_dir: Path = ANALYSIS_DIR / EXPONENTIAL_VS_LINEAR_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file: Path = output_dir / f"lin_exp_core_comparison_table_{args.transition_label}.csv"

    core_df: pd.DataFrame = build_core_comparison_table(args.M0_parsed_results_file, args.linear_parsed_results_file, args.exponential_parsed_results_file)
    core_df.to_csv(output_file, index=False)

    log_run(
        step="analysis",
        script=Path(__file__),
        params={
            "analysis_type": EXPONENTIAL_VS_LINEAR_SUBDIR,
            "transition_label": args.transition_label,
            "m0_parsed_results_file": str(args.M0_parsed_results_file),
            "linear_parsed_results_file": str(args.linear_parsed_results_file),
            "exponential_parsed_results_file": str(args.exponential_parsed_results_file),
        },
        outputs=[str(output_file)],
        description="Built one M0 vs M1 linear vs M1 exponential core comparison table.",
        notes=(
            "One CSV was created for the requested transition. "
            "Each row is one family. "
            "Pairwise winners are based on AICc. "
            "Linear and exponential p2 signs are compared as direction labels."
        ),
        log_relative_path=(
            Path(BASELINE_MODELS_LABEL)
            / EXPONENTIAL_VS_LINEAR_SUBDIR
            / f"build_core_comparison_table_{args.transition_label}.log"
        ),
    )

if __name__ == "__main__":
    main()