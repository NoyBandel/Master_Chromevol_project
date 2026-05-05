import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from source_code.analysis.analysis_constants import *
from source_code.analysis.metadata_analysis import load_family_metadata
from source_code.logger import log_run


# -------- loading --------
def load_parsed_results(configuration: str) -> Optional[pd.DataFrame]:
    parsed_file: Path = PARSED_RESULTS_ROOT / f"{PARSED_RESULTS_FILE_PREFIX}_{configuration}.csv"

    if not parsed_file.exists():
        raise FileNotFoundError(f"Parsed results file not found: {parsed_file}")

    parsed_df: pd.DataFrame = pd.read_csv(parsed_file)

    if FAMILY_NAME_COL not in parsed_df.columns:
        raise ValueError(f"{parsed_file} does not contain column '{FAMILY_NAME_COL}'")

    return parsed_df.copy()


def load_metadata_table(families_metadata_file: Path) -> pd.DataFrame:
    metadata_df: pd.DataFrame = load_family_metadata(families_metadata_file)
    required_cols: List[str] = [FAMILY_NAME_COL, FAMILY_SIZE_COL, MIN_CHROM_COL, MAX_CHROM_COL, DIFF_COL, STD_CHROM_COL]
    missing_cols: List[str] = [col for col in required_cols if col not in metadata_df.columns]

    if missing_cols:
        raise ValueError(f"Missing required metadata columns: {missing_cols}")

    return metadata_df[required_cols].copy()


# -------- baseline summary construction --------
def baseline_configs(transition_label: str) -> Dict[str, str]:
    return {
        LABEL_CONSTANT: M0_LABEL,
        LABEL_LINEAR: f"{M1_LABEL}_{LABEL_LINEAR}_{transition_label}",
        LABEL_IGNORE: f"{M1_LABEL}_{LABEL_IGNORE}_{transition_label}",
    }


def parse_transition_parameters(params_json_value: object, transition_label: str) -> List[object]:
    if pd.isna(params_json_value):
        return []

    if isinstance(params_json_value, dict):
        return params_json_value.get(transition_label, [])

    if isinstance(params_json_value, str):
        parsed_json: Dict[str, object] = json.loads(params_json_value)
        return parsed_json.get(transition_label, [])

    return []


def build_single_config_df(parsed_df: pd.DataFrame, transition_label: str) -> pd.DataFrame:
    events_col: str = TRANSITION_TO_EVENTS_COL[transition_label]
    required_cols: List[str] = [FAMILY_NAME_COL, CONFIG_COL, LABEL_TESTED_TRANSITION_COL, LABEL_FUNC_TYPE_COL, LIKELIHOOD_COL, AICC_COL, BASE_CHROM_NUM_COL, ROOT_CHROM_NUM_COL, events_col, PARAMS_JSON_COL]
    missing_cols: List[str] = [col for col in required_cols if col not in parsed_df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns in parsed results: {missing_cols}")

    config_df: pd.DataFrame = parsed_df[required_cols].copy()
    config_df[PARAMS_COL] = config_df[PARAMS_JSON_COL].apply(lambda value: parse_transition_parameters(value, transition_label))
    config_df[NUM_OF_EVENTS_COL] = config_df[events_col]

    is_constant_config: bool = bool((config_df[LABEL_FUNC_TYPE_COL] == LABEL_CONSTANT).all())
    if is_constant_config:
        config_df[LABEL_TESTED_TRANSITION_COL] = transition_label

    config_df = config_df.drop(columns=[events_col, PARAMS_JSON_COL])
    config_df = config_df.reindex(columns=BASELINE_SUMMARY_TABLE_COLS)

    return config_df


def build_baseline_summary_table(transition_label: str) -> pd.DataFrame:
    config_map: Dict[str, str] = baseline_configs(transition_label)
    summary_parts: List[pd.DataFrame] = []

    for _, configuration in config_map.items():
        parsed_df: pd.DataFrame = load_parsed_results(configuration)
        config_df: pd.DataFrame = build_single_config_df(parsed_df, transition_label)
        summary_parts.append(config_df)

    summary_df: pd.DataFrame = pd.concat(summary_parts, ignore_index=True)
    summary_df[LIKELIHOOD_COL] = pd.to_numeric(summary_df[LIKELIHOOD_COL], errors="coerce")
    summary_df[AICC_COL] = pd.to_numeric(summary_df[AICC_COL], errors="coerce")
    summary_df[BASE_CHROM_NUM_COL] = pd.to_numeric(summary_df[BASE_CHROM_NUM_COL], errors="coerce")
    summary_df[ROOT_CHROM_NUM_COL] = pd.to_numeric(summary_df[ROOT_CHROM_NUM_COL], errors="coerce")
    summary_df[NUM_OF_EVENTS_COL] = pd.to_numeric(summary_df[NUM_OF_EVENTS_COL], errors="coerce")
    summary_df = summary_df.sort_values(by=[FAMILY_NAME_COL, LABEL_FUNC_TYPE_COL]).reset_index(drop=True)

    return summary_df


# -------- support helpers --------
def delta_support_label(delta: float) -> str:
    if pd.isna(delta):
        return DELTA_SUPPORT_MISSING_LABEL
    if delta <= 2:
        return DELTA_SUPPORT_LE_2_LABEL
    if delta <= 4:
        return DELTA_SUPPORT_2_TO_4_LABEL
    if delta <= 10:
        return DELTA_SUPPORT_4_TO_10_LABEL
    return DELTA_SUPPORT_GT_10_LABEL


def delta_support_class(delta: float) -> int:
    if pd.isna(delta):
        return DELTA_SUPPORT_CLASS_MISSING
    if delta <= 2:
        return DELTA_SUPPORT_CLASS_LE_2
    if delta <= 4:
        return DELTA_SUPPORT_CLASS_2_TO_4
    if delta <= 10:
        return DELTA_SUPPORT_CLASS_4_TO_10
    return DELTA_SUPPORT_CLASS_GT_10


def weight_support_label(weight: float) -> str:
    if pd.isna(weight):
        return WEIGHT_SUPPORT_MISSING_LABEL
    if weight < 0.6:
        return WEIGHT_SUPPORT_LT_06_LABEL
    if weight < 0.8:
        return WEIGHT_SUPPORT_06_TO_08_LABEL
    if weight < 0.95:
        return WEIGHT_SUPPORT_08_TO_095_LABEL
    return WEIGHT_SUPPORT_GT_095_LABEL


def weight_support_class(weight: float) -> int:
    if pd.isna(weight):
        return WEIGHT_SUPPORT_CLASS_MISSING
    if weight < 0.6:
        return WEIGHT_SUPPORT_CLASS_LT_06
    if weight < 0.8:
        return WEIGHT_SUPPORT_CLASS_06_TO_08
    if weight < 0.95:
        return WEIGHT_SUPPORT_CLASS_08_TO_095
    return WEIGHT_SUPPORT_CLASS_GT_095


def add_akaike_weights(family_df: pd.DataFrame) -> pd.DataFrame:
    weighted_df: pd.DataFrame = family_df.copy()
    weighted_df[AICC_COL] = pd.to_numeric(weighted_df[AICC_COL], errors="coerce")
    weighted_df = weighted_df.dropna(subset=[AICC_COL])

    if weighted_df.empty:
        weighted_df[DELTA_AICC_COL] = np.nan
        weighted_df[AKAIKE_WEIGHT_COL] = np.nan
        return weighted_df

    best_aicc: float = float(weighted_df[AICC_COL].min())
    rel_lik: pd.Series = np.exp(-0.5 * (weighted_df[AICC_COL] - best_aicc))
    denom: float = float(rel_lik.sum())

    weighted_df[DELTA_AICC_COL] = weighted_df[AICC_COL] - best_aicc
    weighted_df[AKAIKE_WEIGHT_COL] = rel_lik / denom if denom > 0 else np.nan

    return weighted_df


# -------- chosen model table --------
def extract_constant_and_linear_params(family_df: pd.DataFrame) -> Tuple[object, object, object]:
    constant_df: pd.DataFrame = family_df[family_df[LABEL_FUNC_TYPE_COL] == LABEL_CONSTANT]
    linear_df: pd.DataFrame = family_df[family_df[LABEL_FUNC_TYPE_COL] == LABEL_LINEAR]

    constant_params: List[object] = constant_df.iloc[0][PARAMS_COL] if not constant_df.empty and isinstance(constant_df.iloc[0][PARAMS_COL], list) else []
    linear_params: List[object] = linear_df.iloc[0][PARAMS_COL] if not linear_df.empty and isinstance(linear_df.iloc[0][PARAMS_COL], list) else []

    const_val: object = constant_params[0] if len(constant_params) >= 1 else pd.NA
    lin_p1: object = linear_params[0] if len(linear_params) >= 1 else pd.NA
    lin_p2: object = linear_params[1] if len(linear_params) >= 2 else pd.NA

    return const_val, lin_p1, lin_p2


def build_chosen_model_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    chosen_rows: List[Dict[str, object]] = []

    for _, family_df in summary_df.groupby(FAMILY_NAME_COL, sort=True):
        weighted_df: pd.DataFrame = add_akaike_weights(family_df)

        if weighted_df.empty:
            continue

        weighted_df = weighted_df.sort_values(by=[AICC_COL, LABEL_FUNC_TYPE_COL]).reset_index(drop=True)
        best_row: pd.Series = weighted_df.iloc[0]
        second_best_aicc: float = float(weighted_df.iloc[1][AICC_COL]) if len(weighted_df) > 1 and pd.notna(weighted_df.iloc[1][AICC_COL]) else np.nan
        delta_best_vs_second: float = second_best_aicc - float(best_row[AICC_COL]) if pd.notna(second_best_aicc) and pd.notna(best_row[AICC_COL]) else np.nan
        best_akaike_weight: float = float(best_row[AKAIKE_WEIGHT_COL]) if pd.notna(best_row[AKAIKE_WEIGHT_COL]) else np.nan
        chosen_params: List[object] = best_row[PARAMS_COL] if isinstance(best_row[PARAMS_COL], list) else []
        const_val: object
        lin_p1: object
        lin_p2: object
        const_val, lin_p1, lin_p2 = extract_constant_and_linear_params(weighted_df)

        chosen_rows.append(
            {
                FAMILY_NAME_COL: best_row[FAMILY_NAME_COL],
                LABEL_TESTED_TRANSITION_COL: best_row[LABEL_TESTED_TRANSITION_COL],
                CHOSEN_FUNCTION_LABEL_COL: best_row[LABEL_FUNC_TYPE_COL],
                CHOSEN_CONFIG_COL: best_row[CONFIG_COL],
                LIKELIHOOD_COL: best_row[LIKELIHOOD_COL],
                AICC_COL: best_row[AICC_COL],
                SECOND_BEST_AICC_COL: second_best_aicc,
                DELTA_BEST_VS_SECOND_COL: delta_best_vs_second,
                BEST_AKAIKE_WEIGHT_COL: best_akaike_weight,
                DELTA_SUPPORT_COL: delta_support_label(delta_best_vs_second),
                DELTA_SUPPORT_CLASS_COL: delta_support_class(delta_best_vs_second),
                WEIGHT_SUPPORT_COL: weight_support_label(best_akaike_weight),
                WEIGHT_SUPPORT_CLASS_COL: weight_support_class(best_akaike_weight),
                CHOSEN_MODEL_PARAMS_COL: chosen_params,
                CONST_VAL_COL: const_val,
                LIN_SLOPE_P2_COL: lin_p2,
                LIN_P1_COL: lin_p1,
            }
        )

    chosen_df: pd.DataFrame = pd.DataFrame(chosen_rows)
    chosen_df = chosen_df.reindex(columns=BASELINE_CHOSEN_MODEL_TABLE_COLS)
    chosen_df = chosen_df.sort_values(by=FAMILY_NAME_COL).reset_index(drop=True)

    return chosen_df


# -------- feature summary tables --------
def enrich_summary_with_metadata(summary_df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    enriched_df: pd.DataFrame = summary_df.merge(metadata_df, on=FAMILY_NAME_COL, how="left")
    enriched_df[FUNCTION_LABEL_COL] = enriched_df[LABEL_FUNC_TYPE_COL]

    return enriched_df


def add_chosen_model_flag(enriched_df: pd.DataFrame, chosen_df: pd.DataFrame) -> pd.DataFrame:
    chosen_keys_df: pd.DataFrame = chosen_df[[FAMILY_NAME_COL, CHOSEN_FUNCTION_LABEL_COL]].copy()
    chosen_keys_df[CHOSEN_MODEL_COL] = 1

    flagged_df: pd.DataFrame = enriched_df.merge(
        chosen_keys_df,
        left_on=[FAMILY_NAME_COL, FUNCTION_LABEL_COL],
        right_on=[FAMILY_NAME_COL, CHOSEN_FUNCTION_LABEL_COL],
        how="left",
    )

    flagged_df[CHOSEN_MODEL_COL] = flagged_df[CHOSEN_MODEL_COL].fillna(0).astype(int)
    flagged_df = flagged_df.drop(columns=[CHOSEN_FUNCTION_LABEL_COL])

    return flagged_df


def build_function_features_table(flagged_df: pd.DataFrame, function_label: str) -> pd.DataFrame:
    function_df: pd.DataFrame = flagged_df[flagged_df[FUNCTION_LABEL_COL] == function_label].copy()
    function_df = function_df.reindex(columns=FEATURES_SUMMARY_TABLE_COLS)
    function_df = function_df.sort_values(by=FAMILY_NAME_COL).reset_index(drop=True)

    return function_df


def build_all_function_features_tables(summary_df: pd.DataFrame, chosen_df: pd.DataFrame, metadata_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    enriched_df: pd.DataFrame = enrich_summary_with_metadata(summary_df, metadata_df)
    flagged_df: pd.DataFrame = add_chosen_model_flag(enriched_df, chosen_df)
    function_tables: Dict[str, pd.DataFrame] = {}

    for function_label in BASELINE_FUNC_LABELS_ORDERED:
        function_tables[function_label] = build_function_features_table(flagged_df, function_label)

    return function_tables


# -------- saving --------
def build_output_dir(transition_label: str) -> Path:
    output_dir: Path = BASELINE_ANALYSIS_DIR / transition_label
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def build_models_summary_file(output_dir: Path, transition_label: str) -> Path:
    return output_dir / f"{transition_label}_{MODELS_SUMMARY_SUFFIX}"


def build_chosen_model_file(output_dir: Path, transition_label: str) -> Path:
    return output_dir / f"{transition_label}_{CHOSEN_MODEL_SUFFIX}"


def build_features_summary_file(output_dir: Path, transition_label: str, function_label: str) -> Path:
    return output_dir / f"{transition_label}_{function_label}_{FEATURES_SUMMARY_SUFFIX}"


def save_outputs(summary_df: pd.DataFrame, chosen_df: pd.DataFrame, function_tables: Dict[str, pd.DataFrame], transition_label: str) -> List[str]:
    output_dir: Path = build_output_dir(transition_label)
    summary_file: Path = build_models_summary_file(output_dir, transition_label)
    chosen_file: Path = build_chosen_model_file(output_dir, transition_label)
    output_paths: List[str] = []

    summary_df.to_csv(summary_file, index=False)
    chosen_df.to_csv(chosen_file, index=False)

    output_paths.extend([str(summary_file), str(chosen_file)])

    for function_label, function_df in function_tables.items():
        function_file: Path = build_features_summary_file(output_dir, transition_label, function_label)
        function_df.to_csv(function_file, index=False)
        output_paths.append(str(function_file))

    print(f"[✓] Saved model summary table: {summary_file}")
    print(f"[✓] Saved chosen model table: {chosen_file}")

    for function_label in BASELINE_FUNC_LABELS_ORDERED:
        print(f"[✓] Saved {function_label} features summary table: {build_features_summary_file(output_dir, transition_label, function_label)}")

    return output_paths


# -------- args / run --------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build baseline summary, chosen-model, and per-function features tables for one transition.")
    parser.add_argument("--transition_label", required=True, choices=list(LABEL_TRANSITIONS_ORDERED), help="Tested transition to analyze.")
    parser.add_argument("--families_metadata_file", type=Path, default=ALL_FAMILIES_DATA_SUMMARY_FILE, help="CSV file with family metadata from preprocessing.")

    return parser.parse_args()


def main() -> None:
    args: argparse.Namespace = parse_args()
    transition_label: str = args.transition_label
    metadata_file: Path = args.families_metadata_file

    if transition_label not in LABEL_TRANSITIONS_ORDERED:
        raise ValueError(f"Unsupported transition: {transition_label}")

    summary_df: pd.DataFrame = build_baseline_summary_table(transition_label)
    chosen_df: pd.DataFrame = build_chosen_model_table(summary_df)
    metadata_df: pd.DataFrame = load_metadata_table(metadata_file)
    function_tables: Dict[str, pd.DataFrame] = build_all_function_features_tables(summary_df, chosen_df, metadata_df)
    output_paths: List[str] = save_outputs(summary_df, chosen_df, function_tables, transition_label)

    log_run(
        step="analysis",
        script=Path(__file__),
        params={"transition_label": transition_label, "families_metadata_file": str(metadata_file)},
        outputs=output_paths,
        description=f"Built baseline summary, chosen-model, and per-function features tables for transition '{transition_label}'",
        notes="Includes baseline model comparison, family-level chosen-model metrics, and function-specific feature summary tables.",
        log_relative_path=Path(BASELINE_MODELS_LABEL) / f"{transition_label}.log",
    )


if __name__ == "__main__":
    main()