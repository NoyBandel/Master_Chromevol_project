import argparse
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import json
from source_code.analysis.analysis_constants import *
from source_code.analysis.metadata_analysis import load_family_metadata

# baseline models = constant vs linear (vs ignore)

def load_parsed_results(configuration: str) -> Optional[pd.DataFrame]:
    parsed_file = PARSED_RESULTS_ROOT / f"{PARSED_RESULTS_FILE_PREFIX}_{configuration}.csv"
    if not parsed_file.exists():
        raise FileNotFoundError(f"Parsed results file not found: {parsed_file}")

    df = pd.read_csv(parsed_file)
    if FAMILY_NAME_COL not in df.columns:
        raise ValueError(f"{parsed_file} does not contain column '{FAMILY_NAME_COL}'")

    df = df.copy()
    return df

def build_single_config_df(parsed_df: pd.DataFrame, transition_label: str) -> pd.DataFrame:
    events_col_parsed_res_format: str = TRANSITION_TO_EVENTS_COL[transition_label]

    required_cols: List[str] = [
        FAMILY_NAME_COL, CONFIG_COL, LABEL_TESTED_TRANSITION_COL, LABEL_FUNC_TYPE_COL,
        LIKELIHOOD_COL, AICC_COL,
        BASE_CHROM_NUM_COL, ROOT_CHROM_NUM_COL, events_col_parsed_res_format,
        ALL_PARAMS_DICT_STR_COL
    ]

    missing_cols = [col for col in required_cols if col not in parsed_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in parsed_df: {missing_cols}")

    single_config_df = parsed_df[required_cols].copy()

    single_config_df[PARAMS_COL] = single_config_df[ALL_PARAMS_DICT_STR_COL].apply(
        lambda s: json.loads(s).get(transition_label, []) if pd.notna(s) else []
    )

    single_config_df[NUM_OF_EVENTS_COL] = single_config_df[events_col_parsed_res_format]

    is_constant_config = (single_config_df[LABEL_FUNC_TYPE_COL] == LABEL_CONSTANT).all()
    if is_constant_config:
        single_config_df[LABEL_TESTED_TRANSITION_COL] = transition_label

    single_config_df = single_config_df.drop(columns=[events_col_parsed_res_format, ALL_PARAMS_DICT_STR_COL])
    single_config_df = single_config_df.reindex(columns=BASELINE_SUMMARY_TABLE_COLS)

    return single_config_df


def baseline_configs(transition_label: str) -> Dict[str, str]:
    return {LABEL_CONSTANT: M0_LABEL,
            LABEL_LINEAR: f"{M1_LABEL}_{LABEL_LINEAR}_{transition_label}",
            LABEL_IGNORE: f"{M1_LABEL}_{LABEL_IGNORE}_{transition_label}"
            }

def build_baseline_summary_table(transition_label: str) -> pd.DataFrame:
    baseline_configs_dict: Dict[str, str] = baseline_configs(transition_label)
    baseline_dfs_lst: List[pd.DataFrame] = []

    for function_label, configuration in baseline_configs_dict.items():
        parsed_df = load_parsed_results(configuration)
        curr_config_df = build_single_config_df(parsed_df, transition_label)
        baseline_dfs_lst.append(curr_config_df)

    baseline_summary_df = pd.concat(baseline_dfs_lst, ignore_index=True)
    baseline_summary_df = baseline_summary_df.sort_values(by=[FAMILY_NAME_COL, LABEL_FUNC_TYPE_COL]).reset_index(drop=True)
    return baseline_summary_df

def build_chosen_model_table(baseline_models_summary_df: pd.DataFrame, families_metadata_file: Path) -> pd.DataFrame:
    families_metadata_df = load_family_metadata(families_metadata_file)
    chosen_rows: List[Dict[str, object]] = []

    for _, fam_df in baseline_models_summary_df.groupby(FAMILY_NAME_COL, sort=True):
        fam_df = fam_df.copy()
        fam_df[AICC_COL] = pd.to_numeric(fam_df[AICC_COL], errors="coerce")
        fam_df[LIKELIHOOD_COL] = pd.to_numeric(fam_df[LIKELIHOOD_COL], errors="coerce")
        best_row = fam_df.nsmallest(1, AICC_COL).iloc[0]
        chosen_params = best_row[PARAMS_COL] if isinstance(best_row[PARAMS_COL], list) else []

        constant_df = fam_df[fam_df[LABEL_FUNC_TYPE_COL] == LABEL_CONSTANT]
        linear_df = fam_df[fam_df[LABEL_FUNC_TYPE_COL] == LABEL_LINEAR]
        constant_params = constant_df.iloc[0][PARAMS_COL] if not constant_df.empty else []
        linear_params = linear_df.iloc[0][PARAMS_COL] if not linear_df.empty else []
        const_val = constant_params[0] if isinstance(constant_params, list) and len(constant_params) >= 1 else pd.NA
        lin_p1 = linear_params[0] if isinstance(linear_params, list) and len(linear_params) >= 1 else pd.NA
        lin_p2 = linear_params[1] if isinstance(linear_params, list) and len(linear_params) >= 2 else pd.NA

        chosen_rows.append({
            FAMILY_NAME_COL: best_row[FAMILY_NAME_COL],
            LABEL_TESTED_TRANSITION_COL: best_row[LABEL_TESTED_TRANSITION_COL],
            CHOSEN_FUNCTION_LABEL_COL: best_row[LABEL_FUNC_TYPE_COL],
            CHOSEN_CONFIG_COL: best_row[CONFIG_COL],
            LIKELIHOOD_COL: best_row[LIKELIHOOD_COL],
            AICC_COL: best_row[AICC_COL],
            BASE_CHROM_NUM_COL: best_row[BASE_CHROM_NUM_COL],
            ROOT_CHROM_NUM_COL: best_row[ROOT_CHROM_NUM_COL],
            NUM_OF_EVENTS_COL: best_row[NUM_OF_EVENTS_COL],
            CHOSEN_MODEL_PARAMS_COL: chosen_params,
            CONST_VAL_COL: const_val,
            LIN_SLOPE_P2_COL: lin_p2,
            LIN_P1_COL: lin_p1,
        })

    chosen_df = pd.DataFrame(chosen_rows)
    chosen_df = chosen_df.merge(families_metadata_df[[FAMILY_NAME_COL, FAMILY_SIZE_COL, MIN_CHROM_COL, MAX_CHROM_COL, DIFF_COL]],
                                on=FAMILY_NAME_COL,
                                how="left"
                                )
    chosen_df = chosen_df.reindex(columns=BASELINE_CHOSEN_MODEL_TABLE_COLS)
    chosen_df = chosen_df.sort_values(by=FAMILY_NAME_COL).reset_index(drop=True)

    return chosen_df

def save_outputs(models_summary_df: pd.DataFrame, chosen_model_df: pd.DataFrame, tested_transition_label: str) -> None:
    output_dir: Path = ANALYSIS_DIR / BASELINE_MODELS_LABEL / tested_transition_label
    output_dir.mkdir(parents=True, exist_ok=True)

    models_summary_file = output_dir / f"{tested_transition_label}_models_summary_table.csv"
    chosen_model_file = output_dir / f"{tested_transition_label}_chosen_model_table.csv"

    models_summary_df.to_csv(models_summary_file, index=False)
    chosen_model_df.to_csv(chosen_model_file, index=False)

    print(f"[✓] Saved model summary table: {models_summary_df}")
    print(f"[✓] Saved chosen model table: {chosen_model_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build baseline models-summary and chosen-model tables for one transition")
    parser.add_argument("--transition_label", required=True, choices=list(LABEL_TRANSITIONS_ORDERED), help="Tested transition to analyze.")
    parser.add_argument("--families_metadata_file", type=Path, default= ALL_FAMILIES_DATA_SUMMARY_FILE, help="CSV file with family metadata from preprocessing.")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    tested_transition_label = args.transition_label
    if tested_transition_label not in LABEL_TRANSITIONS_ORDERED:
        raise ValueError(f"Unsupported transition: {tested_transition_label}")
    baseline_models_summary_df: pd.DataFrame = build_baseline_summary_table(tested_transition_label)
    chosen_model_df: pd.DataFrame = build_chosen_model_table(baseline_models_summary_df, args.families_metadata_file)
    save_outputs(baseline_models_summary_df, chosen_model_df, tested_transition_label)


    #---- add logging!


if __name__ == "__main__":
    main()