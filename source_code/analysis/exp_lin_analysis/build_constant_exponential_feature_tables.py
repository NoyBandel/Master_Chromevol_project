#!/usr/bin/env python3
"""
Build feature-analysis input tables for constant (H0) vs exponential (H1).

This temporary builder reuses:
- constant and ignore rows from the existing baseline model-summary table;
- the parsed exponential ChromEvol results;
- the family metadata table.

It performs model selection among constant, exponential, and ignore, then
creates the same per-function feature tables used by feature_analysis.py.

Prerequisites for a transition such as ``dupl``:
1. analysis/baseline_models/dupl/dupl_models_summary_table.csv
2. chromevol_parsed_results/parsed_results_M1_exponential_dupl.csv
3. input_data/all_families_data_summary.csv

The script does not run ChromEvol and does not parse raw ChromEvol output.
Only families with valid AICc values for all three candidate models are kept.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from source_code import constants as core
from source_code.analysis import analysis_constants as ac
from source_code.analysis.baseline_models.build_transition_summary_tables import (
    add_akaike_weights,
    add_chosen_model_flag,
    build_function_features_table,
    build_single_config_df,
    delta_support_class,
    delta_support_label,
    enrich_summary_with_metadata,
    load_metadata_table,
    weight_support_class,
    weight_support_label,
)
from source_code.logger import log_run


H0_LABEL = core.LABEL_CONSTANT
H1_LABEL = core.LABEL_EXP
IGNORE_LABEL = core.LABEL_IGNORE
CANDIDATE_MODELS = (H0_LABEL, H1_LABEL, IGNORE_LABEL)
COMPARISON_NAME = f"{H0_LABEL}_vs_{H1_LABEL}"

EXP_P1_COL = "exp_p1"
EXP_P2_COL = "exp_p2"

H0_AICC_COL = "h0_aicc"
H1_AICC_COL = "h1_aicc"
SIGNED_PAIRWISE_DELTA_COL = "signed_delta_aicc_h0_minus_h1"
PAIRWISE_CHOSEN_COL = "pairwise_chosen_function_label"
PAIRWISE_DELTA_COL = "pairwise_delta_best_vs_second"
H0_PAIRWISE_WEIGHT_COL = "h0_pairwise_akaike_weight"
H1_PAIRWISE_WEIGHT_COL = "h1_pairwise_akaike_weight"
PAIRWISE_BEST_WEIGHT_COL = "pairwise_best_akaike_weight"
PAIRWISE_DELTA_SUPPORT_COL = "pairwise_delta_support"
PAIRWISE_DELTA_SUPPORT_CLASS_COL = "pairwise_delta_support_class"
PAIRWISE_WEIGHT_SUPPORT_COL = "pairwise_weight_support"
PAIRWISE_WEIGHT_SUPPORT_CLASS_COL = "pairwise_weight_support_class"


def parse_parameters_cell(value: object) -> List[object]:
    """Convert a list-like CSV cell into a Python list."""
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)

    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass

    if not isinstance(value, str) or not value.strip():
        return []

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(value)
        except (json.JSONDecodeError, SyntaxError, ValueError, TypeError):
            continue
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, tuple):
            return list(parsed)

    return []


def require_columns(
    df: pd.DataFrame,
    required: List[str],
    source: Path,
) -> None:
    """Raise a clear error when an input table has missing columns."""
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def coerce_summary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize numeric and parameter columns in the long model-summary table."""
    result = df.copy()

    for column in (
        core.LIKELIHOOD_COL,
        core.AICC_COL,
        core.BASE_CHROM_NUM_COL,
        core.ROOT_CHROM_NUM_COL,
        ac.NUM_OF_EVENTS_COL,
    ):
        result[column] = pd.to_numeric(result[column], errors="coerce")

    result[ac.PARAMS_COL] = result[ac.PARAMS_COL].apply(parse_parameters_cell)
    return result


def default_baseline_summary_file(transition: str) -> Path:
    return (
        ac.BASELINE_ANALYSIS_DIR
        / transition
        / f"{transition}_{ac.MODELS_SUMMARY_SUFFIX}"
    )


def default_exponential_parsed_file(transition: str) -> Path:
    configuration = f"{core.M1_LABEL}_{core.LABEL_EXP}_{transition}"
    return (
        core.PARSED_RESULTS_ROOT
        / f"{core.PARSED_RESULTS_FILE_PREFIX}_{configuration}.csv"
    )


def default_output_dir(transition: str) -> Path:
    return (
        core.ANALYSIS_DIR
        / "exponential_vs_linear"
        / transition
        / "feature_analysis_inputs"
    )


def load_constant_and_ignore_rows(
    baseline_summary_file: Path,
    transition: str,
) -> pd.DataFrame:
    """Load standardized constant and ignore rows from the baseline summary."""
    if not baseline_summary_file.exists():
        raise FileNotFoundError(
            f"Baseline model-summary table not found: {baseline_summary_file}"
        )

    df = pd.read_csv(baseline_summary_file)
    require_columns(df, ac.BASELINE_SUMMARY_TABLE_COLS, baseline_summary_file)

    df = df[
        (df[core.LABEL_TESTED_TRANSITION_COL] == transition)
        & df[core.LABEL_FUNC_TYPE_COL].isin([H0_LABEL, IGNORE_LABEL])
    ].copy()

    found_models = set(df[core.LABEL_FUNC_TYPE_COL].dropna().unique())
    expected_models = {H0_LABEL, IGNORE_LABEL}
    if found_models != expected_models:
        raise ValueError(
            f"Expected baseline rows for {sorted(expected_models)}, "
            f"but found {sorted(found_models)} in {baseline_summary_file}"
        )

    return coerce_summary_columns(df)


def load_exponential_rows(
    exponential_parsed_file: Path,
    transition: str,
) -> pd.DataFrame:
    """Normalize parsed exponential results to the baseline summary schema."""
    if not exponential_parsed_file.exists():
        raise FileNotFoundError(
            f"Parsed exponential results not found: {exponential_parsed_file}"
        )

    parsed_df = pd.read_csv(exponential_parsed_file)
    exp_df = build_single_config_df(parsed_df, transition)

    found_models = set(exp_df[core.LABEL_FUNC_TYPE_COL].dropna().unique())
    if found_models != {H1_LABEL}:
        raise ValueError(
            f"Expected only '{H1_LABEL}' rows in {exponential_parsed_file}, "
            f"but found {sorted(found_models)}"
        )

    return coerce_summary_columns(exp_df)


def build_candidate_summary(
    baseline_rows: pd.DataFrame,
    exponential_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Combine constant, exponential, and ignore into one long table."""
    summary_df = pd.concat(
        [baseline_rows, exponential_rows],
        ignore_index=True,
    ).reindex(columns=ac.BASELINE_SUMMARY_TABLE_COLS)

    summary_df = coerce_summary_columns(summary_df)

    duplicates = summary_df.duplicated(
        [core.FAMILY_NAME_COL, core.LABEL_FUNC_TYPE_COL],
        keep=False,
    )
    if duplicates.any():
        duplicate_rows = summary_df.loc[
            duplicates,
            [core.FAMILY_NAME_COL, core.LABEL_FUNC_TYPE_COL, core.CONFIG_COL],
        ]
        raise ValueError(
            "Expected one row per family/model, but found duplicates:\n"
            f"{duplicate_rows.head(20).to_string(index=False)}"
        )

    return summary_df.sort_values(
        [core.FAMILY_NAME_COL, core.LABEL_FUNC_TYPE_COL]
    ).reset_index(drop=True)


def build_completeness_report(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Report missing rows or invalid AICc values for each candidate model."""
    rows: List[Dict[str, object]] = []

    for family, family_df in summary_df.groupby(core.FAMILY_NAME_COL, sort=True):
        row: Dict[str, object] = {core.FAMILY_NAME_COL: family}
        missing_or_invalid: List[str] = []

        for model in CANDIDATE_MODELS:
            model_df = family_df[
                family_df[core.LABEL_FUNC_TYPE_COL] == model
            ]
            valid = len(model_df) == 1 and pd.notna(
                model_df.iloc[0][core.AICC_COL]
            )
            row[f"valid_{model}"] = valid
            if not valid:
                missing_or_invalid.append(model)

        row["missing_or_invalid_models"] = ",".join(missing_or_invalid)
        row["included_in_comparison"] = not missing_or_invalid
        rows.append(row)

    return pd.DataFrame(rows)


def keep_complete_families(
    summary_df: pd.DataFrame,
    completeness_df: pd.DataFrame,
) -> pd.DataFrame:
    """Retain families with valid constant, exponential, and ignore results."""
    included = completeness_df.loc[
        completeness_df["included_in_comparison"],
        core.FAMILY_NAME_COL,
    ]

    complete_df = summary_df[
        summary_df[core.FAMILY_NAME_COL].isin(included)
    ].copy()

    expected_rows = len(included) * len(CANDIDATE_MODELS)
    if len(complete_df) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} complete model rows, "
            f"but found {len(complete_df)}"
        )

    return complete_df.reset_index(drop=True)


def get_parameters(family_df: pd.DataFrame, model: str) -> List[object]:
    model_df = family_df[
        family_df[core.LABEL_FUNC_TYPE_COL] == model
    ]
    if len(model_df) != 1:
        raise ValueError(
            f"Expected one '{model}' row for "
            f"{family_df.iloc[0][core.FAMILY_NAME_COL]}"
        )
    return parse_parameters_cell(model_df.iloc[0][ac.PARAMS_COL])


def build_pairwise_metrics(family_df: pd.DataFrame) -> Dict[str, object]:
    """
    Calculate direct H0-vs-H1 support, excluding ignore.

    signed_delta_aicc_h0_minus_h1 = AICc(H0) - AICc(H1):
    positive values favor H1; negative values favor H0.
    """
    pair_df = family_df[
        family_df[core.LABEL_FUNC_TYPE_COL].isin([H0_LABEL, H1_LABEL])
    ].copy()

    pair_df = add_akaike_weights(pair_df).sort_values(
        [core.AICC_COL, core.LABEL_FUNC_TYPE_COL]
    ).reset_index(drop=True)

    h0_row = pair_df[pair_df[core.LABEL_FUNC_TYPE_COL] == H0_LABEL].iloc[0]
    h1_row = pair_df[pair_df[core.LABEL_FUNC_TYPE_COL] == H1_LABEL].iloc[0]
    best_row = pair_df.iloc[0]
    second_row = pair_df.iloc[1]

    h0_aicc = float(h0_row[core.AICC_COL])
    h1_aicc = float(h1_row[core.AICC_COL])
    pairwise_delta = float(
        second_row[core.AICC_COL] - best_row[core.AICC_COL]
    )
    best_weight = float(best_row[ac.AKAIKE_WEIGHT_COL])

    return {
        "h0_function_label": H0_LABEL,
        "h1_function_label": H1_LABEL,
        H0_AICC_COL: h0_aicc,
        H1_AICC_COL: h1_aicc,
        SIGNED_PAIRWISE_DELTA_COL: h0_aicc - h1_aicc,
        PAIRWISE_CHOSEN_COL: best_row[core.LABEL_FUNC_TYPE_COL],
        PAIRWISE_DELTA_COL: pairwise_delta,
        H0_PAIRWISE_WEIGHT_COL: float(h0_row[ac.AKAIKE_WEIGHT_COL]),
        H1_PAIRWISE_WEIGHT_COL: float(h1_row[ac.AKAIKE_WEIGHT_COL]),
        PAIRWISE_BEST_WEIGHT_COL: best_weight,
        PAIRWISE_DELTA_SUPPORT_COL: delta_support_label(pairwise_delta),
        PAIRWISE_DELTA_SUPPORT_CLASS_COL: delta_support_class(pairwise_delta),
        PAIRWISE_WEIGHT_SUPPORT_COL: weight_support_label(best_weight),
        PAIRWISE_WEIGHT_SUPPORT_CLASS_COL: weight_support_class(best_weight),
    }


def build_chosen_model_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Choose the best model among constant, exponential, and ignore."""
    chosen_rows: List[Dict[str, object]] = []

    for family, family_df in summary_df.groupby(core.FAMILY_NAME_COL, sort=True):
        weighted_df = add_akaike_weights(family_df).sort_values(
            [core.AICC_COL, core.LABEL_FUNC_TYPE_COL]
        ).reset_index(drop=True)

        if len(weighted_df) != len(CANDIDATE_MODELS):
            raise ValueError(
                f"Family '{family}' does not have all candidate models"
            )

        best_row = weighted_df.iloc[0]
        second_row = weighted_df.iloc[1]
        best_aicc = float(best_row[core.AICC_COL])
        second_aicc = float(second_row[core.AICC_COL])
        global_delta = second_aicc - best_aicc
        best_weight = float(best_row[ac.AKAIKE_WEIGHT_COL])

        constant_params = get_parameters(weighted_df, H0_LABEL)
        exp_params = get_parameters(weighted_df, H1_LABEL)

        chosen_row: Dict[str, object] = {
            core.FAMILY_NAME_COL: family,
            core.LABEL_TESTED_TRANSITION_COL:
                best_row[core.LABEL_TESTED_TRANSITION_COL],
            ac.CHOSEN_FUNCTION_LABEL_COL:
                best_row[core.LABEL_FUNC_TYPE_COL],
            ac.CHOSEN_CONFIG_COL: best_row[core.CONFIG_COL],
            core.LIKELIHOOD_COL: best_row[core.LIKELIHOOD_COL],
            core.AICC_COL: best_aicc,
            ac.SECOND_BEST_AICC_COL: second_aicc,
            ac.DELTA_BEST_VS_SECOND_COL: global_delta,
            ac.BEST_AKAIKE_WEIGHT_COL: best_weight,
            ac.DELTA_SUPPORT_COL: delta_support_label(global_delta),
            ac.DELTA_SUPPORT_CLASS_COL: delta_support_class(global_delta),
            ac.WEIGHT_SUPPORT_COL: weight_support_label(best_weight),
            ac.WEIGHT_SUPPORT_CLASS_COL: weight_support_class(best_weight),
            ac.CHOSEN_MODEL_PARAMS_COL:
                parse_parameters_cell(best_row[ac.PARAMS_COL]),
            ac.CONST_VAL_COL:
                constant_params[0] if constant_params else pd.NA,
            EXP_P1_COL:
                exp_params[0] if len(exp_params) >= 1 else pd.NA,
            EXP_P2_COL:
                exp_params[1] if len(exp_params) >= 2 else pd.NA,
        }
        chosen_row.update(build_pairwise_metrics(weighted_df))
        chosen_rows.append(chosen_row)

    chosen_df = pd.DataFrame(chosen_rows)
    return chosen_df.sort_values(core.FAMILY_NAME_COL).reset_index(drop=True)


def build_feature_tables(
    summary_df: pd.DataFrame,
    chosen_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """Create feature-summary tables for all three candidate models."""
    enriched_df = enrich_summary_with_metadata(summary_df, metadata_df)
    flagged_df = add_chosen_model_flag(enriched_df, chosen_df)

    return {
        model: build_function_features_table(flagged_df, model)
        for model in CANDIDATE_MODELS
    }


def save_outputs(
    summary_df: pd.DataFrame,
    chosen_df: pd.DataFrame,
    feature_tables: Dict[str, pd.DataFrame],
    completeness_df: pd.DataFrame,
    output_dir: Path,
    transition: str,
) -> List[str]:
    """Save all feature-analysis inputs and the incomplete-family report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_file = output_dir / f"{transition}_{ac.MODELS_SUMMARY_SUFFIX}"
    chosen_file = output_dir / f"{transition}_{ac.CHOSEN_MODEL_SUFFIX}"
    missing_file = (
        output_dir
        / f"{transition}_{COMPARISON_NAME}_missing_families.csv"
    )

    summary_df.to_csv(summary_file, index=False)
    chosen_df.to_csv(chosen_file, index=False)
    completeness_df.loc[
        ~completeness_df["included_in_comparison"]
    ].to_csv(missing_file, index=False)

    output_paths = [str(summary_file), str(chosen_file), str(missing_file)]

    for model, feature_df in feature_tables.items():
        feature_file = (
            output_dir
            / f"{transition}_{model}_{ac.FEATURES_SUMMARY_SUFFIX}"
        )
        feature_df.to_csv(feature_file, index=False)
        output_paths.append(str(feature_file))
        print(f"[✓] Saved {model} feature table: {feature_file}")

    print(f"[✓] Saved model summary: {summary_file}")
    print(f"[✓] Saved chosen-model table: {chosen_file}")
    print(f"[✓] Saved missing-family report: {missing_file}")

    return output_paths


def run(
    transition: str,
    baseline_summary_file: Path,
    exponential_parsed_file: Path,
    metadata_file: Path,
    output_dir: Path,
) -> Dict[str, object]:
    baseline_rows = load_constant_and_ignore_rows(
        baseline_summary_file,
        transition,
    )
    exponential_rows = load_exponential_rows(
        exponential_parsed_file,
        transition,
    )

    candidate_summary = build_candidate_summary(
        baseline_rows,
        exponential_rows,
    )
    completeness_df = build_completeness_report(candidate_summary)
    complete_summary = keep_complete_families(
        candidate_summary,
        completeness_df,
    )

    chosen_df = build_chosen_model_table(complete_summary)
    metadata_df = load_metadata_table(metadata_file)
    feature_tables = build_feature_tables(
        complete_summary,
        chosen_df,
        metadata_df,
    )

    outputs = save_outputs(
        complete_summary,
        chosen_df,
        feature_tables,
        completeness_df,
        output_dir,
        transition,
    )

    return {
        "n_candidate_families": len(completeness_df),
        "n_complete_families": chosen_df[core.FAMILY_NAME_COL].nunique(),
        "n_excluded_families":
            int((~completeness_df["included_in_comparison"]).sum()),
        "chosen_counts":
            chosen_df[ac.CHOSEN_FUNCTION_LABEL_COL].value_counts().to_dict(),
        "outputs": outputs,
    }


def parse_args() -> argparse.Namespace:
    """Parse input/output paths and the tested transition."""
    parser = argparse.ArgumentParser(
        description=(
            "Build constant-vs-exponential feature-analysis input tables.\n\n"
            "Required upstream files:\n"
            "  1. Baseline <transition>_models_summary_table.csv, created by\n"
            "     build_transition_summary_tables.py.\n"
            "  2. parsed_results_M1_exponential_<transition>.csv, created by\n"
            "     the ChromEvol raw-results parser.\n"
            "  3. all_families_data_summary.csv from preprocessing.\n\n"
            "The script compares constant, exponential, and ignore. Ignore\n"
            "winners remain in the outputs and should be excluded downstream\n"
            "by the generalized H0-vs-H1 feature-analysis script."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--transition_label",
        required=True,
        choices=list(core.LABEL_TRANSITIONS_ORDERED),
        help="Transition to prepare, for example gain, loss, or dupl.",
    )
    parser.add_argument(
        "--baseline_models_summary_file",
        type=Path,
        default=None,
        help=(
            "Existing baseline model-summary CSV. By default, it is read "
            "from analysis/baseline_models/<transition>/."
        ),
    )
    parser.add_argument(
        "--exponential_parsed_results_file",
        type=Path,
        default=None,
        help=(
            "Parsed exponential results CSV. By default, it is read from "
            "chromevol_parsed_results/parsed_results_M1_exponential_"
            "<transition>.csv."
        ),
    )
    parser.add_argument(
        "--families_metadata_file",
        type=Path,
        default=core.ALL_FAMILIES_DATA_SUMMARY_FILE,
        help="Family metadata CSV from preprocessing.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Default: analysis/exponential_vs_linear/"
            "<transition>/feature_analysis_inputs/."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transition = args.transition_label

    baseline_file = (
        args.baseline_models_summary_file
        or default_baseline_summary_file(transition)
    )
    exponential_file = (
        args.exponential_parsed_results_file
        or default_exponential_parsed_file(transition)
    )
    output_dir = args.output_dir or default_output_dir(transition)

    result = run(
        transition=transition,
        baseline_summary_file=baseline_file,
        exponential_parsed_file=exponential_file,
        metadata_file=args.families_metadata_file,
        output_dir=output_dir,
    )

    print(
        f"[✓] Complete families: {result['n_complete_families']} / "
        f"{result['n_candidate_families']}"
    )
    print(f"[✓] Excluded incomplete families: {result['n_excluded_families']}")
    print(f"[✓] Chosen-model counts: {result['chosen_counts']}")

    # log_run(
    #     step="analysis",
    #     script=Path(__file__),
    #     params={
    #         "transition_label": transition,
    #         "comparison": COMPARISON_NAME,
    #         "candidate_models": list(CANDIDATE_MODELS),
    #         "baseline_models_summary_file": str(baseline_file),
    #         "exponential_parsed_results_file": str(exponential_file),
    #         "families_metadata_file": str(args.families_metadata_file),
    #         "n_candidate_families": result["n_candidate_families"],
    #         "n_complete_families": result["n_complete_families"],
    #         "n_excluded_families": result["n_excluded_families"],
    #         "chosen_counts": result["chosen_counts"],
    #     },
    #     outputs=result["outputs"],
    #     description=(
    #         "Built constant-vs-exponential feature-analysis inputs for "
    #         f"transition '{transition}'."
    #     ),
    #     notes=(
    #         "Model selection includes constant, exponential, and ignore. "
    #         "Only families with valid AICc for all three models are included."
    #     ),
    #     log_relative_path=(
    #         Path("exponential_vs_linear")
    #         / transition
    #         / f"{COMPARISON_NAME}_feature_inputs.log"
    #     ),
    # )


if __name__ == "__main__":
    main()