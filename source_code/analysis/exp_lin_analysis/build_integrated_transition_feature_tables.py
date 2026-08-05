#!/usr/bin/env python3
"""
Build integrated model/feature tables for ChromEvol transition analyses.

For each requested transition (gain, loss, dupl), this script reads the parsed
results for four candidate function models:

    constant, linear, exponential, ignore

and produces two tables:

1. <transition>_all_models_long.csv
   One row per family x function model. Model-specific inferred quantities stay
   on their own model row. Family metadata is repeated across model rows.

   Model-selection output kept in the long table:
       chosen_model_overall
       is_chosen_overall
       is_chosen_exponential_analysis
       is_chosen_linear_analysis

   Pairwise chosen-model labels are used internally to create the binary flags
   but are not written to the long output.

2. <transition>_family_features_wide.csv
   One row per family. Only model selection and model-specific inferred features
   are retained. Model-specific quantities use <feature>_<function>, e.g.:

       root_chrom_num_constant
       root_chrom_num_linear
       root_chrom_num_exponential
       root_chrom_num_ignore

   Family-level metadata remains unsuffixed.

   Model-selection output kept in the wide table:
       chosen_model_overall
       chosen_model_exponential_analysis
       chosen_model_linear_analysis

   The wide table deliberately excludes:
       validation/missing-model helper columns
       configuration columns
       parameter-count columns
       param_0 / param_1 / param_2 columns
       serialized parameter columns

A family is not silently removed when a model is missing or has invalid AICc.
For a comparison that cannot be made, the corresponding chosen-model value is
left missing; in the long table all binary flags for that comparison are 0.

Default inputs are read from the project paths in source_code/constants.py.
The script does not run ChromEvol and does not parse raw ChromEvol output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

from source_code import constants as core
from source_code.analysis import analysis_constants as ac
from source_code.analysis.baseline_models.build_transition_summary_tables import (
    build_single_config_df,
)
from source_code.analysis.metadata_analysis import load_family_metadata


# ---------------------------------------------------------------------------
# Analysis definition
# ---------------------------------------------------------------------------

TARGET_TRANSITIONS: tuple[str, ...] = (
    core.LABEL_DUPL,
    core.LABEL_GAIN,
    core.LABEL_LOSS,
)

MODEL_ORDER: tuple[str, ...] = (
    core.LABEL_CONSTANT,
    core.LABEL_LINEAR,
    core.LABEL_EXP,
    core.LABEL_IGNORE,
)

COMPARISON_MODELS: Mapping[str, tuple[str, ...]] = {
    "overall": MODEL_ORDER,
    "exponential_analysis": (
        core.LABEL_CONSTANT,
        core.LABEL_EXP,
        core.LABEL_IGNORE,
    ),
    "linear_analysis": (
        core.LABEL_CONSTANT,
        core.LABEL_LINEAR,
        core.LABEL_IGNORE,
    ),
}

# ONLY these model-specific quantities are written to the final wide table.
WIDE_MODEL_FEATURE_COLS: tuple[str, ...] = (
    core.LIKELIHOOD_COL,
    core.AICC_COL,
    core.BASE_CHROM_NUM_COL,
    core.ROOT_CHROM_NUM_COL,
    ac.NUM_OF_EVENTS_COL,
)

# Helper columns that should never be copied from metadata into final outputs.
EXCLUDED_METADATA_COLS: set[str] = {
    "missing_metadata",
}


# ---------------------------------------------------------------------------
# Input loading / normalization
# ---------------------------------------------------------------------------


def model_configurations(transition: str) -> Dict[str, str]:
    """Return function label -> parsed-results configuration for a transition."""
    return {
        core.LABEL_CONSTANT: core.M0_LABEL,
        core.LABEL_LINEAR: f"{core.M1_LABEL}_{core.LABEL_LINEAR}_{transition}",
        core.LABEL_EXP: f"{core.M1_LABEL}_{core.LABEL_EXP}_{transition}",
        core.LABEL_IGNORE: f"{core.M1_LABEL}_{core.LABEL_IGNORE}_{transition}",
    }


def parsed_results_file(parsed_results_root: Path, configuration: str) -> Path:
    return (
        parsed_results_root
        / f"{core.PARSED_RESULTS_FILE_PREFIX}_{configuration}.csv"
    )


def load_and_standardize_model(
    parsed_results_root: Path,
    transition: str,
    function_label: str,
    configuration: str,
) -> pd.DataFrame:
    """Load one parsed-results file and normalize it to the shared model schema."""
    input_file = parsed_results_file(parsed_results_root, configuration)
    if not input_file.exists():
        raise FileNotFoundError(f"Parsed results file not found: {input_file}")

    parsed_df = pd.read_csv(input_file)
    model_df = build_single_config_df(parsed_df, transition)

    found_labels = set(model_df[core.LABEL_FUNC_TYPE_COL].dropna().unique())
    if found_labels != {function_label}:
        raise ValueError(
            f"Expected only function label '{function_label}' in {input_file}, "
            f"but found {sorted(found_labels)}"
        )

    numeric_cols = (
        core.LIKELIHOOD_COL,
        core.AICC_COL,
        core.BASE_CHROM_NUM_COL,
        core.ROOT_CHROM_NUM_COL,
        ac.NUM_OF_EVENTS_COL,
    )
    for column in numeric_cols:
        model_df[column] = pd.to_numeric(model_df[column], errors="coerce")

    # Keep parameter details in the LONG table only.
    model_df[core.TESTED_TRANSITION_N_PARAMS_COL] = model_df[ac.PARAMS_COL].apply(
        lambda params: len(params) if isinstance(params, list) else 0
    )

    for index, column in enumerate(
        (core.PARAM_0_COL, core.PARAM_1_COL, core.PARAM_2_COL)
    ):
        model_df[column] = model_df[ac.PARAMS_COL].apply(
            lambda params, i=index: (
                params[i]
                if isinstance(params, list) and len(params) > i
                else pd.NA
            )
        )
        model_df[column] = pd.to_numeric(model_df[column], errors="coerce")

    return model_df


def build_long_model_table(
    transition: str,
    parsed_results_root: Path,
) -> pd.DataFrame:
    """Combine constant, linear, exponential, and ignore model rows."""
    parts: List[pd.DataFrame] = []

    for function_label, configuration in model_configurations(transition).items():
        parts.append(
            load_and_standardize_model(
                parsed_results_root=parsed_results_root,
                transition=transition,
                function_label=function_label,
                configuration=configuration,
            )
        )

    long_df = pd.concat(parts, ignore_index=True)

    duplicate_mask = long_df.duplicated(
        [core.FAMILY_NAME_COL, core.LABEL_FUNC_TYPE_COL],
        keep=False,
    )
    if duplicate_mask.any():
        duplicates = long_df.loc[
            duplicate_mask,
            [core.FAMILY_NAME_COL, core.LABEL_FUNC_TYPE_COL, core.CONFIG_COL],
        ]
        raise ValueError(
            "Expected at most one row per family/function, but found duplicates:\n"
            f"{duplicates.head(30).to_string(index=False)}"
        )

    return long_df.sort_values(
        [core.FAMILY_NAME_COL, core.LABEL_FUNC_TYPE_COL]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------


def has_one_valid_aicc(family_df: pd.DataFrame, model: str) -> bool:
    """True only when a family has exactly one finite-AICc row for the model."""
    model_df = family_df[family_df[core.LABEL_FUNC_TYPE_COL] == model]
    if len(model_df) != 1:
        return False

    aicc = pd.to_numeric(model_df.iloc[0][core.AICC_COL], errors="coerce")
    return bool(pd.notna(aicc) and np.isfinite(float(aicc)))


def choose_model(
    family_df: pd.DataFrame,
    candidate_models: Sequence[str],
) -> object:
    """Choose the minimum-AICc model, or pd.NA if comparison is incomplete."""
    if not all(has_one_valid_aicc(family_df, model) for model in candidate_models):
        return pd.NA

    candidates = family_df[
        family_df[core.LABEL_FUNC_TYPE_COL].isin(candidate_models)
    ][[core.LABEL_FUNC_TYPE_COL, core.AICC_COL]].copy()

    candidates[core.AICC_COL] = pd.to_numeric(
        candidates[core.AICC_COL], errors="coerce"
    )

    # Explicit model order provides deterministic tie breaking.
    order = {model: index for index, model in enumerate(candidate_models)}
    candidates["_model_order"] = candidates[core.LABEL_FUNC_TYPE_COL].map(order)
    candidates = candidates.sort_values([core.AICC_COL, "_model_order"])

    return candidates.iloc[0][core.LABEL_FUNC_TYPE_COL]


def build_family_selection_table(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build ONLY the three family-level chosen-model labels.

    Validation and missing-model helper columns are intentionally not created.
    """
    rows: List[Dict[str, object]] = []

    for family, family_df in long_df.groupby(core.FAMILY_NAME_COL, sort=True):
        row: Dict[str, object] = {core.FAMILY_NAME_COL: family}

        for comparison_name, candidate_models in COMPARISON_MODELS.items():
            row[f"chosen_model_{comparison_name}"] = choose_model(
                family_df,
                candidate_models,
            )

        rows.append(row)

    return pd.DataFrame(rows).sort_values(core.FAMILY_NAME_COL).reset_index(
        drop=True
    )


# ---------------------------------------------------------------------------
# Metadata and output tables
# ---------------------------------------------------------------------------


def load_metadata_preserving_extra_columns(metadata_file: Path) -> pd.DataFrame:
    """
    Load metadata while retaining legitimate extra future columns.

    Known helper columns such as missing_metadata are removed if already present
    in the metadata source.
    """
    metadata_df = load_family_metadata(metadata_file)

    cols_to_drop = [
        column
        for column in metadata_df.columns
        if (
            column in EXCLUDED_METADATA_COLS
            or column.startswith("valid_")
            or column.startswith("missing_or_invalid_")
        )
    ]
    return metadata_df.drop(columns=cols_to_drop, errors="ignore")


def add_metadata(
    long_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach family metadata to every model row without validation columns."""
    model_columns = set(long_df.columns)
    metadata_only_columns = set(metadata_df.columns) - {core.FAMILY_NAME_COL}
    collisions = model_columns.intersection(metadata_only_columns)
    if collisions:
        raise ValueError(
            "Metadata columns collide with model columns: "
            f"{sorted(collisions)}"
        )

    return long_df.merge(
        metadata_df,
        on=core.FAMILY_NAME_COL,
        how="left",
        validate="many_to_one",
    )


def add_selection_to_long(
    long_df: pd.DataFrame,
    selection_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach selection results to long data.

    Long output keeps:
        chosen_model_overall
        is_chosen_overall
        is_chosen_exponential_analysis
        is_chosen_linear_analysis

    Pairwise chosen-model labels are removed after creating their binary flags.
    """
    result = long_df.merge(
        selection_df,
        on=core.FAMILY_NAME_COL,
        how="left",
        validate="many_to_one",
    )

    for comparison_name in COMPARISON_MODELS:
        chosen_col = f"chosen_model_{comparison_name}"
        flag_col = f"is_chosen_{comparison_name}"
        chosen_values = result[chosen_col]

        result[flag_col] = (
            chosen_values.notna()
            & result[core.LABEL_FUNC_TYPE_COL].astype(str).eq(
                chosen_values.fillna("").astype(str)
            )
        ).astype(int)

    return result.drop(
        columns=[
            "chosen_model_exponential_analysis",
            "chosen_model_linear_analysis",
        ],
        errors="ignore",
    )


def serialize_parameter_lists(df: pd.DataFrame) -> pd.DataFrame:
    """Serialize list-valued parameter cells consistently for CSV output."""
    result = df.copy()
    parameter_columns = [
        column
        for column in result.columns
        if column == ac.PARAMS_COL or column.startswith(f"{ac.PARAMS_COL}_")
    ]

    for column in parameter_columns:
        result[column] = result[column].apply(
            lambda value: json.dumps(value)
            if isinstance(value, (list, tuple))
            else value
        )

    return result


def build_wide_family_table(
    long_with_metadata_df: pd.DataFrame,
    selection_df: pd.DataFrame,
    metadata_columns: Sequence[str],
    transition: str,
) -> pd.DataFrame:
    """
    Build one row per family with an explicit wide-table whitelist.

    Wide output contains only:
      - family name + tested transition
      - legitimate family metadata
      - three chosen-model labels
      - likelihood, AICc, base chromosome number, root chromosome number,
        and tested-transition event count for each function model
    """
    families = pd.DataFrame(
        {
            core.FAMILY_NAME_COL: sorted(
                long_with_metadata_df[core.FAMILY_NAME_COL].dropna().unique()
            )
        }
    )
    families[core.LABEL_TESTED_TRANSITION_COL] = transition

    metadata_cols_no_family = [
        column
        for column in metadata_columns
        if (
            column != core.FAMILY_NAME_COL
            and column not in EXCLUDED_METADATA_COLS
            and not column.startswith("valid_")
            and not column.startswith("missing_or_invalid_")
        )
    ]

    metadata_family_df = (
        long_with_metadata_df[
            [core.FAMILY_NAME_COL, *metadata_cols_no_family]
        ]
        .drop_duplicates(subset=core.FAMILY_NAME_COL)
        .reset_index(drop=True)
    )

    wide_df = families.merge(
        metadata_family_df,
        on=core.FAMILY_NAME_COL,
        how="left",
        validate="one_to_one",
    )
    wide_df = wide_df.merge(
        selection_df,
        on=core.FAMILY_NAME_COL,
        how="left",
        validate="one_to_one",
    )

    for model in MODEL_ORDER:
        model_df = long_with_metadata_df[
            long_with_metadata_df[core.LABEL_FUNC_TYPE_COL] == model
        ][[core.FAMILY_NAME_COL, *WIDE_MODEL_FEATURE_COLS]].copy()

        model_df = model_df.rename(
            columns={
                column: f"{column}_{model}"
                for column in WIDE_MODEL_FEATURE_COLS
            }
        )

        wide_df = wide_df.merge(
            model_df,
            on=core.FAMILY_NAME_COL,
            how="left",
            validate="one_to_one",
        )

    # Explicit final schema: nothing outside this list is written.
    selection_cols = [
        "chosen_model_overall",
        "chosen_model_exponential_analysis",
        "chosen_model_linear_analysis",
    ]

    model_specific_cols = [
        f"{feature}_{model}"
        for feature in WIDE_MODEL_FEATURE_COLS
        for model in MODEL_ORDER
    ]

    ordered_cols = [
        core.FAMILY_NAME_COL,
        core.LABEL_TESTED_TRANSITION_COL,
        *metadata_cols_no_family,
        *selection_cols,
        *model_specific_cols,
    ]

    return wide_df.reindex(columns=ordered_cols).sort_values(
        core.FAMILY_NAME_COL
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Saving / running
# ---------------------------------------------------------------------------


def transition_output_dir(output_root: Path, transition: str) -> Path:
    output_dir = output_root / transition
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_transition_outputs(
    long_df: pd.DataFrame,
    wide_df: pd.DataFrame,
    output_root: Path,
    transition: str,
) -> List[Path]:
    output_dir = transition_output_dir(output_root, transition)

    long_file = output_dir / f"{transition}_all_models_long.csv"
    wide_file = output_dir / f"{transition}_family_features_wide.csv"

    serialize_parameter_lists(long_df).to_csv(long_file, index=False)
    wide_df.to_csv(wide_file, index=False)

    return [long_file, wide_file]


def run_transition(
    transition: str,
    parsed_results_root: Path,
    metadata_df: pd.DataFrame,
    output_root: Path,
) -> Dict[str, object]:
    long_models_df = build_long_model_table(
        transition=transition,
        parsed_results_root=parsed_results_root,
    )
    selection_df = build_family_selection_table(long_models_df)

    long_with_metadata_df = add_metadata(long_models_df, metadata_df)
    long_final_df = add_selection_to_long(
        long_with_metadata_df,
        selection_df,
    )

    wide_df = build_wide_family_table(
        long_with_metadata_df=long_with_metadata_df,
        selection_df=selection_df,
        metadata_columns=list(metadata_df.columns),
        transition=transition,
    )

    output_files = save_transition_outputs(
        long_df=long_final_df,
        wide_df=wide_df,
        output_root=output_root,
        transition=transition,
    )

    return {
        "transition": transition,
        "n_families": int(wide_df[core.FAMILY_NAME_COL].nunique()),
        "n_long_rows": int(len(long_final_df)),
        "chosen_overall_counts": selection_df[
            "chosen_model_overall"
        ].value_counts(dropna=False).to_dict(),
        "chosen_exponential_counts": selection_df[
            "chosen_model_exponential_analysis"
        ].value_counts(dropna=False).to_dict(),
        "chosen_linear_counts": selection_df[
            "chosen_model_linear_analysis"
        ].value_counts(dropna=False).to_dict(),
        "output_files": output_files,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build integrated long and wide model-feature tables for gain, "
            "loss, and/or duplication analyses."
        )
    )
    parser.add_argument(
        "--transitions",
        nargs="+",
        choices=list(TARGET_TRANSITIONS),
        default=list(TARGET_TRANSITIONS),
        help=(
            "Transitions to build. Default: dupl gain loss. "
            "Example: --transitions dupl"
        ),
    )
    parser.add_argument(
        "--parsed_results_root",
        type=Path,
        default=core.PARSED_RESULTS_ROOT,
        help="Directory containing parsed_results_<configuration>.csv files.",
    )
    parser.add_argument(
        "--families_metadata_file",
        type=Path,
        default=core.ALL_FAMILIES_DATA_SUMMARY_FILE,
        help=(
            "Family metadata CSV. Required standard metadata columns are "
            "validated; legitimate additional metadata columns are preserved."
        ),
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=core.ANALYSIS_DIR / "integrated_model_features",
        help=(
            "Root output directory. Each transition gets its own subdirectory."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    metadata_df = load_metadata_preserving_extra_columns(
        args.families_metadata_file
    )

    print(f"[i] Parsed results root: {args.parsed_results_root}")
    print(f"[i] Metadata file: {args.families_metadata_file}")
    print(f"[i] Output root: {args.output_root}")

    for transition in args.transitions:
        result = run_transition(
            transition=transition,
            parsed_results_root=args.parsed_results_root,
            metadata_df=metadata_df,
            output_root=args.output_root,
        )

        print(f"\n[✓] {transition}: {result['n_families']} families")
        print(f"    long rows: {result['n_long_rows']}")
        print(f"    overall chosen counts: {result['chosen_overall_counts']}")
        print(
            "    exponential-analysis chosen counts: "
            f"{result['chosen_exponential_counts']}"
        )
        print(
            "    linear-analysis chosen counts: "
            f"{result['chosen_linear_counts']}"
        )
        for output_file in result["output_files"]:
            print(f"    saved: {output_file}")


if __name__ == "__main__":
    main()