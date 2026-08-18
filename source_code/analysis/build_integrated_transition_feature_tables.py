#!/usr/bin/env python3
"""
Build integrated ChromEvol model/feature tables for baseline or baseline+exponential analyses.

For each requested transition, the script loads the requested model set, selects the minimum-AICc model per family, and writes:
  - <transition>_all_models_long.csv
  - <transition>_family_features_wide.csv

It also writes one cross-transition family matrix:
  - family_feature_matrix.csv
  - family_feature_matrix.xlsx

The matrix has one row per family and, for each transition, a selected-model column plus all FEATURE_ANALYSIS_COLS prefixed by the transition. Inferred features are taken from the selected model; metadata features are family-level values. The Excel selected-model cells are filled using MODEL_COLOR_MAP.

Model sets:
  baseline  = constant, linear, ignore
  with_exp  = constant, linear, exponential, ignore

A family is not silently removed when a requested model is missing or has invalid AICc. If the requested model comparison cannot be made, selected_model and selected-model inferred features are left missing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, TypedDict

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import PatternFill

from source_code import constants as core
from source_code.analysis import analysis_constants as ac
from source_code.analysis.baseline_models.build_transition_summary_tables import baseline_configs, build_single_config_df, load_families_list
from source_code.analysis.metadata_analysis import load_family_metadata
from source_code.logger import log_run

TARGET_TRANSITIONS: tuple[str, ...] = (core.LABEL_GAIN, core.LABEL_LOSS, core.LABEL_DUPL)

MODEL_SETS: Mapping[str, tuple[str, ...]] = {
    "baseline": ac.BASELINE_FUNC_LABELS_ORDERED,
    "with_exp": (core.LABEL_CONSTANT, core.LABEL_LINEAR, core.LABEL_EXP, core.LABEL_IGNORE),
}

SELECTED_MODEL_COL: str = "selected_model"
IS_SELECTED_MODEL_COL: str = "is_selected_model"

WIDE_MODEL_FEATURE_COLS: tuple[str, ...] = (
    core.LIKELIHOOD_COL,
    core.AICC_COL,
    core.BASE_CHROM_NUM_COL,
    core.ROOT_CHROM_NUM_COL,
    ac.NUM_OF_EVENTS_COL,
)

EXCLUDED_METADATA_COLS: set[str] = {"missing_metadata"}


class RunResult(TypedDict):
    transition: str
    n_families: int
    n_long_rows: int
    selected_model_counts: Dict[object, int]
    selected_features_df: pd.DataFrame
    output_files: List[Path]


def model_configurations(transition: str, models: Sequence[str]) -> Dict[str, str]:
    """Return function label -> parsed-results configuration for the requested model set."""
    configurations = baseline_configs(transition)
    configurations[core.LABEL_EXP] = f"{core.M1_LABEL}_{core.LABEL_EXP}_{transition}"
    return {model: configurations[model] for model in models}


def parsed_results_file(parsed_results_root: Path, configuration: str) -> Path:
    return parsed_results_root / f"{core.PARSED_RESULTS_FILE_PREFIX}_{configuration}.csv"


def load_and_standardize_model(parsed_results_root: Path, transition: str, function_label: str, configuration: str) -> pd.DataFrame:
    """Load one parsed-results file and normalize it to the shared model schema."""
    input_file = parsed_results_file(parsed_results_root, configuration)
    if not input_file.exists():
        raise FileNotFoundError(f"Parsed results file not found: {input_file}")

    parsed_df = pd.read_csv(input_file)
    model_df = build_single_config_df(parsed_df, transition)

    found_labels = set(model_df[core.LABEL_FUNC_TYPE_COL].dropna().unique())
    if found_labels != {function_label}:
        raise ValueError(f"Expected only function label '{function_label}' in {input_file}, but found {sorted(found_labels)}")

    for column in WIDE_MODEL_FEATURE_COLS:
        model_df[column] = pd.to_numeric(model_df[column], errors="coerce")

    model_df[core.TESTED_TRANSITION_N_PARAMS_COL] = model_df[ac.PARAMS_COL].apply(lambda params: len(params) if isinstance(params, list) else 0)

    for index, column in enumerate((core.PARAM_0_COL, core.PARAM_1_COL, core.PARAM_2_COL)):
        model_df[column] = model_df[ac.PARAMS_COL].apply(lambda params, i=index: params[i] if isinstance(params, list) and len(params) > i else pd.NA)
        model_df[column] = pd.to_numeric(model_df[column], errors="coerce")

    return model_df


def build_long_model_table(transition: str, parsed_results_root: Path, models: Sequence[str], families: set[str]) -> pd.DataFrame:
    """Combine the requested model rows for one transition."""
    configurations = model_configurations(transition, models)
    parts: List[pd.DataFrame] = [load_and_standardize_model(parsed_results_root, transition, model, configurations[model]) for model in models]
    long_df = pd.concat(parts, ignore_index=True)
    long_df = long_df[long_df[core.FAMILY_NAME_COL].isin(families)].copy()

    duplicate_mask = long_df.duplicated([core.FAMILY_NAME_COL, core.LABEL_FUNC_TYPE_COL], keep=False)
    if duplicate_mask.any():
        duplicates = long_df.loc[duplicate_mask, [core.FAMILY_NAME_COL, core.LABEL_FUNC_TYPE_COL, core.CONFIG_COL]]
        raise ValueError(f"Expected at most one row per family/function, but found duplicates:\n{duplicates.head(30).to_string(index=False)}")

    return long_df.sort_values([core.FAMILY_NAME_COL, core.LABEL_FUNC_TYPE_COL]).reset_index(drop=True)


def has_one_valid_aicc(family_df: pd.DataFrame, model: str) -> bool:
    """True only when a family has exactly one finite-AICc row for the model."""
    model_df = family_df[family_df[core.LABEL_FUNC_TYPE_COL] == model]
    if len(model_df) != 1:
        return False

    aicc = pd.to_numeric(model_df.iloc[0][core.AICC_COL], errors="coerce")
    return bool(pd.notna(aicc) and np.isfinite(float(aicc)))


def choose_model(family_df: pd.DataFrame, candidate_models: Sequence[str]) -> object:
    """Choose the minimum-AICc model, or pd.NA if the requested comparison is incomplete."""
    if not all(has_one_valid_aicc(family_df, model) for model in candidate_models):
        return pd.NA

    candidates = family_df[family_df[core.LABEL_FUNC_TYPE_COL].isin(candidate_models)][[core.LABEL_FUNC_TYPE_COL, core.AICC_COL]].copy()
    candidates[core.AICC_COL] = pd.to_numeric(candidates[core.AICC_COL], errors="coerce")

    order = {model: index for index, model in enumerate(candidate_models)}
    candidates["_model_order"] = candidates[core.LABEL_FUNC_TYPE_COL].map(order)
    candidates = candidates.sort_values([core.AICC_COL, "_model_order"])

    return candidates.iloc[0][core.LABEL_FUNC_TYPE_COL]


def build_family_selection_table(long_df: pd.DataFrame, candidate_models: Sequence[str]) -> pd.DataFrame:
    """Build one selected-model label per family for the requested model set."""
    rows: List[Dict[str, object]] = []

    for family, family_df in long_df.groupby(core.FAMILY_NAME_COL, sort=True):
        rows.append({core.FAMILY_NAME_COL: family, SELECTED_MODEL_COL: choose_model(family_df, candidate_models)})

    return pd.DataFrame(rows).sort_values(core.FAMILY_NAME_COL).reset_index(drop=True)


def load_metadata_preserving_extra_columns(metadata_file: Path) -> pd.DataFrame:
    """Load metadata while retaining legitimate extra future columns."""
    metadata_df = load_family_metadata(metadata_file)
    cols_to_drop = [column for column in metadata_df.columns if column in EXCLUDED_METADATA_COLS or column.startswith("valid_") or column.startswith("missing_or_invalid_")]
    return metadata_df.drop(columns=cols_to_drop, errors="ignore")


def add_metadata(long_df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Attach family metadata to every model row."""
    metadata_only_columns = set(metadata_df.columns) - {core.FAMILY_NAME_COL}
    collisions = set(long_df.columns).intersection(metadata_only_columns)

    if collisions:
        raise ValueError(f"Metadata columns collide with model columns: {sorted(collisions)}")

    return long_df.merge(metadata_df, on=core.FAMILY_NAME_COL, how="left", validate="many_to_one")


def add_selection_to_long(long_df: pd.DataFrame, selection_df: pd.DataFrame) -> pd.DataFrame:
    """Attach selected-model label and binary flag to the long table."""
    result = long_df.merge(selection_df, on=core.FAMILY_NAME_COL, how="left", validate="many_to_one")
    result[IS_SELECTED_MODEL_COL] = (result[SELECTED_MODEL_COL].notna() & result[core.LABEL_FUNC_TYPE_COL].astype(str).eq(result[SELECTED_MODEL_COL].fillna("").astype(str))).astype(int)
    return result


def serialize_parameter_lists(df: pd.DataFrame) -> pd.DataFrame:
    """Serialize list-valued parameter cells consistently for CSV output."""
    result = df.copy()
    parameter_columns = [column for column in result.columns if column == ac.PARAMS_COL or column.startswith(f"{ac.PARAMS_COL}_")]

    for column in parameter_columns:
        result[column] = result[column].apply(lambda value: json.dumps(value) if isinstance(value, (list, tuple)) else value)

    return result


def build_wide_family_table(long_with_metadata_df: pd.DataFrame, selection_df: pd.DataFrame, metadata_columns: Sequence[str], models: Sequence[str], transition: str) -> pd.DataFrame:
    """Build one row per family with metadata, selection, and model-specific inferred quantities."""
    families = pd.DataFrame({core.FAMILY_NAME_COL: sorted(long_with_metadata_df[core.FAMILY_NAME_COL].dropna().unique())})
    families[core.LABEL_TESTED_TRANSITION_COL] = transition

    metadata_cols_no_family = [column for column in metadata_columns if column != core.FAMILY_NAME_COL and column not in EXCLUDED_METADATA_COLS and not column.startswith("valid_") and not column.startswith("missing_or_invalid_")]
    metadata_family_df = long_with_metadata_df[[core.FAMILY_NAME_COL, *metadata_cols_no_family]].drop_duplicates(subset=core.FAMILY_NAME_COL).reset_index(drop=True)

    wide_df = families.merge(metadata_family_df, on=core.FAMILY_NAME_COL, how="left", validate="one_to_one").merge(selection_df, on=core.FAMILY_NAME_COL, how="left", validate="one_to_one")

    for model in models:
        model_df = long_with_metadata_df[long_with_metadata_df[core.LABEL_FUNC_TYPE_COL] == model][[core.FAMILY_NAME_COL, *WIDE_MODEL_FEATURE_COLS]].copy()
        model_df = model_df.rename(columns={column: f"{column}_{model}" for column in WIDE_MODEL_FEATURE_COLS})
        wide_df = wide_df.merge(model_df, on=core.FAMILY_NAME_COL, how="left", validate="one_to_one")

    model_specific_cols = [f"{feature}_{model}" for feature in WIDE_MODEL_FEATURE_COLS for model in models]
    ordered_cols = [core.FAMILY_NAME_COL, core.LABEL_TESTED_TRANSITION_COL, *metadata_cols_no_family, SELECTED_MODEL_COL, *model_specific_cols]

    return wide_df.reindex(columns=ordered_cols).sort_values(core.FAMILY_NAME_COL).reset_index(drop=True)


def build_selected_transition_feature_table(long_with_metadata_df: pd.DataFrame, selection_df: pd.DataFrame, transition: str) -> pd.DataFrame:
    """Build one transition block with model selection and constant-run inferred features."""
    constant_df = long_with_metadata_df[
        long_with_metadata_df[core.LABEL_FUNC_TYPE_COL] == core.LABEL_CONSTANT
    ][[core.FAMILY_NAME_COL, *ac.FEATURE_ANALYSIS_INFERRED_COLS]].copy()

    transition_df = selection_df.merge(
        constant_df,
        on=core.FAMILY_NAME_COL,
        how="left",
        validate="one_to_one",
    )

    transition_df = transition_df.reindex(
        columns=[core.FAMILY_NAME_COL, SELECTED_MODEL_COL, *ac.FEATURE_ANALYSIS_INFERRED_COLS]
    )
    transition_df = transition_df.rename(
        columns={
            column: f"{transition}_{column}"
            for column in transition_df.columns
            if column != core.FAMILY_NAME_COL
        }
    )

    return transition_df.sort_values(core.FAMILY_NAME_COL).reset_index(drop=True)


def combine_transition_feature_tables(transition_tables: Mapping[str, pd.DataFrame], metadata_df: pd.DataFrame, transitions: Sequence[str]) -> pd.DataFrame:
    """Combine global family metadata with transition-specific selected-model features."""
    combined_df = metadata_df[[core.FAMILY_NAME_COL, *ac.FEATURE_ANALYSIS_METADATA_COLS]].copy()

    for transition in transitions:
        combined_df = combined_df.merge(
            transition_tables[transition],
            on=core.FAMILY_NAME_COL,
            how="outer",
            validate="one_to_one",
        )

    return combined_df.sort_values(core.FAMILY_NAME_COL).reset_index(drop=True)


def transition_output_dir(output_root: Path, transition: str) -> Path:
    output_dir = output_root / transition
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_transition_outputs(long_df: pd.DataFrame, wide_df: pd.DataFrame, output_root: Path, transition: str) -> List[Path]:
    output_dir = transition_output_dir(output_root, transition)
    long_file = output_dir / f"{transition}_all_models_long.csv"
    wide_file = output_dir / f"{transition}_family_features_wide.csv"

    serialize_parameter_lists(long_df).to_csv(long_file, index=False)
    wide_df.to_csv(wide_file, index=False)

    return [long_file, wide_file]


def save_colored_excel(df: pd.DataFrame, output_file: Path, selected_model_columns: Sequence[str]) -> None:
    """Save Excel output and color selected-model cells using the shared model color map."""
    df.to_excel(output_file, index=False)

    workbook = load_workbook(output_file)
    worksheet = workbook.active

    worksheet.freeze_panes = "B2"
    worksheet.auto_filter.ref = worksheet.dimensions
    header_to_column = {cell.value: cell.column for cell in worksheet[1]}

    for column_name in selected_model_columns:
        column_index = header_to_column.get(column_name)
        if column_index is None:
            continue

        for row_index in range(2, worksheet.max_row + 1):
            cell = worksheet.cell(row=row_index, column=column_index)
            color = ac.MODEL_COLOR_MAP.get(cell.value)

            if color:
                cell.fill = PatternFill(fill_type="solid", fgColor=color.lstrip("#"))

    workbook.save(output_file)


def save_family_feature_matrix(matrix_df: pd.DataFrame, output_root: Path, transitions: Sequence[str]) -> List[Path]:
    output_root.mkdir(parents=True, exist_ok=True)

    csv_file = output_root / "family_feature_matrix.csv"
    excel_file = output_root / "family_feature_matrix.xlsx"

    matrix_df.to_csv(csv_file, index=False)
    save_colored_excel(matrix_df, excel_file, [f"{transition}_{SELECTED_MODEL_COL}" for transition in transitions])

    return [csv_file, excel_file]


def run_transition(transition: str, parsed_results_root: Path, metadata_df: pd.DataFrame, output_root: Path, models: Sequence[str], families: set[str]) -> RunResult:
    long_models_df = build_long_model_table(transition, parsed_results_root, models, families)
    selection_df = build_family_selection_table(long_models_df, models)
    long_with_metadata_df = add_metadata(long_models_df, metadata_df)
    long_final_df = add_selection_to_long(long_with_metadata_df, selection_df)

    wide_df = build_wide_family_table(long_with_metadata_df, selection_df, list(metadata_df.columns), models, transition)
    selected_features_df = build_selected_transition_feature_table(long_with_metadata_df, selection_df, transition)
    output_files = save_transition_outputs(long_final_df, wide_df, output_root, transition)

    return {
        "transition": transition,
        "n_families": int(wide_df[core.FAMILY_NAME_COL].nunique()),
        "n_long_rows": int(len(long_final_df)),
        "selected_model_counts": selection_df[SELECTED_MODEL_COL].value_counts(dropna=False).to_dict(),
        "selected_features_df": selected_features_df,
        "output_files": output_files,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build integrated ChromEvol model/feature tables for baseline or baseline+exponential analyses.")

    parser.add_argument("--model-set", choices=list(MODEL_SETS), default="baseline", help="Model universe used for selection: baseline=constant/linear/ignore; with_exp=constant/linear/exponential/ignore. Default: baseline.")
    parser.add_argument("--transitions", nargs="+", choices=list(TARGET_TRANSITIONS), default=list(TARGET_TRANSITIONS), help="Transitions to build. Default: gain loss dupl.")
    parser.add_argument("--parsed-results-root", type=Path, default=core.PARSED_RESULTS_ROOT, help="Directory containing parsed_results_<configuration>.csv files.")
    parser.add_argument("--families-metadata-file", type=Path, default=core.ALL_FAMILIES_DATA_SUMMARY_FILE, help="Family metadata CSV. Standard metadata columns are validated and legitimate extra columns are preserved.")
    parser.add_argument("--output-root", type=Path, default=core.ANALYSIS_DIR / "integrated_model_features", help="Parent output directory. Results are written under <output-root>/<model-set>/.")
    parser.add_argument("--families_file", type=Path, required=True, help="TXT file containing one family name per line. Only these families are included in the analysis.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = MODEL_SETS[args.model_set]
    output_root = args.output_root / args.model_set

    families = load_families_list(args.families_file)
    if not families:
        raise ValueError(f"No families found in {args.families_file}")

    metadata_df = load_metadata_preserving_extra_columns(args.families_metadata_file)

    metadata_families = set(metadata_df[core.FAMILY_NAME_COL])
    missing_metadata = families - metadata_families
    if missing_metadata:
        print(f"[!] {len(missing_metadata)} requested families are missing from the metadata file:")
        for family in sorted(missing_metadata):
            print(f"    {family}")

    metadata_df = metadata_df[metadata_df[core.FAMILY_NAME_COL].isin(families)].copy()

    transition_tables: Dict[str, pd.DataFrame] = {}
    all_output_files: List[Path] = []

    print(f"[i] Families file: {args.families_file} ({len(families)} families)")
    print(f"[i] Model set: {args.model_set} -> {', '.join(models)}")
    print(f"[i] Parsed results root: {args.parsed_results_root}")
    print(f"[i] Metadata file: {args.families_metadata_file}")
    print(f"[i] Output root: {output_root}")

    for transition in args.transitions:
        result = run_transition(
            transition,
            args.parsed_results_root,
            metadata_df,
            output_root,
            models,
            families,
        )
        transition_tables[transition] = result["selected_features_df"]
        all_output_files.extend(result["output_files"])

        print(f"\n[✓] {transition}: {result['n_families']} families")
        print(f"    long rows: {result['n_long_rows']}")
        print(f"    selected model counts: {result['selected_model_counts']}")

        for output_file in result["output_files"]:
            print(f"    saved: {output_file}")

    matrix_df = combine_transition_feature_tables(
        transition_tables,
        metadata_df,
        args.transitions,
    )
    matrix_files = save_family_feature_matrix(
        matrix_df,
        output_root,
        args.transitions,
    )
    all_output_files.extend(matrix_files)

    print(f"\n[✓] Family feature matrix: {len(matrix_df)} families")

    for output_file in matrix_files:
        print(f"    saved: {output_file}")

    log_run(
        step="analysis",
        script=Path(__file__),
        params={
            "model_set": args.model_set,
            "models": list(models),
            "transitions": list(args.transitions),
            "families_file": str(args.families_file),
            "n_requested_families": len(families),
            "n_output_families": len(matrix_df),
            "parsed_results_root": str(args.parsed_results_root),
            "families_metadata_file": str(args.families_metadata_file),
            "output_root": str(output_root),
        },
        outputs=[str(path) for path in all_output_files],
        description=f"Built integrated family feature tables using the '{args.model_set}' model set.",
        notes=(
            "The families file defines the analysis population. "
            "Family metadata features are included once. "
            "Transition-specific inferred features are taken from the constant-model run, "
            "while selected_model is chosen from the requested model set."
        ),
        log_relative_path=Path("integrated_model_features") / f"{args.model_set}.log",
    )


if __name__ == "__main__":
    main()