"""
Prepare a canonical fitted-functions table for generic per-family plotting.

The source-table structure is generic. A JSON parameter-mapping file specifies
which source columns contain the named parameters of each fitted function.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from source_code.analysis.fitted_functions.fitted_functions_utils import FUNCTION_REGISTRY, ParameterMapping, build_fitted_functions_table
from source_code.analysis.parameter_utils import VALID_PARAMETER_SCALES
from source_code.analysis.plot_utils import ensure_dir
from source_code.constants import FAMILY_NAME_COL, LABEL_FUNC_TYPE_COL
from source_code.logger import log_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a canonical fitted-functions table.")
    parser.add_argument("--model-results-file", type=Path, required=True, help="One row per family × fitted function, including AICc.")
    parser.add_argument("--chosen-model-file", type=Path, required=True, help="One row per family containing the chosen function.")
    parser.add_argument("--parameter-file", type=Path, required=True, help="One row per family containing fitted parameter columns.")
    parser.add_argument("--metadata-file", type=Path, required=True, help="Family metadata containing chromosome ranges.")
    parser.add_argument("--parameter-mapping-file", type=Path, required=True, help="JSON mapping: function -> parameter name -> source column.")
    parser.add_argument("--parameter-scale", required=True, choices=VALID_PARAMETER_SCALES)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--log-note", type=str, default="")
    return parser.parse_args()


def load_csv(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)


def load_parameter_mapping(file_path: Path) -> ParameterMapping:
    if not file_path.exists():
        raise FileNotFoundError(f"Parameter-mapping file not found: {file_path}")

    mapping: object = json.loads(file_path.read_text(encoding="utf-8"))

    if not isinstance(mapping, dict) or not all(isinstance(parameters, dict) for parameters in mapping.values()):
        raise ValueError("Parameter mapping must have the form: function -> parameter name -> source column.")

    return mapping


def validate_mapping_coverage(model_results_df: pd.DataFrame, parameter_mapping: ParameterMapping) -> None:
    fitted_functions: set[str] = set(model_results_df[LABEL_FUNC_TYPE_COL].dropna().astype(str))
    missing_mapping: list[str] = sorted(function for function in fitted_functions if function in FUNCTION_REGISTRY and function not in parameter_mapping)

    if missing_mapping:
        raise ValueError(f"Missing parameter mappings for registered fitted functions: {missing_mapping}")


def main() -> None:
    args: argparse.Namespace = parse_args()

    model_results_df: pd.DataFrame = load_csv(args.model_results_file)
    chosen_df: pd.DataFrame = load_csv(args.chosen_model_file)
    parameter_df: pd.DataFrame = load_csv(args.parameter_file)
    metadata_df: pd.DataFrame = load_csv(args.metadata_file)
    parameter_mapping: ParameterMapping = load_parameter_mapping(args.parameter_mapping_file)

    validate_mapping_coverage(model_results_df, parameter_mapping)

    fitted_df: pd.DataFrame = build_fitted_functions_table(
        model_results_df=model_results_df,
        chosen_df=chosen_df,
        parameter_df=parameter_df,
        metadata_df=metadata_df,
        parameter_mapping=parameter_mapping,
        parameter_scale=args.parameter_scale,
    )

    ensure_dir(args.output_file.parent)
    fitted_df.to_csv(args.output_file, index=False)

    n_families: int = fitted_df[FAMILY_NAME_COL].nunique()
    functions: list[str] = sorted(fitted_df[LABEL_FUNC_TYPE_COL].unique().tolist())

    notes: str = f"Prepared canonical fitted-function data for generic plotting. Parameter scale: {args.parameter_scale}."
    if args.log_note:
        notes += f"\nRun note: {args.log_note}"

    log_run(
        step="prepare_fitted_functions_table",
        script=Path(__file__),
        params={
            "model_results_file": str(args.model_results_file),
            "chosen_model_file": str(args.chosen_model_file),
            "parameter_file": str(args.parameter_file),
            "metadata_file": str(args.metadata_file),
            "parameter_mapping_file": str(args.parameter_mapping_file),
            "parameter_scale": args.parameter_scale,
            "number_of_families": n_families,
            "functions": functions,
            "number_of_rows": len(fitted_df),
        },
        outputs=[args.output_file.as_posix()],
        description="Prepared a canonical fitted-functions table.",
        notes=notes,
        log_relative_path=Path("fitted_functions") / f"prepare_{args.output_file.stem}.log",
    )

    print("[✓] Fitted-functions table prepared.")
    print(f"Families: {n_families}")
    print(f"Functions: {functions}")
    print(f"Rows: {len(fitted_df)}")
    print(f"Parameter scale: {args.parameter_scale}")
    print(args.output_file)


if __name__ == "__main__":
    main()