"""
Shared utilities for fitted-function preparation and evaluation.

The fitted-functions table contains one row per family × fitted function.
Parameter source columns are supplied by the caller, so the code is independent
of a specific analysis and supports any number of functions and parameters.
"""

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from source_code.analysis.analysis_constants import CHOSEN_FUNCTION_LABEL_COL
from source_code.analysis.parameter_utils import VALID_PARAMETER_SCALES
from source_code.analysis.plot_utils import compute_constant_rate, compute_exp_rate, compute_linear_rate
from source_code.constants import AICC_COL, FAMILY_NAME_COL, LABEL_CONSTANT, LABEL_EXP, LABEL_FUNC_TYPE_COL, LABEL_LINEAR, LABEL_TESTED_TRANSITION_COL, MAX_CHROM_COL, MIN_CHROM_COL


FITTED_PARAMETERS_JSON_COL: str = "fitted_parameters_json"
PARAMETER_SCALE_COL: str = "parameter_scale"

FITTED_FUNCTIONS_TABLE_COLS: tuple[str, ...] = (
    FAMILY_NAME_COL,
    LABEL_TESTED_TRANSITION_COL,
    LABEL_FUNC_TYPE_COL,
    CHOSEN_FUNCTION_LABEL_COL,
    AICC_COL,
    MIN_CHROM_COL,
    MAX_CHROM_COL,
    FITTED_PARAMETERS_JSON_COL,
    PARAMETER_SCALE_COL,
)


RateFunction = Callable[..., float | np.ndarray]
ParameterMapping = Mapping[str, Mapping[str, str]]


@dataclass(frozen=True)
class FunctionSpec:
    rate_function: RateFunction
    parameter_names: tuple[str, ...]


FUNCTION_REGISTRY: dict[str, FunctionSpec] = {
    LABEL_CONSTANT: FunctionSpec(compute_constant_rate, ("value",)),
    LABEL_LINEAR: FunctionSpec(compute_linear_rate, ("p1", "p2")),
    LABEL_EXP: FunctionSpec(compute_exp_rate, ("p1", "p2")),
}


def require_columns(df: pd.DataFrame, columns: list[str], table_name: str) -> None:
    missing: list[str] = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {table_name}: {missing}")


def build_parameters_json(row: pd.Series, parameter_mapping: ParameterMapping) -> str:
    function_label: str = str(row[LABEL_FUNC_TYPE_COL])
    mapping: Mapping[str, str] = parameter_mapping.get(function_label, {})

    parameters: dict[str, float] = {}

    for parameter_name, source_col in mapping.items():
        value: object = pd.to_numeric(row[source_col], errors="coerce")

        if pd.isna(value):
            raise ValueError(f"Missing parameter '{parameter_name}' for {row[FAMILY_NAME_COL]} ({function_label}).")

        parameters[parameter_name] = float(value)

    return json.dumps(parameters, separators=(",", ":"))


def build_fitted_functions_table(
    model_results_df: pd.DataFrame,
    chosen_df: pd.DataFrame,
    parameter_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    parameter_mapping: ParameterMapping,
    parameter_scale: str,
) -> pd.DataFrame:
    if parameter_scale not in VALID_PARAMETER_SCALES:
        raise ValueError(f"Invalid parameter scale: {parameter_scale}")

    family_transition_cols: list[str] = [FAMILY_NAME_COL, LABEL_TESTED_TRANSITION_COL]
    model_cols: list[str] = family_transition_cols + [LABEL_FUNC_TYPE_COL, AICC_COL]
    parameter_cols: list[str] = sorted({col for mapping in parameter_mapping.values() for col in mapping.values()})

    require_columns(model_results_df, model_cols, "model-results table")
    require_columns(chosen_df, family_transition_cols + [CHOSEN_FUNCTION_LABEL_COL], "chosen-model table")
    require_columns(parameter_df, family_transition_cols + parameter_cols, "parameter table")
    require_columns(metadata_df, [FAMILY_NAME_COL, MIN_CHROM_COL, MAX_CHROM_COL], "metadata table")

    df: pd.DataFrame = model_results_df[model_cols].copy()
    df = df.merge(chosen_df[family_transition_cols + [CHOSEN_FUNCTION_LABEL_COL]], on=family_transition_cols, how="left", validate="many_to_one")
    df = df.merge(parameter_df[family_transition_cols + parameter_cols], on=family_transition_cols, how="left", validate="many_to_one")
    df = df.merge(metadata_df[[FAMILY_NAME_COL, MIN_CHROM_COL, MAX_CHROM_COL]], on=FAMILY_NAME_COL, how="left", validate="many_to_one")

    if df.duplicated([FAMILY_NAME_COL, LABEL_TESTED_TRANSITION_COL, LABEL_FUNC_TYPE_COL]).any():
        raise ValueError("Duplicate family × transition × function rows found.")

    required_values: list[str] = [CHOSEN_FUNCTION_LABEL_COL, AICC_COL, MIN_CHROM_COL, MAX_CHROM_COL]
    if df[required_values].isna().any(axis=None):
        raise ValueError("Missing chosen-model, AICc, or chromosome-range values.")

    df[FITTED_PARAMETERS_JSON_COL] = df.apply(lambda row: build_parameters_json(row, parameter_mapping), axis=1)
    df[PARAMETER_SCALE_COL] = parameter_scale

    return df[list(FITTED_FUNCTIONS_TABLE_COLS)].sort_values([FAMILY_NAME_COL, LABEL_FUNC_TYPE_COL]).reset_index(drop=True)


def parse_parameters(value: object) -> dict[str, float]:
    parameters: dict = value if isinstance(value, dict) else json.loads(str(value))
    return {str(name): float(parameter_value) for name, parameter_value in parameters.items()}


def evaluate_fitted_function(function_label: str, chromosome_values: float | np.ndarray, parameters: Mapping[str, float]) -> float | np.ndarray:
    if function_label not in FUNCTION_REGISTRY:
        raise KeyError(f"No registered rate function for '{function_label}'.")

    spec: FunctionSpec = FUNCTION_REGISTRY[function_label]
    missing: list[str] = [name for name in spec.parameter_names if name not in parameters]

    if missing:
        raise ValueError(f"Missing parameters for '{function_label}': {missing}")

    return spec.rate_function(chromosome_values, *[parameters[name] for name in spec.parameter_names])


def is_plottable_function(function_label: str) -> bool:
    return function_label in FUNCTION_REGISTRY