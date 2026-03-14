# -------- M0_all_const --------
# All transition functions are constant, and all transition init values are set to the default values.
#
# -------- M1_{function}_{tested_transition} --------
# All transition functions are constant and are initialized using the M0 inferred values,
# except for {tested_transition}, whose function is set to {function} and whose init values are determined by a function-specific initialization rule.
#
# -------- M2_{function}_{tested_transition} --------
# All transition functions other than {tested_transition} are set to their best previously selected function based on comparison among M0 and the relevant M1 models,
# and are initialized using the corresponding inferred values.
# The {tested_transition} function is set to {function} and its init values are determined by a function-specific initialization rule.

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from source_code.constants import *
from source_code.prepare_data_and_config_for_runs.prepare_data_for_runs_utils import *


# -------------------------------------------------------
# M0_all_const
# -------------------------------------------------------

def build_M0_all_const_configuration_file(input_parsed_results_file: Path) -> pd.DataFrame:
    input_df = pd.read_csv(input_parsed_results_file)

    rows: List[Dict[str, Any]] = []
    transition_to_function_dict = build_all_constant_transition_to_function_dict()
    transition_to_init_values_dict = build_default_constant_init_values_dict()

    for _, family_row in input_df.iterrows():
        row = build_model_config_family_row(family_row, transition_to_function_dict, transition_to_init_values_dict)
        rows.append(row)

    config_df = pd.DataFrame(rows)
    config_df = config_df.reindex(columns=MODEL_CONFIG_COLS)

    return config_df

# -------------------------------------------------------
# M1 family
# -------------------------------------------------------
def build_tested_transition_init_values_from_M0_intercept_and_defaults(baseline_init_values: List[float], tested_function_label: str) -> List[float]:
    default_vals = parse_default_init_values_for_function(tested_function_label)
    if len(default_vals) == 0:
        return []
    return [baseline_init_values[0]] + default_vals[1:]

def build_M1_configuration_file(input_parsed_results_file: Path, families_file: Path, tested_transition_label: str, tested_function_label: str) -> pd.DataFrame:
    input_df: pd.DataFrame = pd.read_csv(input_parsed_results_file)

    families: List[str] = read_families_file(families_file)
    validate_families_in_input_df(input_df, families)

    rows: List[Dict[str, Any]] = []
    for _, family_row in input_df.iterrows():
        transition_to_function_dict: Dict[str, str] = build_all_constant_transition_to_function_dict()
        transition_to_function_dict[tested_transition_label] = tested_function_label
        transition_to_init_values_dict: Dict[str, list[float]] = build_M0_inferred_init_values_dict(family_row)

        baseline_vals = transition_to_init_values_dict[tested_transition_label]
        transition_to_init_values_dict[tested_transition_label] = build_tested_transition_init_values_from_M0_intercept_and_defaults(baseline_vals, tested_function_label)

        row = build_model_config_family_row(family_row, transition_to_function_dict, transition_to_init_values_dict)
        rows.append(row)

    config_df = pd.DataFrame(rows)
    config_df = config_df.reindex(columns=MODEL_CONFIG_COLS)

    return config_df

# -------------------------------------------------------
# M2 placeholder
# -------------------------------------------------------
def build_M2_configuration_file(*args, **kwargs):
    raise NotImplementedError(
        "M2 configuration generation requires the best-function analysis step."
    )