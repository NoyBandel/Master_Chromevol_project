import json
from typing import Any, Dict, List, Set
import pandas as pd
from source_code.constants import *

def build_transition_cell(label_func_type: str, init_values: List[float]) -> str:
    cell: Dict[str, Any] = {
        LABEL_FUNC_TYPE_COL: label_func_type,
        INIT_VALUES_COL: init_values,
    }
    return json.dumps(cell, sort_keys=True)

def parse_default_init_values_for_function(function_label: str) -> List[float]:
    ce_function: str = LABEL_FUNC_TO_CE_FUNC[function_label]
    default_value_str: str = DEFAULT_INIT_VALUES[ce_function]

    if default_value_str == "":
        return []
    return [float(v.strip()) for v in default_value_str.split(",")]


def validate_init_values_count(function_label: str, init_values: List[float]) -> None:
    ce_function: str = LABEL_FUNC_TO_CE_FUNC[function_label]
    expected: int = INIT_VALUES_COUNT_PER_FUNCTION[ce_function]

    if len(init_values) != expected:
        raise ValueError(
            f"{function_label} expects {expected} parameters but got {len(init_values)}: {init_values}"
        )

def read_families_file(families_file: Path) -> List[str]:
    with families_file.open("r", encoding="utf-8") as f:
        families: List[str] = [line.strip() for line in f if line.strip()]

    if len(families) == 0:
        raise ValueError(f"Families file is empty: {families_file}")

    return families

def validate_families_in_input_df(input_df: pd.DataFrame, families: List[str]) -> None:
    txt_families: Set[str] = set(families)
    csv_families: Set[str] = set(input_df[FAMILY_NAME_COL].astype(str))

    missing_in_csv: List[str] = sorted(txt_families - csv_families)
    extra_in_csv: List[str] = sorted(csv_families - txt_families)

    if missing_in_csv or extra_in_csv:
        msg_lines = ["Families mismatch between families file and input parsed results CSV:"]
        if missing_in_csv:
            msg_lines.append(f"  Families in TXT but not in CSV: {missing_in_csv}")
        if extra_in_csv:
            msg_lines.append(f"  Families in CSV but not in TXT: {extra_in_csv}")

        raise ValueError("\n".join(msg_lines))

def build_all_constant_transition_to_function_dict() -> Dict[str, str]:
    return {
        transition_label: LABEL_CONSTANT
        for transition_label in LABEL_TRANSITIONS_ORDERED
    }

def build_M0_inferred_init_values_dict(parsed_row: pd.Series) -> Dict[str, List[float]]:
    all_params_dict: Dict[str, List[float]] = json.loads(parsed_row[ALL_PARAMS_DICT_STR_COL])
    return {
        transition_label: all_params_dict[transition_label]
        for transition_label in LABEL_TRANSITIONS_ORDERED
    }

def build_default_constant_init_values_dict() -> Dict[str, List[float]]:
    default_vals = parse_default_init_values_for_function(LABEL_CONSTANT)
    return {
        transition_label: default_vals
        for transition_label in LABEL_TRANSITIONS_ORDERED
    }

def build_model_config_family_row(parsed_row: pd.Series, transition_to_function_dict: Dict[str, str], transition_to_init_values_dict: Dict[str, List[float]]) -> Dict[str, Any]:
    config_row: Dict[str, Any] = {
        FAMILY_NAME_COL: parsed_row[FAMILY_NAME_COL],
        BASE_CHROM_NUM_COL: parsed_row[BASE_CHROM_NUM_COL],
    }

    for transition_label in LABEL_TRANSITIONS_ORDERED:
        function_label = transition_to_function_dict[transition_label]
        init_values = transition_to_init_values_dict[transition_label]
        validate_init_values_count(function_label, init_values)
        config_row[transition_label] = build_transition_cell(function_label, init_values)

    return config_row

