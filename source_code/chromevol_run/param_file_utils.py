from typing import Dict, Optional, List, Tuple
import random
import pandas as pd
from ..constants import *

def parse_init_values_str(init_values_str: str, *, transition: str) -> List[float]:
    raw_parts: List[str] = init_values_str.split(",")
    stripped_parts: List[str] = []
    for raw in raw_parts:
        part = raw.strip()
        if part == "":
            raise ValueError(f"start_point_value['{transition}'] contains an empty item: {init_values_str!r}")
        stripped_parts.append(part)

    float_values: List[float] = []
    for part in stripped_parts:
        try:
            value = float(part)
        except ValueError as e:
            raise ValueError(
                f"start_point_value['{transition}'] must be comma-separated floats. "
                f"Failed on item {part!r} in {init_values_str!r}"
            ) from e
        float_values.append(value)

    return float_values


def validate_transition_and_start_points(transition_to_function_dict: Dict[str, str], start_point_value: Optional[Dict[str, str]]) -> None:
    allowed_transitions = set(CE_TRANSITIONS_ORDERED)
    allowed_functions = set(CE_FUNCTIONS)

    spv = start_point_value or {}

    extra_spv_keys = set(spv) - set(transition_to_function_dict)
    if extra_spv_keys:
        raise ValueError(
            f"start_point_value has transitions not found in transition_to_function_dict: "
            f"{sorted(extra_spv_keys)}"
        )

    for transition, function in transition_to_function_dict.items():
        if transition not in allowed_transitions:
            raise ValueError(
                f"transition_to_function_dict has invalid transition {transition!r}. "
                f"Allowed: {CE_TRANSITIONS_ORDERED}"
            )

        if function not in allowed_functions:
            raise ValueError(
                f"transition_to_function_dict[{transition!r}] has invalid function {function!r}. "
                f"Allowed: {sorted(allowed_functions)}"
            )

        if transition in spv:
            raw_start_point_value: str = spv[transition]
            if not isinstance(raw_start_point_value, str):
                raise TypeError(f"start_point_value[{transition!r}] of incorrect format. Got {type(raw_start_point_value).__name__}")

            start_point_floats_list = parse_init_values_str(raw_start_point_value, transition=transition)

            expected_len = INIT_VALUES_COUNT_PER_FUNCTION[function]
            if len(start_point_floats_list) != expected_len:
                raise ValueError(
                    f"start_point_value[{transition!r}] has {len(start_point_floats_list)} values, "
                    f"but function {function!r} requires {expected_len} "
                    f"(INIT_VALUES_COUNT_PER_FUNCTION[{function!r}]). Raw: {raw_start_point_value!r}"
                )

def build_param_file_content(dataFile_path: str, treeFile_path: str, resultsPathDir_path: str,
                             config_df: pd.DataFrame, base_num_init,
                             optimizePointsNum: str = "10,3,1", optimizeIterNum: str = "0,2,5", seed: Optional[int] = None) -> str:

    transition_to_function_dict: Dict[str, str] = config_df.to  dict with keys CE_TRANSITION_COL and values are CE_FUNC_TYPE_COL
    init_transition_to_val_dict = config_df.to dictwith keys CE_TRANSITION_INIT_COL and values are CE_PARAMS_COL
    if seed is None:
        seed = random.randint(1, 8000)

    lines: List[str] = [
        f"_dataFile = {dataFile_path}",
        f"_treeFile = {treeFile_path}",
        f"_resultsPathDir = {resultsPathDir_path}",
    ]

    for transition in transition_to_function_dict:
        lines.append(f"{transition} = {transition_to_function_dict[transition]}")

    i = 1
    for transition_init in init_transition_to_val_dict:
        lines.append(f"{transition_init} = {i};{init_transition_to_val_dict[transition_init]}")
        i = i + 1
    lines.append(f"{CE_BASE_CHROM_NUM_INIT} = {i};{base_num_init}")

    lines.extend([
        "_optimizationMethod = Brent",
        "_baseNumOptimizationMethod = Ranges",
        "_minChrNum = -1",
        f"_optimizePointsNum = {optimizePointsNum}",
        f"_optimizeIterNum = {optimizeIterNum}",
        "_maxParsimonyBound = true",
        "_tolParamOptimization = 0.1",
        f"_seed = {seed}",
        "_heterogeneousModel = false",
        "_backwardPhase = false",
    ])

    return "\n".join(lines) + "\n"

def write_param_file(param_file_path: str, content: str) -> None:
    with open(param_file_path, 'w') as f:
        f.write(content)
    print(f"[✓] Param file written: {param_file_path}")