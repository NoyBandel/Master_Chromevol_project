from typing import Dict, List, Optional
import pandas as pd
from ..constants import *

def create_output_folders(configuration_type: str, families_list: List[str], tested_transition: Optional[str] = None, tested_function: Optional[str] = None) -> Dict[str, str]:
    if configuration_type != LABEL_ALL_CONST_CONFIG:
        if tested_transition is None or not isinstance(tested_transition, str):
            raise ValueError(f"When configuration_type != '{LABEL_ALL_CONST_CONFIG}', tested_transition must be a non-empty str.")
        if tested_transition not in LABEL_TRANSITIONS_ORDERED:
            raise ValueError(f"tested_transition must be one of LABEL_TRANSITIONS_LST, got: {tested_transition}")

        if tested_function is None or not isinstance(tested_function, str):
            raise ValueError(f"When configuration_type != '{LABEL_ALL_CONST_CONFIG}', tested_function must be a non-empty str.")
        if tested_function not in LABEL_FUNCTIONS_LST:
            raise ValueError(f"tested_function must be one of LABEL_FUNCTIONS_LST, got: {tested_function}")

    if not RAW_RESULTS_DIR.exists():
        raise FileNotFoundError(f"RAW_RESULTS_DIR does not exist: {RAW_RESULTS_DIR}")

    # Build base folder: RAW_RESULTS_DIR / configuration_type
    config_dir = RAW_RESULTS_DIR / configuration_type
    config_dir.mkdir(parents=True, exist_ok=True)
    results_dirs_dict: Dict[str, str] = {}

    for family in families_list:
        family_dir: Path = config_dir / family
        family_dir.mkdir(parents=True, exist_ok=True)

        if configuration_type == LABEL_ALL_CONST_CONFIG:
            # RAW_RESULTS_DIR / configuration_type / family / Results
            results_dir = family_dir / "Results"
            results_dir.mkdir(parents=True, exist_ok=True)
            results_dirs_dict[family] = str(results_dir)
            continue

        # RAW_RESULTS_DIR / configuration_type / family / tested_transition / tested_function / Results
        transition_dir = family_dir / tested_transition
        function_dir = transition_dir / tested_function
        results_dir = function_dir / "Results"
        results_dir.mkdir(parents=True, exist_ok=True)
        results_dirs_dict[family] = str(results_dir)

    return results_dirs_dict

