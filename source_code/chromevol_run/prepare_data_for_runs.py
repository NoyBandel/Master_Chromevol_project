from typing import List, Optional
from source_code.constants import *

def create_output_folders(configuration_type: str, families_list: List[str], tested_transition: Optional[str] = None, tested_function: Optional[str] = None) -> Dict[str, str]:
    if configuration_type != M0_LABEL:
        if tested_transition is None or not isinstance(tested_transition, str):
            raise ValueError(f"When configuration_type != '{M0_LABEL}', tested_transition must be a non-empty str.")
        if tested_transition not in LABEL_TRANSITIONS_ORDERED:
            raise ValueError(f"tested_transition must be one of LABEL_TRANSITIONS_LST, got: {tested_transition}")

        if tested_function is None or not isinstance(tested_function, str):
            raise ValueError(f"When configuration_type != '{M0_LABEL}', tested_function must be a non-empty str.")
        if tested_function not in LABEL_FUNCTIONS_LST:
            raise ValueError(f"tested_function must be one of LABEL_FUNCTIONS_LST, got: {tested_function}")

    main_config_dir = CHROMEVOL_RAW_RESULTS_ROOT / configuration_type
    main_config_dir.mkdir(parents=True, exist_ok=True)
    config_dir = main_config_dir
    if tested_transition:
        transition_config_dir = main_config_dir / tested_transition
        config_dir = transition_config_dir

    results_dirs_dict: Dict[str, str] = {}

    for family in families_list:
        family_dir: Path = config_dir / family
        family_dir.mkdir(parents=True, exist_ok=True)
        results_dir = family_dir / "Results"
        results_dir.mkdir(parents=True, exist_ok=True)
        results_dirs_dict[family] = str(results_dir)

    return results_dirs_dict

