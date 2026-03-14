import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from source_code.chromevol_run.create_output_folders import create_output_folders
from source_code.chromevol_run.job_file_utils import build_job_script_for_chromevol_run, general_submit_job
from source_code.chromevol_run.param_file_utils import build_param_file_content, write_param_file
from source_code.constants import *
from source_code.logger import log_run

def read_families_file(families_file: Path) -> List[str]:
    with families_file.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def load_model_config_file(config_data_file: Path) -> pd.DataFrame:
    config_df = pd.read_csv(config_data_file, sep=None, engine="python")
    missing_cols = [col for col in MODEL_CONFIG_COLS if col not in config_df.columns]
    if missing_cols:
        raise ValueError(f"Config data file is missing required columns: {missing_cols}")
    return config_df

def parse_transition_config_cell(cell_value: Any, transition_label: str) -> Dict[str, Any]:
    if pd.isna(cell_value):
        raise ValueError(f"Missing value in transition column '{transition_label}'")

    if isinstance(cell_value, dict):
        parsed = cell_value
    elif isinstance(cell_value, str):
        try:
            parsed = json.loads(cell_value)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed parsing JSON for transition '{transition_label}': {cell_value}"
            ) from e
    else:
        raise TypeError(
            f"Transition column '{transition_label}' must contain JSON string/dict, "
            f"got {type(cell_value).__name__}"
        )

    if "func_type_label" not in parsed:
        raise ValueError(f"Missing 'func_type_label' in transition '{transition_label}' config")
    if "init_values" not in parsed:
        raise ValueError(f"Missing 'init_values' in transition '{transition_label}' config")

    func_type_label = parsed["func_type_label"]
    init_values = parsed["init_values"]

    if func_type_label not in LABEL_FUNC_TO_CE_FUNC:
        raise ValueError(
            f"Invalid func_type_label '{func_type_label}' in transition '{transition_label}'"
        )

    if not isinstance(init_values, list):
        raise TypeError(
            f"'init_values' for transition '{transition_label}' must be a list, got {type(init_values).__name__}"
        )

    ce_func = LABEL_FUNC_TO_CE_FUNC[func_type_label]
    expected_len = INIT_VALUES_COUNT_PER_FUNCTION[ce_func]
    if len(init_values) != expected_len:
        raise ValueError(
            f"Transition '{transition_label}' with func_type_label='{func_type_label}' "
            f"requires {expected_len} init values, got {len(init_values)}"
        )

    return parsed


def build_family_config_df_from_row(family_row: pd.Series) -> Tuple[pd.DataFrame, int]:
    rows: List[Dict[str, str]] = []

    for transition_label in LABEL_TRANSITIONS_ORDERED:
        parsed = parse_transition_config_cell(family_row[transition_label], transition_label)
        func_type_label = parsed[LABEL_FUNC_TYPE_COL]
        init_values = parsed[INIT_VALUES_COL]

        ce_transition = TRANSITIONS_LABEL_TO_CE[transition_label]
        ce_function = LABEL_FUNC_TO_CE_FUNC[func_type_label]
        ce_transition_init = TRANSITIONS_LABEL_TO_CE_INIT[transition_label]
        chromevol_parameters = ",".join(str(v) for v in init_values)

        rows.append(
            {
                CE_TRANSITION_COL: ce_transition,
                CE_FUNC_TYPE_COL: ce_function,
                CE_TRANSITION_INIT_COL: ce_transition_init,
                CE_PARAMS_COL: chromevol_parameters,
            }
        )

    family_config_df = pd.DataFrame(rows)
    base_num_init = int(family_row[BASE_CHROM_NUM_COL])

    return family_config_df, base_num_init

def prepare_and_submit_one_family_job(family_name: str, family_main_run_dir: Path, results_dir_path: str, config_df: pd.DataFrame, base_num_init: int, configuration_type: str, tested_transition: Optional[str], tested_function: Optional[str]) -> None:
    family_data_input_dir: Path = DATABASE_INPUT_DIR / family_name
    data_file_path = str(family_data_input_dir / CHROM_COUNTS_FILE_NAME)
    tree_file_path = str(family_data_input_dir / TREE_FILE_NAME)

    param_file_content = build_param_file_content(data_file_path, tree_file_path, results_dir_path, config_df, base_num_init)
    param_file_path: Path = family_main_run_dir / PARAM_FILE
    write_param_file(str(param_file_path), param_file_content)

    job_name = f"{family_name}_{configuration_type}"
    if tested_function:
        job_name += f"_{tested_function}"
    if tested_transition:
        job_name += f"_{tested_transition}"

    job_file_path: Path = family_main_run_dir / f"{job_name}.sh"
    job_file_content = build_job_script_for_chromevol_run(family_main_run_dir, job_name, param_file_path=str(param_file_path))
    general_submit_job(job_file_path, job_name, job_file_content)

def prepare_and_submit_chromevol_jobs_for_families(families_file: Path, model_type: str, tested_transition: Optional[str] = None, tested_function: Optional[str] = None, config_data_file: Optional[Path] = None) -> None:
    families_list: List[str] = read_families_file(families_file)
    results_dirs_dict: dict[str, str] = create_output_folders(model_type, families_list, tested_transition, tested_function)
    print("Output folders created\n")

    model_config_df: Optional[pd.DataFrame] = None
    if model_type != M0_LABEL:
        model_config_df = load_model_config_file(config_data_file)

    for family_name in families_list:
        print(f"\n{family_name}")
        family_main_run_dir: Path = Path(results_dirs_dict[family_name]).parent

        if model_type == M0_LABEL:
            family_config_df = pd.DataFrame({
                LABEL_TRANSITION_COL: LABEL_TRANSITIONS_ORDERED,
                LABEL_FUNC_TYPE_COL: [LABEL_CONSTANT] * len(LABEL_TRANSITIONS_ORDERED),
                CE_TRANSITION_COL: CE_TRANSITIONS_ORDERED,
                CE_FUNC_TYPE_COL: [CE_CONSTANT] * len(CE_TRANSITIONS_ORDERED),
                CE_TRANSITION_INIT_COL: CE_TRANSITIONS_INIT_ORDERED,
                CE_PARAMS_COL: [DEFAULT_INIT_VALUES[CE_CONSTANT]] * len(CE_TRANSITIONS_ORDERED),
            })
            base_num_init = DEFAULT_INIT_VALUES[CE_BASE_CHROM_NUM_INIT]

        else:
            matched_rows = model_config_df[model_config_df[FAMILY_NAME_COL] == family_name]
            family_row = matched_rows.iloc[0]
            family_config_df, base_num_init = build_family_config_df_from_row(family_row)

        prepare_and_submit_one_family_job(family_name, family_main_run_dir, results_dirs_dict[family_name], family_config_df, base_num_init, model_type, tested_transition, tested_function)
        time.sleep(3)

    print("\n[✓][✓][✓] Finished submission")

def parse_args() -> tuple[Path, str, Optional[str], Optional[str], Optional[Path]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--families_file", type=Path, required=True, help="Path to txt file where each row is a family name.")
    parser.add_argument("--model_type", required=True, help="Model type.")
    parser.add_argument("--tested_transition", default=None,help="Optional tested transition (required unless configuration_type is M0_all_const).")
    parser.add_argument("--tested_function", default=None, help="Optional tested function (required for M1/M2 config-driven runs).")
    parser.add_argument("--config_data_file", type=Path, default=None, help="Optional path to config data file.")
    args = parser.parse_args()

    return args.families_file, args.model_type, args.tested_transition, args.tested_function, args.config_data_file

def main() -> None:
    families_file, model_type, tested_transition, tested_function, config_data_file = parse_args()
    prepare_and_submit_chromevol_jobs_for_families(families_file, model_type, tested_transition, tested_function, config_data_file)
    families_list = read_families_file(families_file)

    description = f"Starting Chromevol run for configuration '{model_type}'"
    params = {
        "model_type": model_type,
        "families_file": str(families_file),
        "config_data_file": str(config_data_file) if config_data_file else "",
        "n_families": len(families_list),
    }
    if tested_transition and tested_function:
        description = f"{description}, {tested_transition}, {tested_function}"
        params["tested_transition"] = tested_transition
        params["tested_function"] = tested_function

    configuration_name = model_type if model_type == M0_LABEL else f"{model_type}_{tested_function}_{tested_transition}"
    log_relative_path = Path(model_type) / f"{configuration_name}.log"

    log_run(
        step="chromevol_run",
        script=Path(__file__),
        params=params,
        outputs=[],
        description=description,
        notes="Preparing and submitting ChromEvol jobs for families.",
        log_relative_path=log_relative_path,
    )


if __name__ == "__main__":
    main()