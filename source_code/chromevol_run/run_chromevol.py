import argparse
from pathlib import Path
from typing import Optional

from source_code.constants import CHROMEVOL_RAW_RESULTS_ROOT, CONDA_ENV, CONDA_EXPORT, MODEL_TYPES, LABEL_FUNCTIONS_LST, \
    LABEL_TRANSITIONS_ORDERED, M0_LABEL, PREPARE_AND_SUBMIT_CHROMEVOL_PY_FILE, PROJECT_ROOT, M1_LABEL
from source_code.logger import log_run
from source_code.chromevol_run.job_file_utils import general_job_format, general_submit_job

def parse_args() -> tuple[Path, str, Optional[str], Optional[str], Optional[Path], Optional[str], Optional[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--families_file", type=Path, required=True, help="Path to txt file where each row is a family name.")
    parser.add_argument("--model_type", required=True, help="Model type.")
    parser.add_argument("--tested_transition", default=None, help="Optional tested transition (required unless model_type is M0_all_const).")
    parser.add_argument("--tested_function", default=None, help="Optional tested function (required unless model_type is M0_all_const).")
    parser.add_argument("--config_data_file", type=Path, default=None, help="Optional path to config data file.")
    parser.add_argument("--job_suffix", type=str, default=None, help="Optional suffix for the main job name (useful for reruns).")
    parser.add_argument("--notes", type=str, default=None, help="Optional notes for this run (e.g. why rerunning, manual fixes, etc.)")
    args = parser.parse_args()
    return args.families_file, args.model_type, args.tested_transition, args.tested_function, args.config_data_file, args.job_suffix, args.notes

def validate_args(model_type: str, tested_transition: Optional[str], tested_function: Optional[str], config_data_file: Optional[Path]) -> None:
    if model_type not in MODEL_TYPES:
        raise ValueError(f"model_type must be one of {MODEL_TYPES}, got: {model_type}")

    if model_type == M0_LABEL:
        return

    if tested_transition is None:
        raise ValueError(f"{model_type} requires --tested_transition")
    if tested_transition not in LABEL_TRANSITIONS_ORDERED:
        raise ValueError(f"tested_transition must be one of {LABEL_TRANSITIONS_ORDERED}, got: {tested_transition}")

    if tested_function is None:
        raise ValueError(f"{model_type} requires --tested_function")
    if tested_function not in LABEL_FUNCTIONS_LST:
        raise ValueError(f"tested_function must be one of {LABEL_FUNCTIONS_LST}, got: {tested_function}")

    if config_data_file is None:
        raise ValueError(f"{model_type} requires --config_data_file")

def build_configuration_name(model_type: str, tested_function: Optional[str], tested_transition: Optional[str]) -> str:
    if model_type == M0_LABEL:
        return model_type
    return f"{model_type}_{tested_function}_{tested_transition}"

def build_job_folder_path(model_type: str, tested_transition: Optional[str], tested_function: Optional[str]) -> Path:
    if model_type == M0_LABEL:
        return CHROMEVOL_RAW_RESULTS_ROOT / model_type
    return CHROMEVOL_RAW_RESULTS_ROOT / model_type / tested_function / tested_transition

def build_cmd(families_file: Path, model_type: str, tested_transition: Optional[str], tested_function: Optional[str], config_data_file: Optional[Path]) -> str:
    cmd = ""
    cmd += f'export PYTHONPATH="{PROJECT_ROOT}:$PYTHONPATH"\n'
    cmd += f'cd "{PROJECT_ROOT}"\n'
    cmd += f"{CONDA_ENV}\n"
    cmd += f"{CONDA_EXPORT}\n"

    cmd += f"python {PREPARE_AND_SUBMIT_CHROMEVOL_PY_FILE} \\\n"
    cmd += f' --model_type "{model_type}" \\\n'
    cmd += f' --families_file "{families_file.resolve()}"'

    if tested_transition:
        cmd += f' \\\n --tested_transition "{tested_transition}"'

    if tested_function:
        cmd += f' \\\n --tested_function "{tested_function}"'

    if config_data_file:
        cmd += f' \\\n --config_data_file "{config_data_file.resolve()}"'

    cmd += "\n"
    return cmd

def build_log_relative_path(model_type: str, configuration_name: str) -> Path:
    return Path(model_type) / f"{configuration_name}.log"

def main() -> None:
    families_file, model_type, tested_transition, tested_function, config_data_file, job_suffix, notes = parse_args()
    validate_args(model_type, tested_transition, tested_function, config_data_file)

    configuration_name: str = build_configuration_name(model_type, tested_function, tested_transition)
    job_folder_path: Path = build_job_folder_path(model_type, tested_transition, tested_function)
    job_folder_path.mkdir(parents=True, exist_ok=True)

    cmd: str = build_cmd(families_file, model_type, tested_transition, tested_function, config_data_file)

    job_name = f"main_{configuration_name}"
    if job_suffix:
        job_name += f"_{job_suffix}"
    job_name = job_name.replace("/", "_")[:80]

    job_path: Path = job_folder_path / f"{job_name}.sh"
    job_content: str = general_job_format(str(job_folder_path), job_name, cmd, workdir=PROJECT_ROOT)
    job_id = general_submit_job(job_path, job_name, job_content)

    params = {"model_type": model_type, "configuration_name": configuration_name, "families_file": str(families_file), "job_name": job_name, "job_id": job_id}
    if tested_transition:
        params["tested_transition"] = tested_transition
    if tested_function:
        params["tested_function"] = tested_function
    if config_data_file:
        params["config_data_file"] = str(config_data_file)
    if job_suffix:
        params["job_suffix"] = job_suffix

    log_relative_path: Path = build_log_relative_path(model_type, configuration_name)

    log_run(
        step="chromevol_run",
        script=Path(__file__),
        params=params,
        outputs=[str(job_path)],
        description=f"Created and submitted main {configuration_name} ChromEvol job.",
        notes=notes,
        log_relative_path=log_relative_path,
    )


if __name__ == "__main__":
    main()