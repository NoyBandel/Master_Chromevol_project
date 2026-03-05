import argparse
from pathlib import Path
from typing import Optional

from source_code.constants import (
    CHROMEVOL_RAW_RESULTS_ROOT,
    CONDA_ENV,
    CONDA_EXPORT,
    PROJECT_ROOT,
    PREPARE_AND_SUBMIT_CHROMEVOL_PY_FILE
)
from source_code.logger import log_run
from source_code.chromevol_run.job_file_utils import general_job_format, general_submit_job


def parse_args() -> tuple[Path, str, Optional[str], Optional[Path]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--families_file", type=Path, required=True, help="Path to txt file where each row is a family name.")
    parser.add_argument("--configuration_type", required=True, help="Configuration type.")
    parser.add_argument("--tested_transition", default=None, help="Optional tested transition (required unless configuration_type is all-const).")
    parser.add_argument("--config_data_file", type=Path, default=None,help="Optional path to config data file.")
    args = parser.parse_args()
    return args.families_file, args.configuration_type, args.tested_transition, args.config_data_file

def build_cmd(families_file: Path, configuration_type: str, tested_transition: Optional[str], config_data_file: Optional[Path]) -> str:
    cmd = ""
    cmd += f'export PYTHONPATH="{PROJECT_ROOT}:$PYTHONPATH"\n'
    cmd += f'cd "{PROJECT_ROOT}"\n'
    cmd += f"{CONDA_ENV}\n"
    cmd += f"{CONDA_EXPORT}"
    cmd += f"python {PREPARE_AND_SUBMIT_CHROMEVOL_PY_FILE}"
    cmd += f' --configuration_type "{configuration_type}"'
    cmd += f' --families_file "{families_file.resolve()}"'
    if tested_transition:
        cmd += f' --tested_transition "{tested_transition}"'
    if config_data_file:
        cmd += f' --config_data_file "{config_data_file.resolve()}"'
    return cmd


def main() -> None:
    families_file, configuration_type, tested_transition, config_data_file = parse_args()

    job_folder_path = CHROMEVOL_RAW_RESULTS_ROOT / configuration_type
    if tested_transition:
        job_folder_path = job_folder_path / tested_transition
    job_folder_path.mkdir(parents=True, exist_ok=True)

    cmd = build_cmd(families_file, configuration_type, tested_transition, config_data_file)

    job_name = f"main_{configuration_type}" + (f"_{tested_transition}" if tested_transition else "")
    job_name = job_name.replace("/", "_")[:80]

    job_path: Path = job_folder_path / f"{job_name}.sh"
    job_content: str = general_job_format(str(job_folder_path), job_name, cmd, workdir=PROJECT_ROOT)
    job_id = general_submit_job(job_path, job_name, job_content)

    log_run(
        step="chromevol_run",
        script=Path(__file__),
        params={
            "configuration_type": configuration_type,
            "tested_transition": tested_transition or "",
            "families_file": str(families_file),
            "config_data_file": (str(config_data_file) if config_data_file else ""),
            "job_name": job_name,
            "job_id": job_id,
        },
        outputs=[str(job_path)],
        description=f"Created and submitted main {configuration_type} Chromevol job.",
        step_extension=f"_{configuration_type}",
    )


if __name__ == "__main__":
    main()