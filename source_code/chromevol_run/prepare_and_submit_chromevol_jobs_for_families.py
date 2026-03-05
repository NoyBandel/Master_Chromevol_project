import time
import argparse
from source_code.chromevol_run.prepare_data_for_runs import *
from source_code.chromevol_run.job_file_utils import *
from source_code.chromevol_run.param_file_utils import *
from source_code.chromevol_run.create_config_df  import *
from source_code.logger  import log_run

def prepare_and_submit_chromevol_jobs_for_families(families_file: Path, configuration_type: str, tested_transition: Optional[str] = None, config_data_file: Optional[Path] = None):
    with families_file.open("r", encoding="utf-8") as f:
        families_list: List[str] = [line.strip() for line in f if line.strip()]

    results_dirs_dict: Dict[str, str] = create_output_folders(configuration_type, families_list, tested_transition)
    print("Output folders created\n")

    for family_name in families_list:
        print(f"\n{family_name}")
        family_main_run_dir: Path = Path(results_dirs_dict[family_name]).parent

        family_data_input_dir: Path = DATABASE_INPUT_DIR / family_name
        dataFile_path = str(family_data_input_dir / CHROM_COUNTS_FILE_NAME)
        treeFile_path = str(family_data_input_dir / TREE_FILE_NAME)
        resultsPathDir_path = results_dirs_dict[family_name]

        config_df, base_num_init = create_configuration(configuration_type, tested_transition, config_data_file)
        param_file_content: str = build_param_file_content(dataFile_path, treeFile_path, resultsPathDir_path, config_df, base_num_init)

        param_file_path = family_main_run_dir/PARAM_FILE
        write_param_file(param_file_path, param_file_content)

        job_name = f"{family_name}_{configuration_type}" + (f"_{tested_transition}" if tested_transition else "")
        job_file_path: Path = family_main_run_dir / f"{job_name}.sh"
        job_file_content = build_job_script_for_chromevol_run(job_folder_path=family_main_run_dir, job_name=job_name, param_file_path=param_file_path)
        general_submit_job(job_file_path, job_name, job_file_content)

        time.sleep(3)

    print("\n[✓][✓][✓] Finished submission")


def parse_args() -> tuple[Path, str, Optional[str], Optional[Path]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--families_file", type=Path, required=True, help="Path to txt file where each row is a family name.")
    parser.add_argument("--configuration_type", required=True, help="Configuration type.")
    parser.add_argument("--tested_transition", default=None, help="Optional tested transition (required unless configuration_type is all-const).")
    parser.add_argument("--config_data_file", type=Path, default=None, help="Optional path to config data file.")
    args = parser.parse_args()
    return args.families_file, args.configuration_type, args.tested_transition, args.config_data_file

def main() -> None:
    families_file, configuration_type, tested_transition, config_data_file = parse_args()

    prepare_and_submit_chromevol_jobs_for_families(
        families_file=families_file,
        configuration_type=configuration_type,
        tested_transition=tested_transition,
        config_data_file=config_data_file,
    )

    with families_file.open("r", encoding="utf-8") as f:
        families_list = [line.strip() for line in f if line.strip()]

    n_families = len(families_list)

    log_run(
        step="chromevol_run",
        script=Path(__file__),
        params={
            "configuration_type": configuration_type,
            "tested_transition": tested_transition or "",
            "families_file": str(families_file),
            "config_data_file": str(config_data_file) if config_data_file else "",
            "n_families": n_families,
        },
        outputs=[],
        description=f"Starting Chromevol run for configuration '{configuration_type}'",
        notes="Preparing and submitting Chromevol jobs for families.",
        step_extension=f"_{configuration_type}",
    )


if __name__ == "__main__":
    main()
