import time
from prepare_data_for_runs import *
from job_file_utils import *
from param_file_utils import *
from create_config_df  import *


def create_pfiles_and_submit_jobs(families_list: List[str], configuration_type: str, tested_transition: Optional[str] = None, config_data_file: Optional[Path] = None):
    results_dirs_dict: Dict[str, str] = create_output_folders(configuration_type, families_list)
    for family_name in families_list:
        print(f"--------{family_name}--------")
        family_main_run_dir: Path = Path(results_dirs_dict[family_name].strip("/Results"))

        family_data_input_dir: Path = DATABASE_INPUT_DIR / family_name
        dataFile_path = str(family_data_input_dir / CHROM_COUNTS_FILE_NAME)
        treeFile_path = str(family_data_input_dir / TREE_FILE_NAME)
        resultsPathDir_path = results_dirs_dict[family_name]

        config_df:pd.DataFrame, base_num_init:str = create_configuration(configuration_type, tested_transition, config_data_file)
        param_file_content: str = build_param_file_content(dataFile_path, treeFile_path, resultsPathDir_path, config_df, base_num_init)

        param_file_path = family_main_run_dir/PARAM_FILE
        write_param_file(param_file_path, param_file_content)

        job_name: str = f"{family_name}_{configuration_type}_{tested_transition}"
        job_file_path: Path = family_main_run_dir / f"{job_name}.sh"
        job_file_content = build_job_script_for_chromevol_run(job_folder_path=family_main_run_dir, job_name=job_name, param_file_path=param_file_path)
        general_submit_job(job_file_path, job_name, job_file_content)

        time.sleep(3)
