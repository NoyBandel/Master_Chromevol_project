argspars

if __name__ == '__main__':

    1. take as input: all the needed \ optionat inputs of create_pfiles_and_submit_jobs: families_list: List[str], configuration_type: str, tested_transition: Optional[str] = None, config_data_file: Optional[Path] = None):
    2. create a folder job_folder_path=CHROMEVOL_RAW_RESULTS_ROOT / configuration_type / tested_transition (if exists as input)
    3. creates the main job that runs create_pfiles_and_submit_jobs, saves it in the created above folder job_folder_path
    4. submits the job

    cmd =
    python "/groups/itay_mayrose/noybandel/Master_ChromEvol_project/source_code/chromevol_run/create_pfiles_and_submit_jobs.py"

    job_name =

    job_content: str = general_job_format(job_folder_path, job_name, cmd: str)
    general_submit_job(job_path, job_name, job_content)

    write to the general log



