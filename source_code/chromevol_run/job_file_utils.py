from subprocess import Popen
from ..constants import *

def general_job_format(job_folder_path: str, job_name: str, cmd: str, memory: str = "4G", ncpu: int="1") -> str:
    text = ""
    text += f"#!/bin/bash\n\n"
    text += f"#SBATCH --job-name={job_name}\n"
    text += f"#SBATCH --account=itaym-users_v2\n"
    text += f"#SBATCH --partition=itaym-pool\n"
    text += f"#SBATCH --ntasks={ncpu}\n"
    text += f"#SBATCH --cpus-per-task=1\n"
    text += f"#SBATCH --time=7-00:00:00\n"
    text += f"#SBATCH --mem-per-cpu={memory}\n"
    text += f"#SBATCH --output={job_folder_path}/{job_name}_out.OU\n"
    text += f"#SBATCH --error={job_folder_path}/{job_name}_err.ER\n"
    text += f"cd {job_folder_path}\n"
    text += f"{cmd}\n"
    return text

def build_job_script_for_chromevol_run(job_folder_path: Path, job_name: str, param_file_path: str) -> str:
    log_file = job_folder_path / "log.txt"
    err_file = job_folder_path / "ERR.txt"
    cmd = f'{CONDA_ENV}\n{CONDA_EXPORT}{CHROMEVOL_EXE} "param={param_file_path}" > {log_file} 2> {err_file}\n'
    job_content = general_job_format(str(job_folder_path), job_name, cmd)
    return job_content

def general_submit_job(job_path: Path, job_name: str, content: str) -> None:
    with open(job_path, 'w') as f:
        f.write(content)
    Popen(["sbatch", str(job_path)])
    print(f"[→] Submitting job: {job_name}")
