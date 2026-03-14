from subprocess import run
from typing import Optional
from source_code.constants import *

def general_job_format(job_folder_path: str, job_name: str, cmd: str, memory: str = "4G", ncpu: str="1", workdir: Optional[Path] = None) -> str:
    out_path = f"{job_folder_path}/{job_name}_out.OU"
    err_path = f"{job_folder_path}/{job_name}_err.ER"
    cd_path = workdir if workdir else job_folder_path

    text = ""
    text += "#!/bin/bash\n\n"
    text += f"#SBATCH --job-name={job_name}\n"
    text += f"#SBATCH --account=itaym-users_v2\n"
    text += f"#SBATCH --partition=itaym-pool\n"
    text += f"#SBATCH --ntasks={ncpu}\n"
    text += f"#SBATCH --cpus-per-task=1\n"
    text += f"#SBATCH --time=7-00:00:00\n"
    text += f"#SBATCH --mem-per-cpu={memory}\n"
    text += f'#SBATCH --output="{out_path}"\n'
    text += f'#SBATCH --error="{err_path}"\n\n'

    text += f'cd "{cd_path}"\n'
    text += cmd + "\n"

    return text

def build_job_script_for_chromevol_run(job_folder_path: Path, job_name: str, param_file_path: str) -> str:
    log_file = job_folder_path / "log.txt"
    err_file = job_folder_path / "ERR.txt"

    cmd = ""
    cmd += f"{CONDA_ENV}\n"
    cmd += f"{CONDA_EXPORT}\n"
    cmd += f'echo "START $(date)" > "{log_file}"\n'
    cmd += 'start_ts=$(date +%s)\n'
    cmd += f'{CHROMEVOL_EXE} "param={param_file_path}" >> "{log_file}" 2>> "{err_file}"\n'
    cmd += 'end_ts=$(date +%s)\n'
    cmd += 'echo "END $(date)" >> "{log_file}"\n'
    cmd += 'echo "DURATION_SEC=$((end_ts - start_ts))" >> "{log_file}"\n'

    job_content = general_job_format(str(job_folder_path), job_name, cmd)
    return job_content

def general_submit_job(job_path: Path, job_name: str, content: str) -> str:
    with open(job_path, "w") as f:
        f.write(content)

    result = run(["sbatch", str(job_path)], capture_output=True, text=True)

    if result.returncode != 0:
        error_msg = result.stderr.strip()
        print(f"[✗] Job submission failed | {job_name} | {job_path}")
        print(f"    Error: {error_msg}")
        return error_msg

    job_id = result.stdout.strip().split()[-1]
    print(f"[→] Job submitted: {job_id}  |  {job_name}  |  {job_path}")
    return job_id
