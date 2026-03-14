import argparse
import subprocess
from pathlib import Path
from typing import List, Optional
from source_code.logger import log_run
from source_code.constants import *

OK_SYMBOL = "[✓]"
RUN_SYMBOL = "[→]"
PROBLEM_SYMBOL = "[!]"


def get_slurm_job_status(job_name: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["squeue", "--noheader", "--format=%T", "--name", job_name],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return None

    if result.returncode != 0:
        return None

    states = [line.strip().upper() for line in result.stdout.splitlines() if line.strip()]

    if "RUNNING" in states:
        return "RUNNING"

    if "PENDING" in states:
        return "PENDING"

    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor ChromEvol runs for a given configuration.")
    parser.add_argument("--families_file", type=Path, required=True, help="Path to families.txt")
    parser.add_argument("--families_root", type=Path, required=True, help="Path to the configuration root folder")
    parser.add_argument("--configuration", type=str, required=True, help="Configuration name, e.g. M0_all_const")
    return parser.parse_args()


def load_families(families_file: Path) -> List[str]:
    with families_file.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def file_nonempty(file_path: Path) -> bool:
    return file_path.exists() and file_path.is_file() and file_path.stat().st_size > 0


def read_full_file(file_path: Path) -> str:
    if not file_path.exists():
        return ""

    with file_path.open("r", encoding="utf-8", errors="replace") as f:
        return f.read().strip()


def check_family(family: str, families_root: Path, configuration: str) -> tuple[Optional[str], Optional[str]]:
    family_dir = families_root / family

    job_script_path = family_dir / f"{family}_{configuration}.sh"
    job_err_path = family_dir / f"{family}_{configuration}_err.ER"
    run_err_path = family_dir / "ERR.txt"
    results_dir = family_dir / "Results"
    res_path = results_dir / "chromEvol.res"

    job_name = f"{family}_{configuration}"
    slurm_status = get_slurm_job_status(job_name)
    if slurm_status in {"RUNNING", "PENDING"}:
        return f"{family}: {slurm_status}", None

    if not job_script_path.exists():
        return None, (
            f"{family}:\n"
            f"error type: MISSING_JOB_SCRIPT\n"
            f"error message: {job_script_path}\n"
        )

    if not job_err_path.exists():
        return None, (
            f"{family}:\n"
            f"error type: MISSING_JOB_ERR_FILE\n"
            f"error message: {job_err_path}\n"
        )

    if file_nonempty(job_err_path):
        job_err_text = read_full_file(job_err_path)
        return None, (
            f"{family}:\n"
            f"error type: JOB_ERR_NONEMPTY\n"
            f"error message:\n{job_err_text}\n"
        )

    if not run_err_path.exists():
        return None, (
            f"{family}:\n"
            f"error type: MISSING_RUN_ERR_FILE\n"
            f"error message: {run_err_path}\n"
        )

    if file_nonempty(run_err_path):
        run_err_text = read_full_file(run_err_path)
        return None, (
            f"{family}:\n"
            f"error type: RUN_ERR_NONEMPTY\n"
            f"error message:\n{run_err_text}\n"
        )

    if not results_dir.exists():
        return None, (
            f"{family}:\n"
            f"error type: MISSING_RESULTS_DIR\n"
            f"error message: {results_dir}\n"
        )

    if not res_path.exists():
        return None, (
            f"{family}:\n"
            f"error type: MISSING_RES_FILE\n"
            f"error message: {res_path}\n"
        )

    return None, None


def print_results(job_status_lines: List[str], error_blocks: List[str]) -> None:
    if not job_status_lines and not error_blocks:
        print(f"{OK_SYMBOL} all jobs finished successfully")
        return

    if job_status_lines:
        print("--- job status ---")
        for line in job_status_lines:
            print(f"{RUN_SYMBOL} {line}")
        print()

    if error_blocks:
        print("--- errors ---")
        for block in error_blocks:
            family_line = block.splitlines()[0]
            error_type_line = block.splitlines()[1]
            print(f"{PROBLEM_SYMBOL} {family_line} | {error_type_line.replace('error type: ', '')}")


def save_errors_txt(error_blocks: List[str], families_root: Path, configuration: str) -> Optional[Path]:
    if not error_blocks:
        return None

    output_path = families_root / f"run_errors_summary_{configuration}.txt"

    with output_path.open("w", encoding="utf-8") as f:
        for i, block in enumerate(error_blocks):
            if i > 0:
                f.write("\n" + "=" * 80 + "\n\n")
            f.write(block.strip() + "\n")

    return output_path


def build_notes(job_status_lines: List[str], error_blocks: List[str]) -> str:
    if not job_status_lines and not error_blocks:
        return f"{OK_SYMBOL} all jobs finished successfully"

    notes_lines: List[str] = []

    if job_status_lines:
        notes_lines.append(f"Running/pending jobs: {len(job_status_lines)}")

    if error_blocks:
        if notes_lines:
            notes_lines.append("")
        notes_lines.append(f"Families with errors: {len(error_blocks)}")

        for block in error_blocks:
            family_name = block.splitlines()[0].rstrip(":")
            notes_lines.append(f"   {family_name}")

    return "\n".join(notes_lines)


def main() -> None:
    args = parse_args()
    families = load_families(args.families_file)

    job_status_lines: List[str] = []
    error_blocks: List[str] = []

    for family in families:
        job_status_line, error_block = check_family(
            family=family,
            families_root=args.families_root,
            configuration=args.configuration,
        )

        if job_status_line is not None:
            job_status_lines.append(job_status_line)

        if error_block is not None:
            error_blocks.append(error_block)

    errors_txt_path = save_errors_txt(error_blocks, args.families_root, args.configuration)
    print_results(job_status_lines, error_blocks)

    notes = build_notes(job_status_lines, error_blocks)
    model_type = M0_LABEL if args.configuration == M0_LABEL else args.configuration.split("_")[0]
    log_relative_path = Path(model_type) / f"{args.configuration}.log"

    log_run(
        step="chromevol_run",
        script=Path(__file__),
        params={
            "families_file": str(args.families_file),
            "families_root": str(args.families_root),
            "configuration": args.configuration,
        },
        outputs=[str(errors_txt_path)] if errors_txt_path else [],
        description=f"Monitored ChromEvol runs for configuration {args.configuration}.",
        notes=notes,
        log_relative_path=log_relative_path,
    )


if __name__ == "__main__":
    main()