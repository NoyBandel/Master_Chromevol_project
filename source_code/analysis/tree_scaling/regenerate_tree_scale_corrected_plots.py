"""
Regenerate plots using tree-scale-corrected values in temporary CSV copies.

Expected JSON format:

{
  "transitions": ["dupl", "gain", "loss"],
  "jobs": [
    {
      "name": "job_name",
      "enabled": true,
      "module": "source_code.analysis.module_name",
      "temporary_inputs": [
        {
          "name": "input_name",
          "source": "analysis/{transition}/input.csv",
          "column_replacements": {
            "legacy_column": "corrected_column"
          },
          "validate_scaling": true
        }
      ],
      "arguments": [
        "--input", "{temp_input_name}",
        "--output-dir", "{temporary_output_dir}"
      ],
      "outputs": [
        {
          "source": "generated_plot.png",
          "destination": "analysis/{transition}/generated_plot.png"
        }
      ]
    }
  ]
}

Available placeholders:
{project_root}, {transition}, {job_name}, {temporary_output_dir},
and {temp_<input_name>}.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd

from source_code.logger import log_run


JsonDict = dict[str, Any]
OutputPair = tuple[Path, Path]

TREE_SCALING_FACTOR_COL: str = "tree_scaling_factor"
CORRECTED_SCALING_NOTE: str = (
    "Tree-scale-corrected regeneration run. Plotting scripts used temporary "
    "CSV copies in which legacy parameter columns contained the corresponding "
    "tree-scale-corrected values. Permanent analysis tables were not modified."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate plots using temporary tree-scale-corrected CSV inputs.")
    parser.add_argument("--config", type=Path, required=True, help="JSON configuration file.")
    parser.add_argument("--transition", action="append", default=None, help="Run only this transition. May be supplied multiple times.")
    parser.add_argument("--job", action="append", default=None, help="Run only this job. May be supplied multiple times.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print commands without running them.")
    return parser.parse_args()


def load_config(config_file: Path) -> JsonDict:
    if not config_file.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    with config_file.open("r", encoding="utf-8") as file_handle:
        config: JsonDict = json.load(file_handle)

    if not isinstance(config.get("transitions"), list) or not config["transitions"]:
        raise ValueError("The configuration must contain a nonempty 'transitions' list.")

    if not isinstance(config.get("jobs"), list) or not config["jobs"]:
        raise ValueError("The configuration must contain a nonempty 'jobs' list.")

    return config


def render(value: object, context: dict[str, str]) -> str:
    return str(value).format(**context)


def resolve_path(project_root: Path, value: object, context: dict[str, str]) -> Path:
    path: Path = Path(render(value, context))
    return path if path.is_absolute() else project_root / path


def select_transitions(configured: list[str], requested: list[str] | None) -> list[str]:
    if requested is None:
        return configured

    unknown: list[str] = sorted(set(requested) - set(configured))

    if unknown:
        raise ValueError(f"Unknown transitions: {unknown}")

    return [transition for transition in configured if transition in requested]


def select_jobs(configured: list[JsonDict], requested: list[str] | None) -> list[JsonDict]:
    enabled_jobs: list[JsonDict] = [job for job in configured if bool(job.get("enabled", True))]
    job_names: list[str] = [str(job["name"]) for job in enabled_jobs]

    if len(job_names) != len(set(job_names)):
        raise ValueError("Job names must be unique.")

    if requested is None:
        return enabled_jobs

    unknown: list[str] = sorted(set(requested) - set(job_names))

    if unknown:
        raise ValueError(f"Unknown or disabled jobs: {unknown}")

    return [job for job in enabled_jobs if str(job["name"]) in requested]


def validate_scaling(df: pd.DataFrame, replacements: dict[str, str], source_file: Path) -> None:
    if TREE_SCALING_FACTOR_COL not in df.columns:
        raise KeyError(f"Missing '{TREE_SCALING_FACTOR_COL}' in {source_file}")

    scaling_factor: pd.Series = pd.to_numeric(df[TREE_SCALING_FACTOR_COL], errors="coerce")

    for raw_column, corrected_column in replacements.items():
        raw_values: pd.Series = pd.to_numeric(df[raw_column], errors="coerce")
        corrected_values: pd.Series = pd.to_numeric(df[corrected_column], errors="coerce")
        comparable: pd.Series = raw_values.notna() & corrected_values.notna() & scaling_factor.notna()

        if not comparable.any():
            raise ValueError(f"No comparable rows for validating {raw_column} against {corrected_column} in {source_file}")

        expected_values: pd.Series = raw_values[comparable] * scaling_factor[comparable]
        matches: np.ndarray = np.isclose(corrected_values[comparable], expected_values, rtol=1e-8, atol=1e-12)

        if not matches.all():
            mismatch_rows: list[int] = corrected_values[comparable].index[~matches].tolist()
            raise ValueError(f"Invalid corrected values for {raw_column} in {source_file}. Mismatched rows: {mismatch_rows[:20]}")


def create_temporary_input(input_spec: JsonDict, project_root: Path, temporary_root: Path, context: dict[str, str]) -> tuple[str, Path]:
    input_name: str = str(input_spec["name"])
    source_file: Path = resolve_path(project_root, input_spec["source"], context)

    if not source_file.is_file():
        raise FileNotFoundError(f"Temporary-input source file not found: {source_file}")

    df: pd.DataFrame = pd.read_csv(source_file)
    replacements: dict[str, str] = {str(raw): str(corrected) for raw, corrected in dict(input_spec["column_replacements"]).items()}
    required_columns: set[str] = set(replacements) | set(replacements.values())
    missing_columns: list[str] = sorted(required_columns - set(df.columns))

    if missing_columns:
        raise KeyError(f"Missing columns in {source_file}: {missing_columns}")

    if bool(input_spec.get("validate_scaling", False)):
        validate_scaling(df, replacements, source_file)

    for legacy_column, corrected_column in replacements.items():
        df[legacy_column] = df[corrected_column]

    temporary_input_dir: Path = temporary_root / "inputs" / context["transition"] / context["job_name"]
    temporary_input_dir.mkdir(parents=True, exist_ok=True)

    temporary_file: Path = temporary_input_dir / f"{input_name}_{source_file.name}"
    df.to_csv(temporary_file, index=False)

    return input_name, temporary_file


def run_job(job: JsonDict, transition: str, project_root: Path, temporary_root: Path, dry_run: bool) -> list[OutputPair]:
    job_name: str = str(job["name"])
    module_name: str = str(job["module"])
    temporary_output_dir: Path = temporary_root / "outputs" / transition / job_name
    temporary_output_dir.mkdir(parents=True, exist_ok=True)

    context: dict[str, str] = {
        "project_root": project_root.as_posix(),
        "transition": transition,
        "job_name": job_name,
        "temporary_output_dir": temporary_output_dir.as_posix(),
    }

    for input_spec in job.get("temporary_inputs", []):
        input_name, temporary_file = create_temporary_input(input_spec, project_root, temporary_root, context)
        context[f"temp_{input_name}"] = temporary_file.as_posix()

    command: list[str] = [sys.executable, "-m", module_name, *[render(argument, context) for argument in job.get("arguments", [])]]

    print(f"\n=== {transition}: {job_name} ===")
    print(" ".join(command))

    if dry_run:
        for output_spec in job.get("outputs", []):
            print(f"[DRY RUN] Final output: {render(output_spec['destination'], context)}")
        return []

    subprocess.run(command, cwd=project_root, check=True)

    outputs: list[OutputPair] = []

    for output_spec in job.get("outputs", []):
        generated_file: Path = temporary_output_dir / render(output_spec["source"], context)
        destination_file: Path = resolve_path(project_root, output_spec["destination"], context)

        if not generated_file.is_file():
            raise FileNotFoundError(f"Expected plot was not generated: {generated_file}")

        if generated_file.stat().st_size == 0:
            raise ValueError(f"Generated plot is empty: {generated_file}")

        outputs.append((generated_file, destination_file))

    return outputs


def replace_outputs(outputs: list[OutputPair]) -> None:
    destinations: list[Path] = [destination for _, destination in outputs]

    if len(destinations) != len(set(destinations)):
        raise ValueError("Multiple jobs target the same destination file.")

    for generated_file, destination_file in outputs:
        destination_file.parent.mkdir(parents=True, exist_ok=True)

        temporary_destination: Path = destination_file.with_name(f".{destination_file.name}.tree_scale_tmp")
        shutil.copy2(generated_file, temporary_destination)
        temporary_destination.replace(destination_file)

        print(f"[✓] Replaced: {destination_file}")


def main() -> None:
    args: argparse.Namespace = parse_args()
    project_root: Path = Path.cwd().resolve()
    config_file: Path = args.config.resolve()
    config: JsonDict = load_config(config_file)

    transitions: list[str] = select_transitions([str(transition) for transition in config["transitions"]], args.transition)
    jobs: list[JsonDict] = select_jobs(list(config["jobs"]), args.job)
    outputs: list[OutputPair] = []

    with TemporaryDirectory(prefix="tree_scale_corrected_plots_") as temporary_directory:
        temporary_root: Path = Path(temporary_directory)

        for transition in transitions:
            for job in jobs:
                outputs.extend(run_job(job, transition, project_root, temporary_root, args.dry_run))

        if not args.dry_run:
            replace_outputs(outputs)

    if not args.dry_run:
        final_outputs: list[str] = [destination.as_posix() for _, destination in outputs]

        log_run(
            step="regenerate_tree_scale_corrected_plots",
            script=Path(__file__),
            params={"config_file": str(config_file), "transitions": transitions, "jobs": [str(job["name"]) for job in jobs], "number_of_plots": len(final_outputs)},
            outputs=final_outputs,
            description="Regenerated parameter-dependent plots using tree-scale-corrected values.",
            notes=CORRECTED_SCALING_NOTE,
        )

    print("\n[✓] Plot-regeneration workflow completed.")
    print(f"Transitions: {', '.join(transitions)}")
    print(f"Jobs: {', '.join(str(job['name']) for job in jobs)}")
    print(f"Plots replaced: {0 if args.dry_run else len(outputs)}")


if __name__ == "__main__":
    main()