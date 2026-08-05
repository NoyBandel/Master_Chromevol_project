"""
Extract family-specific tree-scaling information from the empirical M0 constant-model ChromEvol results.

Input:
--families-file
Text file containing one family name per line.

For each listed family, the script reads: <M0_RESULTS_ROOT>/<family>/Results/chromEvol.res
and extracts:
- Tree scaling factor
- Original tree length
- Scaled tree length

Outputs:
tree_scaling_values.csv
tree_scaling_histogram.png

Tree scaling must be corrected because rates and slopes estimated on differently scaled trees are not directly comparable across families.
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from source_code.constants import *
from source_code.logger import log_run


M0_RESULTS_ROOT: Path = CHROMEVOL_RAW_RESULTS_ROOT / M0_LABEL
OUTPUT_DIR: Path = ANALYSIS_DIR / "tree_scaling"

CHROMEVOL_RESULTS_FILE_NAME: str = "chromEvol.res"
TREE_SCALING_FACTOR_COL: str = "tree_scaling_factor"
ORIGINAL_TREE_LENGTH_COL: str = "original_tree_length"
SCALED_TREE_LENGTH_COL: str = "scaled_tree_length"

NUMBER_PATTERN: str = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
SCALING_PATTERN: re.Pattern[str] = re.compile(rf"Tree scaling factor is:\s*{NUMBER_PATTERN}", re.IGNORECASE)
ORIGINAL_LENGTH_PATTERN: re.Pattern[str] = re.compile(rf"Original tree length was:\s*{NUMBER_PATTERN}", re.IGNORECASE)
SCALED_LENGTH_PATTERN: re.Pattern[str] = re.compile(rf"tree Length was scaled to:\s*{NUMBER_PATTERN}", re.IGNORECASE)


def read_family_names(families_file: Path) -> list[str]:
    if not families_file.is_file():
        raise FileNotFoundError(f"Missing families file: {families_file}")

    family_names: list[str] = []

    for line in families_file.read_text(encoding="utf-8").splitlines():
        family_name: str = line.strip()

        if family_name and not family_name.startswith("#"):
            family_names.append(family_name)

    return sorted(set(family_names))


def extract_value(text: str, pattern: re.Pattern[str], value_name: str, results_file: Path) -> float:
    match: re.Match[str] | None = pattern.search(text)

    if match is None:
        raise ValueError(f"Could not find {value_name} in {results_file}")

    return float(match.group(1))


def extract_family_scaling(family_name: str) -> dict[str, float | str]:
    results_file: Path = M0_RESULTS_ROOT / family_name / "Results" / CHROMEVOL_RESULTS_FILE_NAME

    if not results_file.is_file():
        raise FileNotFoundError(f"Missing results file: {results_file}")

    text: str = results_file.read_text(encoding="utf-8", errors="replace")

    tree_scaling_factor: float = extract_value(text, SCALING_PATTERN, "tree scaling factor", results_file)
    original_tree_length: float = extract_value(text, ORIGINAL_LENGTH_PATTERN, "original tree length", results_file)
    scaled_tree_length: float = extract_value(text, SCALED_LENGTH_PATTERN, "scaled tree length", results_file)

    return {
        FAMILY_NAME_COL: family_name,
        TREE_SCALING_FACTOR_COL: tree_scaling_factor,
        ORIGINAL_TREE_LENGTH_COL: original_tree_length,
        SCALED_TREE_LENGTH_COL: scaled_tree_length,
    }


def plot_scaling_histogram(scaling_df: pd.DataFrame, output_path: Path) -> None:
    values: pd.Series = scaling_df[TREE_SCALING_FACTOR_COL]
    mean_value: float = values.mean()
    median_value: float = values.median()

    plt.figure(figsize=(8, 5))
    plt.hist(values, bins="auto", edgecolor="black")

    plt.axvline(mean_value, color="red", linestyle="--", linewidth=1.5, label=f"Mean = {mean_value:.3f}")
    plt.axvline(median_value, color="black", linestyle="--", linewidth=1.5, label=f"Median = {median_value:.3f}")

    plt.xlabel("Tree scaling factor")
    plt.ylabel("Number of families")
    plt.title("Distribution of ChromEvol tree scaling factors")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract ChromEvol tree-scaling values for a specified list of families.")
    parser.add_argument("--families-file", type=Path, required=True, help="Text file containing one family name per line.")
    return parser.parse_args()

def main() -> None:
    args: argparse.Namespace = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    family_names: list[str] = read_family_names(args.families_file)
    rows: list[dict[str, float | str]] = []

    for family_name in family_names:
        try:
            rows.append(extract_family_scaling(family_name))
        except (FileNotFoundError, ValueError) as error:
            print(f"WARNING: {error}")

    if not rows:
        raise RuntimeError("No tree-scaling information was extracted.")

    scaling_df: pd.DataFrame = pd.DataFrame(
        rows,
        columns=[
            FAMILY_NAME_COL,
            TREE_SCALING_FACTOR_COL,
            ORIGINAL_TREE_LENGTH_COL,
            SCALED_TREE_LENGTH_COL,
        ],
    ).sort_values(FAMILY_NAME_COL).reset_index(drop=True)

    output_csv: Path = OUTPUT_DIR / "tree_scaling_values.csv"
    output_plot: Path = OUTPUT_DIR / "tree_scaling_histogram.png"

    scaling_df.to_csv(output_csv, index=False)
    plot_scaling_histogram(scaling_df=scaling_df, output_path=output_plot)

    log_run(
        step="tree_scaling",
        script=Path(__file__),
        params={
            "families_file": args.families_file,
            "m0_results_root": M0_RESULTS_ROOT,
            "families_requested": len(family_names),
            "families_extracted": len(scaling_df),
        },
        outputs=[output_csv.as_posix(), output_plot.as_posix()],
        description="Extracted family-specific tree-scaling values from empirical M0 constant-model results and plotted their distribution.",
    )

    print(f"Extracted scaling information for {len(scaling_df)} of {len(family_names)} requested families.")
    print(f"CSV saved to: {output_csv}")
    print(f"Histogram saved to: {output_plot}")


if __name__ == "__main__":
    main()