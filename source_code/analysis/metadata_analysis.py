import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from source_code.analysis.analysis_constants import *
from source_code.logger import log_run


FEATURES = [
    (FAMILY_SIZE_COL, "Family size distribution", "Family size", "family_size_histogram.png"),
    (MIN_CHROM_COL, "Minimum chromosome number distribution", "Minimum chromosome number", "min_chrom_histogram.png"),
    (MAX_CHROM_COL, "Maximum chromosome number distribution", "Maximum chromosome number", "max_chrom_histogram.png"),
    (STD_CHROM_COL, "Chromosome standard deviation distribution", "Chromosome standard deviation", "std_chrom_histogram.png"),
    (DIFF_COL, "Chromosome range distribution", "Chromosome range", "chrom_range_histogram.png"),
]


# Load metadata
def load_family_metadata(families_metadata_file: Path) -> pd.DataFrame:
    if not families_metadata_file.exists():
        raise FileNotFoundError(f"File not found: {families_metadata_file}")

    df = pd.read_csv(families_metadata_file)

    required = [FAMILY_NAME_COL, FAMILY_SIZE_COL, MIN_CHROM_COL, MAX_CHROM_COL, STD_CHROM_COL, DIFF_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df.drop_duplicates(subset=FAMILY_NAME_COL).reset_index(drop=True)


# Build stats
def build_stats(series: pd.Series) -> dict[str, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    return {
        "n": int(len(s)),
        "mean": s.mean(),
        "median": s.median(),
        "std": s.std(),
        "min": s.min(),
        "max": s.max(),
    }


# Format stats (single source of truth)
def format_stats(stats: dict[str, float], precision: int = 2) -> str:
    return (
        f"n = {stats['n']}\n"
        f"mean = {stats['mean']:.{precision}f}\n"
        f"median = {stats['median']:.{precision}f}\n"
        f"std = {stats['std']:.{precision}f}\n"
        f"min = {stats['min']:.{precision}f}\n"
        f"max = {stats['max']:.{precision}f}"
    )


# Plot histogram
def plot_histogram(df: pd.DataFrame, col: str, title: str, xlabel: str, out_file: Path, bin_count: int | None = 30, bin_size: float | None = None) -> dict[str, float]:
    values = pd.to_numeric(df[col], errors="coerce").dropna()
    stats = build_stats(values)

    if bin_count is not None and bin_size is not None:
        raise ValueError("Use either bin_count or bin_size, not both.")

    if bin_size is not None:
        import numpy as np

        start = np.floor(values.min() / bin_size) * bin_size
        end = np.ceil(values.max() / bin_size) * bin_size + bin_size
        bins = np.arange(start, end, bin_size)
    else:
        bins = bin_count if bin_count is not None else 30

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(values, bins=bins, edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of families")

    ax.text(
        0.98, 0.98,
        format_stats(stats),
        transform=ax.transAxes,
        ha="right",
        va="top",
        multialignment="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    fig.tight_layout()
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return stats


# Save summary
def save_summary(all_stats: dict[str, dict[str, float]], out_file: Path) -> None:
    lines = ["Metadata summary", "================", ""]

    for col, title, _, _ in FEATURES:
        stats = all_stats[col]
        lines.extend([
            title,
            "-" * len(title),
            format_stats(stats, precision=6),
            "",
        ])

    out_file.write_text("\n".join(lines), encoding="utf-8")


# Main analysis
def run_metadata_analysis(families_metadata_file: Path) -> None:
    df = load_family_metadata(families_metadata_file)

    out_dir = ANALYSIS_DIR / "metadata"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_stats = {}
    output_files = []

    for col, title, xlabel, filename in FEATURES:
        out_file = out_dir / filename
        all_stats[col] = plot_histogram(df, col, title, xlabel, out_file, bin_size=1)
        output_files.append(out_file.as_posix())

    summary_file = out_dir / "metadata_summary.txt"
    save_summary(all_stats, summary_file)
    output_files.append(summary_file.as_posix())

    log_run(
        step="analysis",
        script=Path(__file__),
        params={
            "analysis_type": "metadata",
            "families_metadata_file": families_metadata_file.as_posix(),
            "n_families": len(df),
            "features": [col for col, _, _, _ in FEATURES],
        },
        outputs=output_files,
        description="Created metadata histograms and summary statistics.",
        log_relative_path=Path("metadata/metadata_analysis.log"),
    )


# CLI
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run metadata analysis.")
    parser.add_argument("--families_metadata_file", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_metadata_analysis(args.families_metadata_file)

if __name__ == "__main__":
    main()