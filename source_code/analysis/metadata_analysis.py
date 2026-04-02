import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from source_code.analysis.analysis_constants import *


def load_family_metadata(families_metadata_file: Path) -> pd.DataFrame:
    if not families_metadata_file.exists():
        raise FileNotFoundError(f"Family metadata file not found: {families_metadata_file}")

    family_df = pd.read_csv(families_metadata_file)

    required_cols: List[str] = [FAMILY_NAME_COL, FAMILY_SIZE_COL, MIN_CHROM_COL, MAX_CHROM_COL, DIFF_COL]
    missing_cols = [col for col in required_cols if col not in family_df.columns]
    if missing_cols:
        raise ValueError(f"Family metadata file is missing required columns: {missing_cols}")

    family_df = family_df.copy()
    family_df = family_df.drop_duplicates(subset=[FAMILY_NAME_COL]).reset_index(drop=True)
    return family_df


def build_summary_stats(series: pd.Series) -> Dict[str, float]:
    clean_series = pd.to_numeric(series, errors="coerce").dropna()

    return {
        "n": int(clean_series.shape[0]),
        "mean": clean_series.mean(),
        "median": clean_series.median(),
        "std": clean_series.std(),
        "min": clean_series.min(),
        "q25": clean_series.quantile(0.25),
        "q75": clean_series.quantile(0.75),
        "max": clean_series.max(),
    }


def format_stats_text(stats: Dict[str, float]) -> str:
    return (
        f"n = {stats['n']}\n"
        f"mean = {stats['mean']:.2f}\n"
        f"median = {stats['median']:.2f}\n"
        f"std = {stats['std']:.2f}\n"
        f"min = {stats['min']:.2f}\n"
        f"q25 = {stats['q25']:.2f}\n"
        f"q75 = {stats['q75']:.2f}\n"
        f"max = {stats['max']:.2f}"
    )


def build_correlation_stats(x: pd.Series, y: pd.Series) -> Dict[str, float]:
    pair_df = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()

    if pair_df.empty or pair_df.shape[0] < 2:
        return {
            "n": int(pair_df.shape[0]),
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
            "spearman_rho": float("nan"),
            "spearman_p": float("nan"),
        }

    pearson_r, pearson_p = pearsonr(pair_df["x"], pair_df["y"])
    spearman_rho, spearman_p = spearmanr(pair_df["x"], pair_df["y"])

    return {
        "n": int(pair_df.shape[0]),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
    }


def plot_metadata_histogram(
    family_df: pd.DataFrame,
    value_col: str,
    title: str,
    xlabel: str,
    output_plot_file: Path,
) -> Dict[str, float]:
    values = pd.to_numeric(family_df[value_col], errors="coerce").dropna()
    stats = build_summary_stats(values)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(values, bins=30, edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of families")

    ax.text(
        0.98,
        0.98,
        format_stats_text(stats),
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    fig.tight_layout()
    fig.savefig(output_plot_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return stats


def plot_family_size_chrom_range_scatter(
    family_df: pd.DataFrame,
    output_plot_file: Path,
) -> Dict[str, float]:
    pair_df = family_df[[FAMILY_SIZE_COL, DIFF_COL]].copy()
    pair_df[FAMILY_SIZE_COL] = pd.to_numeric(pair_df[FAMILY_SIZE_COL], errors="coerce")
    pair_df[DIFF_COL] = pd.to_numeric(pair_df[DIFF_COL], errors="coerce")
    pair_df = pair_df.dropna()

    corr_stats = build_correlation_stats(pair_df[FAMILY_SIZE_COL], pair_df[DIFF_COL])

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(pair_df[FAMILY_SIZE_COL], pair_df[DIFF_COL], alpha=0.7)
    ax.set_title("Family size vs chromosome range")
    ax.set_xlabel("Family size: number of species")
    ax.set_ylabel("Chromosome range")

    corr_text = (
        f"n = {corr_stats['n']}\n"
        f"Pearson r = {corr_stats['pearson_r']:.3f}\n"
        f"Pearson p = {corr_stats['pearson_p']:.3e}\n"
        f"Spearman rho = {corr_stats['spearman_rho']:.3f}\n"
        f"Spearman p = {corr_stats['spearman_p']:.3e}"
    )
    ax.text(
        0.98,
        0.98,
        corr_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    fig.tight_layout()
    fig.savefig(output_plot_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return corr_stats


def save_metadata_summary(
    family_size_stats: Dict[str, float],
    chrom_range_stats: Dict[str, float],
    corr_stats: Dict[str, float],
    output_file: Path,
) -> None:
    lines = [
        "Metadata summary",
        "================",
        "",
        "Family size distribution",
        "------------------------",
        f"n: {family_size_stats['n']}",
        f"mean: {family_size_stats['mean']:.6f}",
        f"median: {family_size_stats['median']:.6f}",
        f"std: {family_size_stats['std']:.6f}",
        f"min: {family_size_stats['min']:.6f}",
        f"q25: {family_size_stats['q25']:.6f}",
        f"q75: {family_size_stats['q75']:.6f}",
        f"max: {family_size_stats['max']:.6f}",
        "",
        "Chromosome range distribution",
        "-----------------------------",
        f"n: {chrom_range_stats['n']}",
        f"mean: {chrom_range_stats['mean']:.6f}",
        f"median: {chrom_range_stats['median']:.6f}",
        f"std: {chrom_range_stats['std']:.6f}",
        f"min: {chrom_range_stats['min']:.6f}",
        f"q25: {chrom_range_stats['q25']:.6f}",
        f"q75: {chrom_range_stats['q75']:.6f}",
        f"max: {chrom_range_stats['max']:.6f}",
        "",
        "Family size vs chromosome range correlation",
        "-------------------------------------------",
        f"n: {corr_stats['n']}",
        f"pearson_r: {corr_stats['pearson_r']:.6f}",
        f"pearson_p: {corr_stats['pearson_p']:.6e}",
        f"spearman_rho: {corr_stats['spearman_rho']:.6f}",
        f"spearman_p: {corr_stats['spearman_p']:.6e}",
        "",
    ]
    output_file.write_text("\n".join(lines), encoding="utf-8")


def run_metadata_analysis(family_df: pd.DataFrame) -> None:
    output_dir = ANALYSIS_DIR / "metadata"
    output_dir.mkdir(parents=True, exist_ok=True)

    family_size_stats = plot_metadata_histogram(
        family_df=family_df,
        value_col=FAMILY_SIZE_COL,
        title="Family size distribution",
        xlabel="Family size: number of species",
        output_plot_file=output_dir / "family_size_distribution_histogram.png",
    )

    chrom_range_stats = plot_metadata_histogram(
        family_df=family_df,
        value_col=DIFF_COL,
        title="Chromosome range distribution",
        xlabel="Chromosome range",
        output_plot_file=output_dir / "chromosome_range_distribution_histogram.png",
    )

    corr_stats = plot_family_size_chrom_range_scatter(
        family_df=family_df,
        output_plot_file=output_dir / "family_size_vs_chrom_range_scatter.png",
    )

    save_metadata_summary(
        family_size_stats=family_size_stats,
        chrom_range_stats=chrom_range_stats,
        corr_stats=corr_stats,
        output_file=output_dir / "metadata_summary.txt",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create metadata plots and summary.")
    parser.add_argument("--families_metadata_file", type=Path, required=True, help="CSV file with family metadata.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    family_df = load_family_metadata(args.families_metadata_file)
    run_metadata_analysis(family_df)


if __name__ == "__main__":
    main()