"""
Compare feature-analysis results between two dataset/run roots.

This script does not redo the statistical tests. It reads the
``*_feature_stats.csv`` files produced by ``feature_analysis.py`` for an
original and a filtered analysis, combines them, and summarizes whether the
H0-vs-H1 difference became smaller or larger after filtering.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from source_code import constants as core
from source_code.analysis import analysis_constants as ac
from source_code.logger import log_run


PAIRED_RUNS_MODE = "paired_runs"
H0_REFERENCE_BY_WINNER_MODE = "h0_reference_by_winner"
INFERRED_FEATURE_MODES = (PAIRED_RUNS_MODE, H0_REFERENCE_BY_WINNER_MODE)

DEFAULT_TRANSITIONS = [core.LABEL_GAIN, core.LABEL_LOSS, core.LABEL_DUPL]

DEFAULT_FEATURES = [
    ac.FAMILY_SIZE_COL,
    ac.DIFF_COL,
    ac.NUM_OF_EVENTS_COL,
]


StatsRow = dict[str, Any]


def stats_file(
    root: Path, transition: str, h0: str, h1: str, inferred_feature_mode: str
) -> Path:
    label = f"{h0}_vs_{h1}"
    output_name = (
        label
        if inferred_feature_mode == PAIRED_RUNS_MODE
        else f"{label}_{inferred_feature_mode}"
    )
    return (
        root
        / transition
        / ac.FEATURE_ANALYSIS_SUBDIR
        / output_name
        / f"{transition}_{label}_feature_stats.csv"
    )


def require_columns(df: pd.DataFrame, cols: list[str], path: Path) -> None:
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")


def load_run_stats(
    root: Path,
    dataset_label: str,
    transitions: list[str],
    features: list[str],
    h0: str,
    h1: str,
    inferred_feature_mode: str,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    for transition in transitions:
        path = stats_file(root, transition, h0, h1, inferred_feature_mode)
        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(path)
        require_columns(
            df,
            [
                "feature",
                "comparison_type",
                "h0_model",
                "h1_model",
                "n_h0",
                "n_h1",
                "mean_h0",
                "mean_h1",
                "mean_diff_h1_minus_h0",
                "effect_size_h1_minus_h0",
                "welch_t_stat",
                "welch_p_value",
            ],
            path,
        )

        df = df[df["feature"].isin(features)].copy()
        missing_features = [feature for feature in features if feature not in set(df["feature"])]
        if missing_features:
            raise ValueError(
                f"{path} is missing requested feature rows: {missing_features}. "
                "Run feature_analysis.py with the same --features list first."
            )

        if set(df["h0_model"].astype(str)) != {h0} or set(df["h1_model"].astype(str)) != {h1}:
            raise ValueError(f"{path} does not contain the expected comparison {h0} vs {h1}")

        if inferred_feature_mode == H0_REFERENCE_BY_WINNER_MODE:
            inferred_rows = df[df["feature"].isin(ac.FEATURE_ANALYSIS_INFERRED_COLS)]
            bad = inferred_rows[
                inferred_rows["comparison_type"] != "unpaired_h0_reference_by_winner"
            ]
            if not bad.empty:
                raise ValueError(
                    f"{path} was not produced with --inferred-feature-mode "
                    f"{H0_REFERENCE_BY_WINNER_MODE}"
                )

        df.insert(0, "dataset", dataset_label)
        df.insert(1, "transition", transition)
        rows.append(df)

    return pd.concat(rows, ignore_index=True)


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg FDR adjustment, preserving NaNs and input order."""
    values = pd.to_numeric(p_values, errors="coerce").to_numpy(dtype=float)
    adjusted = np.full(len(values), np.nan, dtype=float)
    valid_idx = np.flatnonzero(np.isfinite(values))
    if not len(valid_idx):
        return pd.Series(adjusted, index=p_values.index)

    valid_p = values[valid_idx]
    order = np.argsort(valid_p)
    ranked_p = valid_p[order]
    m = len(ranked_p)

    ranked_adjusted = ranked_p * m / np.arange(1, m + 1)
    ranked_adjusted = np.minimum.accumulate(ranked_adjusted[::-1])[::-1]
    ranked_adjusted = np.clip(ranked_adjusted, 0, 1)

    valid_adjusted = np.empty(m, dtype=float)
    valid_adjusted[order] = ranked_adjusted
    adjusted[valid_idx] = valid_adjusted
    return pd.Series(adjusted, index=p_values.index)


def magnitude_change(filtered: float, original: float, tolerance: float = 1e-12) -> str:
    if pd.isna(filtered) or pd.isna(original):
        return "missing"
    delta = abs(filtered) - abs(original)
    if delta < -tolerance:
        return "smaller_after_filtering"
    if delta > tolerance:
        return "larger_after_filtering"
    return "unchanged"


def build_comparison(combined: pd.DataFrame) -> pd.DataFrame:
    value_cols = [
        "comparison_type",
        "n_h0",
        "n_h1",
        "mean_h0",
        "mean_h1",
        "mean_diff_h1_minus_h0",
        "effect_size_h1_minus_h0",
        "welch_t_stat",
        "welch_p_value",
        "welch_p_fdr_bh",
    ]

    wide = combined.pivot(
        index=["transition", "feature"],
        columns="dataset",
        values=value_cols,
    )
    wide.columns = [f"{dataset}_{metric}" for metric, dataset in wide.columns]
    wide = wide.reset_index()

    required_datasets = {"original", "filtered"}
    present = {
        column.split("_", 1)[0]
        for column in wide.columns
        if column.startswith("original_") or column.startswith("filtered_")
    }
    if present != required_datasets:
        raise ValueError(f"Expected original and filtered results, found: {sorted(present)}")

    wide["original_abs_mean_diff"] = wide["original_mean_diff_h1_minus_h0"].abs()
    wide["filtered_abs_mean_diff"] = wide["filtered_mean_diff_h1_minus_h0"].abs()
    wide["abs_mean_diff_change_filtered_minus_original"] = (
        wide["filtered_abs_mean_diff"] - wide["original_abs_mean_diff"]
    )

    wide["original_abs_effect_size"] = wide["original_effect_size_h1_minus_h0"].abs()
    wide["filtered_abs_effect_size"] = wide["filtered_effect_size_h1_minus_h0"].abs()
    wide["abs_effect_size_change_filtered_minus_original"] = (
        wide["filtered_abs_effect_size"] - wide["original_abs_effect_size"]
    )
    wide["effect_magnitude_change"] = [
        magnitude_change(filtered, original)
        for filtered, original in zip(
            wide["filtered_effect_size_h1_minus_h0"],
            wide["original_effect_size_h1_minus_h0"],
        )
    ]

    wide["original_significant_p_lt_0_05"] = wide["original_welch_p_value"] < 0.05
    wide["filtered_significant_p_lt_0_05"] = wide["filtered_welch_p_value"] < 0.05
    wide["significance_change"] = np.select(
        [
            wide["original_significant_p_lt_0_05"] & ~wide["filtered_significant_p_lt_0_05"],
            ~wide["original_significant_p_lt_0_05"] & wide["filtered_significant_p_lt_0_05"],
            wide["original_significant_p_lt_0_05"] & wide["filtered_significant_p_lt_0_05"],
        ],
        [
            "significant_to_not_significant",
            "not_significant_to_significant",
            "significant_in_both",
        ],
        default="not_significant_in_both",
    )

    return wide.sort_values(["transition", "feature"]).reset_index(drop=True)


def pretty_feature(feature: str) -> str:
    labels = {
        ac.FAMILY_SIZE_COL: "Family size",
        ac.DIFF_COL: "Chromosome range",
        ac.NUM_OF_EVENTS_COL: "Expected events",
    }
    return labels.get(feature, feature.replace("_", " ").title())


def plot_effect_size_comparison(comparison: pd.DataFrame, output_dir: Path) -> Path:
    plot_df = comparison.copy()
    plot_df["label"] = [
        f"{transition} — {pretty_feature(feature)}"
        for transition, feature in zip(plot_df["transition"], plot_df["feature"])
    ]

    y = np.arange(len(plot_df))
    height = 0.36

    fig, ax = plt.subplots(figsize=(10, max(6, 0.65 * len(plot_df) + 2)))
    ax.barh(
        y - height / 2, plot_df["original_abs_effect_size"], height=height,
        label="Original", color="#BDBDBD",
    )
    ax.barh(
        y + height / 2, plot_df["filtered_abs_effect_size"], height=height,
        label="Filtered", color="#525252",
    )
    ax.set_yticks(y, labels=plot_df["label"])
    ax.invert_yaxis()
    ax.set_xlabel("Absolute effect size |d| (Linear-selected minus Constant-selected)")
    ax.set_title("Constant-vs-Linear feature difference: original vs filtered")
    ax.axvline(0, linewidth=0.8, color="black")
    ax.legend()

    path = output_dir / "original_vs_filtered_absolute_effect_size.png"
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_p_value_comparison(comparison: pd.DataFrame, output_dir: Path) -> Path:
    plot_df = comparison.copy()
    plot_df["label"] = [
        f"{transition} — {pretty_feature(feature)}"
        for transition, feature in zip(plot_df["transition"], plot_df["feature"])
    ]

    tiny = np.finfo(float).tiny
    original_p = pd.to_numeric(plot_df["original_welch_p_value"], errors="coerce").astype(float)
    filtered_p = pd.to_numeric(plot_df["filtered_welch_p_value"], errors="coerce").astype(float)
    original_score = -np.log10(original_p.clip(lower=tiny))
    filtered_score = -np.log10(filtered_p.clip(lower=tiny))

    y = np.arange(len(plot_df))
    height = 0.36

    fig, ax = plt.subplots(figsize=(10, max(6, 0.65 * len(plot_df) + 2)))
    ax.barh(y - height / 2, original_score, height=height, label="Original", color="#BDBDBD")
    ax.barh(y + height / 2, filtered_score, height=height, label="Filtered", color="#525252")
    ax.set_yticks(y, labels=plot_df["label"])
    ax.invert_yaxis()
    ax.set_xlabel("-log10(Welch p-value)")
    ax.set_title("Welch-test significance: original vs filtered")
    ax.axvline(-np.log10(0.05), linestyle="--", color="black", linewidth=1, label="p = 0.05")
    ax.legend()

    path = output_dir / "original_vs_filtered_welch_p_values.png"
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare feature_analysis.py results between original and filtered datasets."
    )
    parser.add_argument("--original-analysis-root", required=True, type=Path)
    parser.add_argument("--filtered-analysis-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--h0-function", default=core.LABEL_CONSTANT)
    parser.add_argument("--h1-function", default=core.LABEL_LINEAR)
    parser.add_argument(
        "--inferred-feature-mode",
        choices=INFERRED_FEATURE_MODES,
        default=H0_REFERENCE_BY_WINNER_MODE,
        help="Mode used when feature_analysis.py produced the input statistics.",
    )
    parser.add_argument(
        "--transitions",
        nargs="+",
        default=DEFAULT_TRANSITIONS,
        choices=list(core.LABEL_TRANSITIONS_ORDERED),
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=DEFAULT_FEATURES,
        choices=ac.FEATURE_ANALYSIS_COLS,
    )
    parser.add_argument("--log-note", default="")
    parser.add_argument("--skip-log", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    original_root = args.original_analysis_root.resolve()
    filtered_root = args.filtered_analysis_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    original = load_run_stats(
        original_root,
        "original",
        args.transitions,
        args.features,
        args.h0_function,
        args.h1_function,
        args.inferred_feature_mode,
    )
    filtered = load_run_stats(
        filtered_root,
        "filtered",
        args.transitions,
        args.features,
        args.h0_function,
        args.h1_function,
        args.inferred_feature_mode,
    )

    combined = pd.concat([original, filtered], ignore_index=True)
    combined["welch_p_fdr_bh"] = benjamini_hochberg(combined["welch_p_value"])

    combined_file = output_dir / "feature_analysis_stats_original_and_filtered.csv"
    combined.to_csv(combined_file, index=False)

    comparison = build_comparison(combined)
    comparison_file = output_dir / "feature_analysis_original_vs_filtered_comparison.csv"
    comparison.to_csv(comparison_file, index=False)

    effect_plot = plot_effect_size_comparison(comparison, output_dir)
    p_plot = plot_p_value_comparison(comparison, output_dir)

    print(f"[✓] Compared {len(comparison)} transition-feature combinations")
    print(f"[✓] Combined statistics: {combined_file}")
    print(f"[✓] Before/after comparison: {comparison_file}")
    print(f"[✓] Effect-size plot: {effect_plot}")
    print(f"[✓] Welch-p plot: {p_plot}")

    if not args.skip_log:
        notes = (
            "This script compares statistics already produced by feature_analysis.py; it does not rerun the tests. "
            "Absolute effect-size reduction is used as the primary indicator that Constant- and Linear-selected "
            "families became more similar after filtering; p-value changes are reported separately because the "
            "filtered dataset has fewer families."
        )
        if args.log_note:
            notes += f" Run note: {args.log_note}"

        log_run(
            step="analysis",
            script=Path(__file__),
            params={
                "original_analysis_root": str(original_root),
                "filtered_analysis_root": str(filtered_root),
                "output_dir": str(output_dir),
                "h0_model": args.h0_function,
                "h1_model": args.h1_function,
                "inferred_feature_mode": args.inferred_feature_mode,
                "transitions": args.transitions,
                "features": args.features,
            },
            outputs=[str(combined_file), str(comparison_file), str(effect_plot), str(p_plot)],
            description="Compare original vs filtered H0/H1 feature-analysis results.",
            notes=notes,
            log_relative_path=Path("feature_analysis") / "original_vs_filtered_comparison.log",
        )


if __name__ == "__main__":
    main()