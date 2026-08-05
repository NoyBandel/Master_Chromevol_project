import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from source_code.constants import LABEL_DUPL, LABEL_GAIN, LABEL_LINEAR, LABEL_LOSS
from source_code.analysis.analysis_constants import (
    BASELINE_ANALYSIS_DIR,
    CHOSEN_FUNCTION_LABEL_COL,
    LIN_SLOPE_P2_COL,
    NEGATIVE_SLOPE_LABEL,
    POSITIVE_SLOPE_LABEL,
    SLOPE_SIGN_COLOR_MAP,
)
from source_code.analysis.plot_utils import ensure_dir, save_close


VALID_TRANSITIONS = (LABEL_GAIN, LABEL_LOSS, LABEL_DUPL)
SLOPE_ANALYSIS_DIR_NAME = "slope_analysis"

def resolve_input_file(transition: str, user_file: Optional[Path]) -> Path:
    if user_file is not None:
        if not user_file.exists():
            raise FileNotFoundError(f"Input file not found: {user_file}")
        return user_file

    candidate_files = [
        BASELINE_ANALYSIS_DIR / transition / f"{transition}_chosen_model_table.csv",
        BASELINE_ANALYSIS_DIR / transition / SLOPE_ANALYSIS_DIR_NAME / f"{transition}_slope_analysis.csv",
        BASELINE_ANALYSIS_DIR / transition / SLOPE_ANALYSIS_DIR_NAME / f"{transition}_slope_analysis_table.csv",
    ]

    for candidate_file in candidate_files:
        if candidate_file.exists():
            return candidate_file

    candidates_text = "\n".join(f"  - {path}" for path in candidate_files)
    raise FileNotFoundError(
        "Could not find a default input table. Tried:\n"
        f"{candidates_text}\n"
        "Pass the file explicitly with --slope-analysis-file."
    )


def validate_input_table(df: pd.DataFrame, input_file: Path) -> None:
    required_cols = {CHOSEN_FUNCTION_LABEL_COL, LIN_SLOPE_P2_COL}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing required columns in {input_file}: {sorted(missing_cols)}. "
            f"Required columns are: {sorted(required_cols)}"
        )


def get_linear_chosen_slopes(df: pd.DataFrame) -> pd.Series:
    linear_df = df.loc[df[CHOSEN_FUNCTION_LABEL_COL] == LABEL_LINEAR].copy()
    slopes = pd.to_numeric(linear_df[LIN_SLOPE_P2_COL], errors="coerce").dropna()

    if slopes.empty:
        raise ValueError(
            f"No valid '{LIN_SLOPE_P2_COL}' values found for rows where "
            f"{CHOSEN_FUNCTION_LABEL_COL} == '{LABEL_LINEAR}'."
        )

    return slopes


def build_summary_table(slopes: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "n": int(len(slopes)),
                "n_negative": int((slopes < 0).sum()),
                "n_positive": int((slopes > 0).sum()),
                "n_zero": int((slopes == 0).sum()),
                "mean": float(slopes.mean()),
                "median": float(slopes.median()),
                "std": float(slopes.std()),
                "min": float(slopes.min()),
                "q25": float(slopes.quantile(0.25)),
                "q75": float(slopes.quantile(0.75)),
                "max": float(slopes.max()),
            }
        ]
    )


def plot_linear_chosen_slope_histogram(
    slopes: pd.Series,
    transition: str,
    out_file: Path,
    bins: int,
) -> None:
    n_total = len(slopes)
    median_value = float(slopes.median())
    n_negative = int((slopes < 0).sum())
    n_positive = int((slopes > 0).sum())
    n_zero = int((slopes == 0).sum())

    negative_slopes = slopes[slopes < 0]
    positive_or_zero_slopes = slopes[slopes >= 0]

    fig, ax = plt.subplots(figsize=(8, 6))

    if len(negative_slopes) > 0:
        ax.hist(
            negative_slopes,
            bins=bins,
            alpha=0.75,
            edgecolor="black",
            color=SLOPE_SIGN_COLOR_MAP[NEGATIVE_SLOPE_LABEL],
            label=f"{NEGATIVE_SLOPE_LABEL} (n={len(negative_slopes)})",
        )

    if len(positive_or_zero_slopes) > 0:
        ax.hist(
            positive_or_zero_slopes,
            bins=bins,
            alpha=0.75,
            edgecolor="black",
            color=SLOPE_SIGN_COLOR_MAP[POSITIVE_SLOPE_LABEL],
            label=f"{POSITIVE_SLOPE_LABEL} / zero (n={len(positive_or_zero_slopes)})",
        )

    ax.axvline(0, color="black", linestyle="--", linewidth=1.2, label="Zero")
    ax.axvline(
        median_value,
        color="black",
        linestyle="-",
        linewidth=1.8,
        label=f"Median = {median_value:.3g}",
    )

    summary_text = (
        f"n = {n_total}\n"
        f"negative = {n_negative}\n"
        f"positive = {n_positive}\n"
        f"zero = {n_zero}"
    )
    ax.text(
        0.98,
        0.98,
        summary_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.9),
    )

    ax.set_title(f"{transition}: slope distribution among linear-chosen families")
    ax.set_xlabel("linear slope (p2)")
    ax.set_ylabel("Number of families")
    ax.legend()

    save_close(fig, out_file)




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=("Plot the distribution of fitted linear slope values only for families where the chosen model is linear."))
    parser.add_argument("--transition_label", required=True, choices=VALID_TRANSITIONS, help="Transition to analyze: gain / loss / dupl.")
    parser.add_argument("--slope-analysis-file", type=Path, default=None, help=("Optional input CSV. Recommended input: analysis/baseline_models/<transition>/<transition>_chosen_model_table.csv"))
    parser.add_argument("--output-dir", type=Path, default=None, help=( "Optional output directory.  Default: analysis/baseline_models/<transition>/slope_analysis"))
    parser.add_argument("--bins", type=int, default=30, help="Number of histogram bins.")
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    input_file = resolve_input_file(args.transition_label, args.slope_analysis_file)
    output_dir = args.output_dir or (
        BASELINE_ANALYSIS_DIR / args.transition / SLOPE_ANALYSIS_DIR_NAME
    )
    ensure_dir(output_dir)

    df = pd.read_csv(input_file)
    validate_input_table(df, input_file)

    slopes = get_linear_chosen_slopes(df)

    out_png = output_dir / f"{args.transition}_linear_chosen_slope_histogram.png"
    out_summary = output_dir / f"{args.transition}_linear_chosen_slope_summary.csv"

    plot_linear_chosen_slope_histogram(
        slopes=slopes,
        transition=args.transition,
        out_file=out_png,
        bins=args.bins,
    )
    build_summary_table(slopes).to_csv(out_summary, index=False)

    print("[✓] Linear-chosen slope histogram completed")
    print(f"Input: {input_file}")
    print(f"Plot: {out_png}")
    print(f"Summary: {out_summary}")


if __name__ == "__main__":
    main()