"""
Analyze fitted LINEAR functions in the baseline ChromEvol analysis.

Outputs:
1. Slope histogram for all retained families with a fitted LINEAR model.
2. Slope histogram for families where LINEAR was selected.
3. Positive/negative slope-direction pie for LINEAR-selected families.
4. Slope summary table.
5. Overlay of all LINEAR-selected fitted functions.

The user explicitly chooses whether to use raw ChromEvol parameters or
tree-scale-corrected parameters.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from source_code.analysis.analysis_constants import CHOSEN_FUNCTION_LABEL_COL, LIN_P1_COL, LIN_SLOPE_P2_COL, NEGATIVE_SLOPE_LABEL, POSITIVE_SLOPE_LABEL, SLOPE_SIGN_COLOR_MAP
from source_code.analysis.parameter_utils import VALID_PARAMETER_SCALES, resolve_parameter_column
from source_code.analysis.plot_utils import compute_linear_rate, plot_direction_colored_function_overlay
from source_code.constants import FAMILY_NAME_COL, LABEL_DUPL, LABEL_GAIN, LABEL_LINEAR, LABEL_LOSS, MAX_CHROM_COL, MIN_CHROM_COL
from source_code.logger import log_run


VALID_TRANSITIONS: tuple[str, ...] = (LABEL_GAIN, LABEL_LOSS, LABEL_DUPL)
DEFAULT_SLOPE_HIST_BIN_WIDTH: float = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze fitted LINEAR slopes and functions in the baseline ChromEvol analysis.")
    parser.add_argument("--transition", required=True, choices=VALID_TRANSITIONS, help="Transition to analyze: gain / loss / dupl.")
    parser.add_argument("--chosen-model-file", type=Path, required=True, help="Chosen-model table containing fitted LINEAR parameters.")
    parser.add_argument("--metadata-file", type=Path, required=True, help="Family metadata table containing min_chrom and max_chrom.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory in which to save slope-analysis outputs.")
    parser.add_argument("--parameter-scale", required=True, choices=VALID_PARAMETER_SCALES, help="Use raw or tree-scale-corrected fitted parameters.")
    parser.add_argument("--bin-width", type=float, default=DEFAULT_SLOPE_HIST_BIN_WIDTH, help="Slope histogram bin width.")
    parser.add_argument("--log-note", type=str, default="", help="Optional note describing this analysis run.")
    return parser.parse_args()


def load_analysis_table(chosen_model_file: Path, metadata_file: Path, parameter_scale: str) -> tuple[pd.DataFrame, str, str]:
    if not chosen_model_file.exists():
        raise FileNotFoundError(f"Chosen-model file not found: {chosen_model_file}")
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    p1_col: str = resolve_parameter_column(LIN_P1_COL, parameter_scale)
    p2_col: str = resolve_parameter_column(LIN_SLOPE_P2_COL, parameter_scale)

    chosen_df: pd.DataFrame = pd.read_csv(chosen_model_file)
    metadata_df: pd.DataFrame = pd.read_csv(metadata_file)

    chosen_required: list[str] = [FAMILY_NAME_COL, CHOSEN_FUNCTION_LABEL_COL, p1_col, p2_col]
    metadata_required: list[str] = [FAMILY_NAME_COL, MIN_CHROM_COL, MAX_CHROM_COL]

    missing_chosen: list[str] = [column for column in chosen_required if column not in chosen_df.columns]
    missing_metadata: list[str] = [column for column in metadata_required if column not in metadata_df.columns]

    if missing_chosen:
        raise KeyError(f"Missing required columns in {chosen_model_file}: {missing_chosen}")
    if missing_metadata:
        raise KeyError(f"Missing required columns in {metadata_file}: {missing_metadata}")
    if chosen_df[FAMILY_NAME_COL].duplicated().any():
        raise ValueError(f"Duplicate families found in chosen-model table: {chosen_model_file}")
    if metadata_df[FAMILY_NAME_COL].duplicated().any():
        raise ValueError(f"Duplicate families found in metadata table: {metadata_file}")

    df: pd.DataFrame = chosen_df.merge(metadata_df[metadata_required], on=FAMILY_NAME_COL, how="left", validate="one_to_one")

    for column in [p1_col, p2_col, MIN_CHROM_COL, MAX_CHROM_COL]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df, p1_col, p2_col


def build_slope_sets(df: pd.DataFrame, p2_col: str) -> dict[str, pd.DataFrame]:
    return {
        "all_slopes": df.dropna(subset=[p2_col]).copy(),
        "chosen_linear_slopes": df[(df[CHOSEN_FUNCTION_LABEL_COL] == LABEL_LINEAR) & df[p2_col].notna()].copy(),
    }


def build_slope_bins(values: pd.Series, bin_width: float) -> np.ndarray:
    if bin_width <= 0:
        raise ValueError("--bin-width must be greater than 0.")

    bin_start: float = np.floor(values.min() / bin_width) * bin_width
    bin_end: float = np.ceil(values.max() / bin_width) * bin_width

    if bin_start == bin_end:
        bin_start -= bin_width
        bin_end += bin_width

    return np.arange(bin_start, bin_end + bin_width, bin_width)


def plot_slope_histogram(df: pd.DataFrame, p2_col: str, transition: str, slope_set_label: str, parameter_scale: str, bin_width: float, output_file: Path) -> None:
    values: pd.Series = pd.to_numeric(df[p2_col], errors="coerce").dropna()
    negative_values: pd.Series = values[values < 0]
    positive_values: pd.Series = values[values > 0]
    zero_values: pd.Series = values[values == 0]
    bin_edges: np.ndarray = build_slope_bins(values, bin_width)

    fig, ax = plt.subplots(figsize=(9, 6))

    if not negative_values.empty:
        ax.hist(negative_values, bins=bin_edges, alpha=0.7, color=SLOPE_SIGN_COLOR_MAP[NEGATIVE_SLOPE_LABEL], label=f"negative (n={len(negative_values)})")

    if not positive_values.empty:
        ax.hist(positive_values, bins=bin_edges, alpha=0.7, color=SLOPE_SIGN_COLOR_MAP[POSITIVE_SLOPE_LABEL], label=f"positive (n={len(positive_values)})")

    zero_label: str = f"zero (n={len(zero_values)})" if not zero_values.empty else None
    ax.axvline(0, color="black", linestyle="--", linewidth=1, label=zero_label)
    ax.axvline(float(values.median()), color="black", linestyle="-", linewidth=1.5, label=f"median = {values.median():.3g}")

    ax.set_title(f"{transition}: slope distribution ({slope_set_label}, n={len(values)})")
    ax.set_xlabel("Tree-scale-corrected linear slope (p2)" if parameter_scale == "corrected" else "Raw linear slope (p2)")
    ax.set_ylabel("Number of families")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_slope_direction_pie(chosen_linear_df: pd.DataFrame, p2_col: str, transition: str, parameter_scale: str, output_file: Path) -> None:
    slopes: pd.Series = pd.to_numeric(chosen_linear_df[p2_col], errors="coerce").dropna()
    n_positive: int = int((slopes > 0).sum())
    n_negative: int = int((slopes < 0).sum())
    n_zero: int = int((slopes == 0).sum())

    labels: list[str] = [f"Positive (n={n_positive})", f"Negative (n={n_negative})"]
    values: list[int] = [n_positive, n_negative]
    colors: list[str] = [SLOPE_SIGN_COLOR_MAP[POSITIVE_SLOPE_LABEL], SLOPE_SIGN_COLOR_MAP[NEGATIVE_SLOPE_LABEL]]

    if n_zero > 0:
        labels.append(f"Zero (n={n_zero})")
        values.append(n_zero)
        colors.append("gray")

    if sum(values) == 0:
        raise ValueError(f"No valid LINEAR-chosen slopes available for transition '{transition}'.")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(values, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90, counterclock=False, wedgeprops={"edgecolor": "black"})
    ax.set_title(f"{transition}: slope direction among LINEAR-chosen families\n({parameter_scale} parameters, n={len(slopes)})")

    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_slope_summary(slope_sets: dict[str, pd.DataFrame], p2_col: str, parameter_scale: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for slope_set_label, df in slope_sets.items():
        slopes: pd.Series = pd.to_numeric(df[p2_col], errors="coerce").dropna()

        rows.append({
            "slope_set": slope_set_label,
            "parameter_scale": parameter_scale,
            "slope_column": p2_col,
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
        })

    return pd.DataFrame(rows)


def plot_linear_function_overlay(df: pd.DataFrame, p1_col: str, p2_col: str, transition: str, parameter_scale: str, output_file: Path) -> int:
    linear_df: pd.DataFrame = df[df[CHOSEN_FUNCTION_LABEL_COL] == LABEL_LINEAR].copy()
    required_numeric: list[str] = [p1_col, p2_col, MIN_CHROM_COL, MAX_CHROM_COL]
    missing_df: pd.DataFrame = linear_df[linear_df[required_numeric].isna().any(axis=1)]

    if not missing_df.empty:
        missing_families: list[str] = missing_df[FAMILY_NAME_COL].astype(str).tolist()
        raise ValueError(f"Missing fitted parameters or chromosome ranges for LINEAR-chosen families: {missing_families}")

    linear_df = linear_df[linear_df[p2_col] != 0].copy()

    if linear_df.empty:
        raise ValueError(f"No plottable LINEAR-chosen functions found for transition '{transition}'.")

    plot_direction_colored_function_overlay(
        df=linear_df,
        family_col=FAMILY_NAME_COL,
        p1_col=p1_col,
        p2_col=p2_col,
        min_chrom_col=MIN_CHROM_COL,
        max_chrom_col=MAX_CHROM_COL,
        rate_function=compute_linear_rate,
        direction_col=p2_col,
        title=f"{transition}: LINEAR functions selected in the baseline analysis ({parameter_scale})",
        ylabel="Inferred transition rate",
        out_file=output_file,
    )

    return len(linear_df)


def main() -> None:
    args: argparse.Namespace = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    analysis_df, p1_col, p2_col = load_analysis_table(args.chosen_model_file, args.metadata_file, args.parameter_scale)
    slope_sets: dict[str, pd.DataFrame] = build_slope_sets(analysis_df, p2_col)
    chosen_linear_df: pd.DataFrame = slope_sets["chosen_linear_slopes"]

    all_histogram_file: Path = args.output_dir / f"{args.transition}_slope_histogram_all_slopes_{args.parameter_scale}.png"
    chosen_histogram_file: Path = args.output_dir / f"{args.transition}_slope_histogram_chosen_linear_slopes_{args.parameter_scale}.png"
    direction_pie_file: Path = args.output_dir / f"{args.transition}_slope_direction_pie_chosen_linear_{args.parameter_scale}.png"
    summary_file: Path = args.output_dir / f"{args.transition}_slope_summary_{args.parameter_scale}.csv"
    overlay_file: Path = args.output_dir / f"{args.transition}_chosen_linear_functions_overlay_{args.parameter_scale}.png"

    plot_slope_histogram(slope_sets["all_slopes"], p2_col, args.transition, "all fitted LINEAR slopes", args.parameter_scale, args.bin_width, all_histogram_file)
    plot_slope_histogram(chosen_linear_df, p2_col, args.transition, "LINEAR-chosen slopes", args.parameter_scale, args.bin_width, chosen_histogram_file)
    plot_slope_direction_pie(chosen_linear_df, p2_col, args.transition, args.parameter_scale, direction_pie_file)
    build_slope_summary(slope_sets, p2_col, args.parameter_scale).to_csv(summary_file, index=False)
    n_overlay_families: int = plot_linear_function_overlay(analysis_df, p1_col, p2_col, args.transition, args.parameter_scale, overlay_file)

    output_files: list[Path] = [all_histogram_file, chosen_histogram_file, direction_pie_file, summary_file, overlay_file]

    notes: str = f"Generated baseline LINEAR slope histograms, slope-direction pie, descriptive summary, and LINEAR-function overlay. Parameter scale used: {args.parameter_scale}."
    if args.log_note:
        notes += f"\nRun note: {args.log_note}"

    log_run(
        step="baseline_slope_analysis",
        script=Path(__file__),
        params={
            "transition": args.transition,
            "parameter_scale": args.parameter_scale,
            "p1_column": p1_col,
            "p2_column": p2_col,
            "chosen_model_file": str(args.chosen_model_file),
            "metadata_file": str(args.metadata_file),
            "output_dir": str(args.output_dir),
            "bin_width": args.bin_width,
            "n_all_fitted_slopes": len(slope_sets["all_slopes"]),
            "n_linear_chosen": len(chosen_linear_df),
            "n_overlay_families": n_overlay_families,
        },
        outputs=[str(output_file) for output_file in output_files],
        description=f"Analyzed fitted LINEAR slopes and functions for baseline transition '{args.transition}'.",
        notes=notes,
        log_relative_path=Path("baseline_models") / "slope_analysis" / f"{args.transition}.log",
    )

    print(f"[✓] Finished baseline slope analysis for '{args.transition}'.")
    print(f"Parameter scale: {args.parameter_scale}")
    print(f"p1 column: {p1_col}")
    print(f"p2 column: {p2_col}")
    print(f"All fitted slopes: {len(slope_sets['all_slopes'])}")
    print(f"LINEAR-chosen families: {len(chosen_linear_df)}")
    print(f"Overlay families: {n_overlay_families}")

    for output_file in output_files:
        print(output_file)


if __name__ == "__main__":
    main()