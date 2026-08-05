import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnchoredText
from scipy.stats import ttest_ind, ttest_rel

from source_code.analysis.analysis_constants import *
from source_code.logger import log_run
from matplotlib.ticker import MaxNLocator


# ==================================================================================== #
# Column name resolution
# ==================================================================================== #
FAMILY_COL_NAME = globals().get("FAMILY_NAME_COL", "family_name")
FUNCTION_LABEL_COL_NAME = globals().get("FUNCTION_LABEL_COL", "function_label")
CONFIGURATION_COL_NAME = globals().get("CONFIGURATION_COL", "configuration")
LABEL_TESTED_TRANSITION_COL_NAME = globals().get("LABEL_TESTED_TRANSITION_COL", "label_tested_transition")

CONST_VALUE_COL_NAME = globals().get("CONST_VAL_COL", "constant_value")
LIN_P1_COL_NAME = globals().get("LIN_P1_COL", "lin_p1")
LIN_SLOPE_P2_COL_NAME = globals().get("LIN_SLOPE_P2_COL", "lin_slope_p2")

FAMILY_SIZE_COL_NAME = globals().get("FAMILY_SIZE_COL", "family_size")
MIN_CHROM_COL_NAME = globals().get("MIN_CHROM_COL", "min_chrom")
MAX_CHROM_COL_NAME = globals().get("MAX_CHROM_COL", "max_chrom")
DIFF_COL_NAME = globals().get("DIFF_COL", "chrom_range")

ROOT_CHROM_NUM_COL_NAME = globals().get("ROOT_CHROM_NUM_COL", "root_chrom_num")
NUM_OF_EVENTS_COL_NAME = globals().get("NUM_OF_EVENTS_COL", "num_of_events")


# ==================================================================================== #
# Defaults
# ==================================================================================== #
DEFAULT_METADATA_FILE = Path("/groups/itay_mayrose/noybandel/Master_ChromEvol_project/input_data/families_for_analysis_metadata.csv")
DEFAULT_CHOSEN_MODELS_DIR = Path("/groups/itay_mayrose/noybandel/Master_ChromEvol_project/chromevol_parsed_results/model_selection")
DEFAULT_OUTPUT_DIR = Path("/groups/itay_mayrose/noybandel/Master_ChromEvol_project/source_code/analysis/baseline_models/feature_analysis_outputs")

METADATA_FEATURES = [
    (FAMILY_SIZE_COL_NAME, "Family size"),
    (MIN_CHROM_COL_NAME, "Minimum chromosome number"),
    (MAX_CHROM_COL_NAME, "Maximum chromosome number"),
    (DIFF_COL_NAME, "Chromosome range"),
]

INFERRED_FEATURE_SPECS = [
    (ROOT_CHROM_NUM_COL_NAME, "Root chromosome number"),
    (NUM_OF_EVENTS_COL_NAME, "Number of events"),
]


# ==================================================================================== #
# IO helpers
# ==================================================================================== #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze metadata and inferred features for constant vs linear models.")
    parser.add_argument("--transition", required=True, help="Transition label, e.g. gain / loss / dupl / demi / baseNum")
    parser.add_argument("--metadata_file", type=Path, default=DEFAULT_METADATA_FILE)
    parser.add_argument("--chosen_models_csv", type=Path, default=None)
    parser.add_argument("--model_summary_csv", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--slope_analysis_csv", type=Path,default=None, help="Baseline slope-analysis table used for slope histograms.")
    parser.add_argument("--slope_histograms_only",action="store_true", help="Generate only the all-LINEAR and chosen-LINEAR slope histograms.")
    return parser.parse_args()


def resolve_default_paths(transition: str, chosen_models_csv: Optional[Path], model_summary_csv: Optional[Path]) -> Tuple[Path, Path]:
    if chosen_models_csv is None:
        chosen_models_csv = DEFAULT_CHOSEN_MODELS_DIR / f"{transition}_chosen_models.csv"

    if model_summary_csv is None:
        model_summary_csv = DEFAULT_CHOSEN_MODELS_DIR / f"{transition}_models_summary_table.csv"

    return chosen_models_csv, model_summary_csv


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_csv(file_path: Path, description: str) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"{description} file not found: {file_path}")
    return pd.read_csv(file_path)


# ==================================================================================== #
# Column utilities
# ==================================================================================== #
def resolve_column(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise KeyError(f"Could not find any of these columns: {candidates}")
    return None


def find_inferred_feature_columns(df: pd.DataFrame, feature_base_name: str) -> Tuple[str, str]:
    constant_candidates = [
        f"constant_{feature_base_name}",
        f"{feature_base_name}_constant",
        f"{feature_base_name}_const",
        f"const_{feature_base_name}",
        f"constant_model_{feature_base_name}",
    ]
    linear_candidates = [
        f"linear_{feature_base_name}",
        f"{feature_base_name}_linear",
        f"{feature_base_name}_lin",
        f"lin_{feature_base_name}",
        f"linear_model_{feature_base_name}",
    ]

    constant_col = resolve_column(df, constant_candidates, required=False)
    linear_col = resolve_column(df, linear_candidates, required=False)

    if constant_col is None or linear_col is None:
        raise KeyError(
            f"Could not find inferred-feature columns for '{feature_base_name}'. "
            f"Tried constant candidates {constant_candidates} and linear candidates {linear_candidates}"
        )

    return constant_col, linear_col


# ==================================================================================== #
# Stats helpers
# ==================================================================================== #
def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").dropna()


def cohen_d_independent(group1: pd.Series, group2: pd.Series) -> float:
    x1 = safe_numeric(group1)
    x2 = safe_numeric(group2)

    n1 = len(x1)
    n2 = len(x2)

    if n1 < 2 or n2 < 2:
        return np.nan

    s1 = x1.std(ddof=1)
    s2 = x2.std(ddof=1)

    pooled_sd = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled_sd == 0:
        return np.nan

    return (x2.mean() - x1.mean()) / pooled_sd


def paired_cohen_d(series1: pd.Series, series2: pd.Series) -> float:
    x1 = pd.to_numeric(series1, errors="coerce")
    x2 = pd.to_numeric(series2, errors="coerce")

    paired = pd.DataFrame({"x1": x1, "x2": x2}).dropna()
    if len(paired) < 2:
        return np.nan

    diffs = paired["x2"] - paired["x1"]
    diff_sd = diffs.std(ddof=1)
    if diff_sd == 0:
        return np.nan

    return diffs.mean() / diff_sd


def format_float(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.{digits}f}"


def welch_stats_text(group_constant: pd.Series, group_linear: pd.Series) -> str:
    x_const = safe_numeric(group_constant)
    x_lin = safe_numeric(group_linear)

    if len(x_const) < 2 or len(x_lin) < 2:
        return (
            f"n constant = {len(x_const)}\n"
            f"n linear = {len(x_lin)}\n"
            f"Welch t-test p = NA\n"
            f"Cohen's d = NA"
        )

    _, p_value = ttest_ind(x_const, x_lin, equal_var=False, nan_policy="omit")
    d_value = cohen_d_independent(x_const, x_lin)

    return (
        f"n constant = {len(x_const)}\n"
        f"n linear = {len(x_lin)}\n"
        f"Welch t-test p = {format_float(p_value, 4)}\n"
        f"Cohen's d = {format_float(d_value, 3)}"
    )


def paired_stats_text(series_constant: pd.Series, series_linear: pd.Series) -> str:
    x_const = pd.to_numeric(series_constant, errors="coerce")
    x_lin = pd.to_numeric(series_linear, errors="coerce")

    paired = pd.DataFrame({"constant": x_const, "linear": x_lin}).dropna()
    if len(paired) < 2:
        return (
            f"paired n = {len(paired)}\n"
            f"Paired t-test p = NA\n"
            f"Paired Cohen's d = NA"
        )

    _, p_value = ttest_rel(paired["constant"], paired["linear"], nan_policy="omit")
    d_value = paired_cohen_d(paired["constant"], paired["linear"])

    return (
        f"paired n = {len(paired)}\n"
        f"Paired t-test p = {format_float(p_value, 4)}\n"
        f"Paired Cohen's d = {format_float(d_value, 3)}"
    )


# ==================================================================================== #
# Plot helpers
# ==================================================================================== #
def add_top_right_left_aligned_box(ax: plt.Axes, text: str) -> None:
    anchored = AnchoredText(
        text,
        loc="upper right",
        prop={"size": 9},
        frameon=True,
        borderpad=0.6,
    )
    anchored.patch.set_alpha(0.9)
    anchored.patch.set_facecolor("white")
    anchored.patch.set_edgecolor("gray")
    ax.add_artist(anchored)


def prepare_hist_bins(values: pd.Series) -> int:
    clean = safe_numeric(values)
    if len(clean) < 2:
        return 10
    return min(20, max(8, int(np.sqrt(len(clean)))))


def save_histogram_two_groups(
    group_constant: pd.Series,
    group_linear: pd.Series,
    x_label: str,
    title: str,
    output_file: Path,
    stats_text: str,
) -> None:
    x_const = safe_numeric(group_constant)
    x_lin = safe_numeric(group_linear)

    fig, ax = plt.subplots(figsize=(8, 5))

    all_values = pd.concat([x_const, x_lin], ignore_index=True)
    bins = prepare_hist_bins(all_values)

    ax.hist(x_const, bins=bins, alpha=0.6, label="Chosen constant")
    ax.hist(x_lin, bins=bins, alpha=0.6, label="Chosen linear")

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Number of families")
    ax.legend()

    add_top_right_left_aligned_box(ax, stats_text)

    plt.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_boxplot_two_groups(
    group_constant: pd.Series,
    group_linear: pd.Series,
    y_label: str,
    title: str,
    output_file: Path,
    stats_text: str,
) -> None:
    x_const = safe_numeric(group_constant)
    x_lin = safe_numeric(group_linear)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot(
        [x_const, x_lin],
        labels=["Chosen constant", "Chosen linear"],
        patch_artist=False,
    )

    ax.set_title(title)
    ax.set_ylabel(y_label)

    add_top_right_left_aligned_box(ax, stats_text)

    plt.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_paired_scatter(
    df: pd.DataFrame,
    constant_col: str,
    linear_col: str,
    y_label: str,
    title: str,
    output_file: Path,
    stats_text: str,
) -> None:
    paired = df[[constant_col, linear_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if paired.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    x_positions = [0, 1]
    for _, row in paired.iterrows():
        ax.plot(x_positions, [row[constant_col], row[linear_col]], alpha=0.25, linewidth=1)
        ax.scatter(x_positions, [row[constant_col], row[linear_col]], s=20)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(["Constant model value", "Linear model value"])
    ax.set_ylabel(y_label)
    ax.set_title(title)

    add_top_right_left_aligned_box(ax, stats_text)

    plt.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_count_barplot(summary_df: pd.DataFrame, output_file: Path, transition: str) -> None:
    counts = summary_df[FUNCTION_LABEL_COL_NAME].value_counts().reindex(["constant", "linear"]).fillna(0)

    fig, ax = plt.subplots(figsize=(5, 5))
    bars = ax.bar(counts.index, counts.values)

    ax.set_title(f"Chosen model counts for {transition}")
    ax.set_xlabel("Chosen model")
    ax.set_ylabel("Number of families")

    total = counts.sum()
    for bar, count in zip(bars, counts.values):
        percentage = 100 * count / total if total > 0 else 0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(count)}\n({percentage:.1f}%)",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ==================================================================================== #
# Data preparation
# ==================================================================================== #
def load_chosen_models(chosen_models_csv: Path, transition: str) -> pd.DataFrame:
    df = load_csv(chosen_models_csv, "Chosen models")

    required_cols = [FAMILY_COL_NAME, FUNCTION_LABEL_COL_NAME]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column '{col}' in {chosen_models_csv}")

    df = df[df[FUNCTION_LABEL_COL_NAME].isin(["constant", "linear"])].copy()

    if LABEL_TESTED_TRANSITION_COL_NAME in df.columns:
        df = df[df[LABEL_TESTED_TRANSITION_COL_NAME] == transition].copy()

    if df.empty:
        raise ValueError(f"No constant/linear chosen-model rows found for transition '{transition}'")

    return df


def load_metadata(metadata_file: Path) -> pd.DataFrame:
    df = load_csv(metadata_file, "Metadata")

    if FAMILY_COL_NAME not in df.columns:
        raise KeyError(f"Missing required metadata family column '{FAMILY_COL_NAME}'")

    return df


def build_metadata_analysis_df(chosen_df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    merged = chosen_df[[FAMILY_COL_NAME, FUNCTION_LABEL_COL_NAME]].merge(
        metadata_df,
        on=FAMILY_COL_NAME,
        how="left",
    )
    return merged


def load_model_summary(model_summary_csv: Path, transition: str) -> pd.DataFrame:
    df = load_csv(model_summary_csv, "Model summary")

    if FAMILY_COL_NAME not in df.columns:
        raise KeyError(f"Missing required column '{FAMILY_COL_NAME}' in {model_summary_csv}")
    if FUNCTION_LABEL_COL_NAME not in df.columns:
        raise KeyError(f"Missing required column '{FUNCTION_LABEL_COL_NAME}' in {model_summary_csv}")

    df = df[df[FUNCTION_LABEL_COL_NAME].isin(["constant", "linear"])].copy()

    if LABEL_TESTED_TRANSITION_COL_NAME in df.columns:
        df = df[df[LABEL_TESTED_TRANSITION_COL_NAME] == transition].copy()

    if df.empty:
        raise ValueError(f"No constant/linear rows found in model summary for transition '{transition}'")

    return df


def build_inferred_analysis_df(chosen_df: pd.DataFrame, model_summary_df: pd.DataFrame) -> pd.DataFrame:
    chosen_small = chosen_df[[FAMILY_COL_NAME, FUNCTION_LABEL_COL_NAME]].drop_duplicates()

    pivot = model_summary_df.pivot_table(
        index=FAMILY_COL_NAME,
        columns=FUNCTION_LABEL_COL_NAME,
        values=[ROOT_CHROM_NUM_COL_NAME, NUM_OF_EVENTS_COL_NAME],
        aggfunc="first",
    )

    pivot.columns = [f"{model}_{feature}" for feature, model in pivot.columns]
    pivot = pivot.reset_index()

    merged = chosen_small.merge(pivot, on=FAMILY_COL_NAME, how="inner")

    expected_columns = [
        FAMILY_COL_NAME,
        FUNCTION_LABEL_COL_NAME,
        f"constant_{ROOT_CHROM_NUM_COL_NAME}",
        f"linear_{ROOT_CHROM_NUM_COL_NAME}",
        f"constant_{NUM_OF_EVENTS_COL_NAME}",
        f"linear_{NUM_OF_EVENTS_COL_NAME}",
    ]
    missing = [col for col in expected_columns if col not in merged.columns]
    if missing:
        raise KeyError(
            f"Could not build inferred feature table. Missing columns after pivot: {missing}"
        )

    return merged


# ==================================================================================== #
# Analysis runners
# ==================================================================================== #
def run_metadata_analysis(analysis_df: pd.DataFrame, output_dir: Path, transition: str) -> List[Path]:
    saved_files = []

    for feature_col, feature_label in METADATA_FEATURES:
        if feature_col not in analysis_df.columns:
            continue

        chosen_constant = analysis_df.loc[analysis_df[FUNCTION_LABEL_COL_NAME] == "constant", feature_col]
        chosen_linear = analysis_df.loc[analysis_df[FUNCTION_LABEL_COL_NAME] == "linear", feature_col]

        stats_text = welch_stats_text(chosen_constant, chosen_linear)

        hist_file = output_dir / f"{transition}_{feature_col}_metadata_histogram.png"
        box_file = output_dir / f"{transition}_{feature_col}_metadata_boxplot.png"

        save_histogram_two_groups(
            group_constant=chosen_constant,
            group_linear=chosen_linear,
            x_label=feature_label,
            title=f"{feature_label}: chosen constant vs chosen linear",
            output_file=hist_file,
            stats_text=stats_text,
        )

        save_boxplot_two_groups(
            group_constant=chosen_constant,
            group_linear=chosen_linear,
            y_label=feature_label,
            title=f"{feature_label}: chosen constant vs chosen linear",
            output_file=box_file,
            stats_text=stats_text,
        )

        saved_files.extend([hist_file, box_file])

    return saved_files


def run_inferred_analysis(analysis_df: pd.DataFrame, output_dir: Path, transition: str) -> List[Path]:
    saved_files = []

    for feature_col, feature_label in INFERRED_FEATURE_SPECS:
        constant_model_col = f"constant_{feature_col}"
        linear_model_col = f"linear_{feature_col}"

        if constant_model_col not in analysis_df.columns or linear_model_col not in analysis_df.columns:
            continue

        chosen_constant_df = analysis_df[analysis_df[FUNCTION_LABEL_COL_NAME] == "constant"].copy()
        chosen_linear_df = analysis_df[analysis_df[FUNCTION_LABEL_COL_NAME] == "linear"].copy()

        constant_value_hist_text = welch_stats_text(
            chosen_constant_df[constant_model_col],
            chosen_linear_df[constant_model_col],
        )
        linear_value_hist_text = welch_stats_text(
            chosen_constant_df[linear_model_col],
            chosen_linear_df[linear_model_col],
        )

        analysis_df[f"delta_linear_minus_constant_{feature_col}"] = (
            pd.to_numeric(analysis_df[linear_model_col], errors="coerce")
            - pd.to_numeric(analysis_df[constant_model_col], errors="coerce")
        )

        delta_col = f"delta_linear_minus_constant_{feature_col}"
        delta_hist_text = welch_stats_text(
            chosen_constant_df[delta_col],
            chosen_linear_df[delta_col],
        )

        paired_text = paired_stats_text(
            analysis_df[constant_model_col],
            analysis_df[linear_model_col],
        )

        constant_hist_file = output_dir / f"{transition}_{feature_col}_constant_model_value_by_chosen_model_histogram.png"
        linear_hist_file = output_dir / f"{transition}_{feature_col}_linear_model_value_by_chosen_model_histogram.png"
        delta_hist_file = output_dir / f"{transition}_{feature_col}_delta_linear_minus_constant_by_chosen_model_histogram.png"
        paired_scatter_file = output_dir / f"{transition}_{feature_col}_paired_constant_vs_linear.png"

        save_histogram_two_groups(
            group_constant=chosen_constant_df[constant_model_col],
            group_linear=chosen_linear_df[constant_model_col],
            x_label=f"{feature_label} under constant model",
            title=f"{feature_label} under constant model\nGrouped by chosen model",
            output_file=constant_hist_file,
            stats_text=constant_value_hist_text,
        )

        save_histogram_two_groups(
            group_constant=chosen_constant_df[linear_model_col],
            group_linear=chosen_linear_df[linear_model_col],
            x_label=f"{feature_label} under linear model",
            title=f"{feature_label} under linear model\nGrouped by chosen model",
            output_file=linear_hist_file,
            stats_text=linear_value_hist_text,
        )

        save_histogram_two_groups(
            group_constant=chosen_constant_df[delta_col],
            group_linear=chosen_linear_df[delta_col],
            x_label=f"Linear - constant ({feature_label})",
            title=f"Difference between linear and constant model values\nfor {feature_label}",
            output_file=delta_hist_file,
            stats_text=delta_hist_text,
        )

        save_paired_scatter(
            df=analysis_df,
            constant_col=constant_model_col,
            linear_col=linear_model_col,
            y_label=feature_label,
            title=f"Paired constant vs linear values for {feature_label}",
            output_file=paired_scatter_file,
            stats_text=paired_text,
        )

        saved_files.extend(
            [
                constant_hist_file,
                linear_hist_file,
                delta_hist_file,
                paired_scatter_file,
            ]
        )

    return saved_files

# -------- histograms: signed slopes --------
SLOPE_HIST_BIN_WIDTH: float = 0.01
def plot_slope_histograms(slope_df: pd.DataFrame, transition_label: str, out_dir: Path, output_paths: List[str]) -> None:
    slope_sets = {
        "all_slopes": slope_df.dropna(subset=[LIN_SLOPE_P2_COL]).copy(),
        "chosen_linear_slopes": slope_df[
            (slope_df[CHOSEN_FUNCTION_LABEL_COL] == LABEL_LINEAR)
            & slope_df[LIN_SLOPE_P2_COL].notna()
        ].copy(),
    }

    for slope_set_label, df in slope_sets.items():
        if df.empty:
            continue

        out_file = out_dir / f"{transition_label}_slope_histogram_{slope_set_label}.png"
        fig, ax = plt.subplots(figsize=(9, 6))

        pos_df = df[df[LIN_SLOPE_P2_COL] >= 0]
        neg_df = df[df[LIN_SLOPE_P2_COL] < 0]

        pos_vals: pd.Series = pd.to_numeric(pos_df[LIN_SLOPE_P2_COL], errors="coerce").dropna()
        neg_vals: pd.Series = pd.to_numeric(neg_df[LIN_SLOPE_P2_COL], errors="coerce").dropna()
        all_vals: pd.Series = pd.concat([neg_vals, pos_vals], ignore_index=True)

        bin_start: float = np.floor(all_vals.min() / SLOPE_HIST_BIN_WIDTH) * SLOPE_HIST_BIN_WIDTH
        bin_end: float = np.ceil(all_vals.max() / SLOPE_HIST_BIN_WIDTH) * SLOPE_HIST_BIN_WIDTH
        bin_edges: np.ndarray = np.arange(bin_start, bin_end + SLOPE_HIST_BIN_WIDTH, SLOPE_HIST_BIN_WIDTH)

        ax.hist(neg_vals, bins=bin_edges, alpha=0.7, color=SLOPE_SIGN_COLOR_MAP[NEGATIVE_SLOPE_LABEL],
                label=f"negative (n={len(neg_vals)})")
        ax.hist(pos_vals, bins=bin_edges, alpha=0.7, color=SLOPE_SIGN_COLOR_MAP[POSITIVE_SLOPE_LABEL],
                label=f"positive (n={len(pos_vals)})")

        ax.axvline(0, color="black", linestyle="--", linewidth=1)

        total_n: int = len(all_vals)
        ax.set_title(f"{transition_label}: slope distribution ({slope_set_label}, n={total_n})")
        ax.set_xlabel("slope")
        ax.set_ylabel("Number of families")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend()

        fig.tight_layout()
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close(fig)

        output_paths.append(str(out_file))

# ==================================================================================== #
# Main
# ==================================================================================== #
def main() -> None:
    args: argparse.Namespace = parse_args()
    output_dir: Path = args.output_dir or (DEFAULT_OUTPUT_DIR / args.transition)
    ensure_dir(output_dir)

    output_files: List[Path] = []

    if args.slope_histograms_only:
        if args.slope_analysis_csv is None:
            raise ValueError("--slope_analysis_csv is required with --slope_histograms_only.")

        slope_df: pd.DataFrame = load_csv(args.slope_analysis_csv, "Slope analysis")
        slope_output_paths: List[str] = []

        plot_slope_histograms(slope_df=slope_df, transition_label=args.transition, out_dir=output_dir, output_paths=slope_output_paths)
        output_files = [Path(path) for path in slope_output_paths]

        step: str = "baseline_slope_histograms"
        params: Dict[str, object] = {"transition": args.transition, "slope_analysis_csv": str(args.slope_analysis_csv), "output_dir": str(output_dir)}
        description: str = f"Generated baseline LINEAR slope histograms for transition {args.transition}."
        notes: str = "Generated one histogram for all fitted LINEAR slopes and one histogram restricted to baseline LINEAR-chosen families."
        log_relative_path: str = f"baseline_models/slope_analysis/{args.transition}"
        completion_message: str = f"Finished baseline slope histograms for transition '{args.transition}'"

    else:
        chosen_models_csv, model_summary_csv = resolve_default_paths(transition=args.transition, chosen_models_csv=args.chosen_models_csv, model_summary_csv=args.model_summary_csv)

        chosen_df: pd.DataFrame = load_chosen_models(chosen_models_csv, args.transition)
        metadata_df: pd.DataFrame = load_metadata(args.metadata_file)
        metadata_analysis_df: pd.DataFrame = build_metadata_analysis_df(chosen_df, metadata_df)
        model_summary_df: pd.DataFrame = load_model_summary(model_summary_csv, args.transition)
        inferred_analysis_df: pd.DataFrame = build_inferred_analysis_df(chosen_df, model_summary_df)

        counts_plot_file: Path = output_dir / f"{args.transition}_chosen_model_counts.png"
        save_count_barplot(chosen_df, counts_plot_file, args.transition)

        output_files.append(counts_plot_file)
        output_files.extend(run_metadata_analysis(metadata_analysis_df, output_dir, args.transition))
        output_files.extend(run_inferred_analysis(inferred_analysis_df, output_dir, args.transition))

        step = "analysis"
        params = {"transition": args.transition, "metadata_file": str(args.metadata_file), "chosen_models_csv": str(chosen_models_csv), "model_summary_csv": str(model_summary_csv), "output_dir": str(output_dir)}
        description = f"Feature analysis for transition {args.transition}."
        notes = "Generated chosen-model counts and comparisons of metadata and inferred features between constant- and linear-chosen families."
        log_relative_path = f"baseline_models/feature_analysis/{args.transition}"
        completion_message = f"Finished feature analysis for transition '{args.transition}'"

    log_run(
        step=step,
        script=Path(__file__),
        params=params,
        outputs=[str(path) for path in output_files],
        description=description,
        notes=notes,
        log_relative_path=log_relative_path,
    )

    print(completion_message)
    print(f"Output directory: {output_dir}")

    for output_file in output_files:
        print(output_file)


if __name__ == "__main__":
    main()