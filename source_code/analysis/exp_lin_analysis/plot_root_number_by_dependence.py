"""
Plot inferred root chromosome number by chromosome-number dependence status.

Poster-oriented exploratory analysis for the FILTERED family set only.

Definitions
-----------
H0:
    Constant is the best-supported model.
H1:
    Linear or Exponential is the best-supported model.

The selected model is taken from the filtered Constant/Linear/Exponential
model-selection table with IGNORE excluded.

To keep root chromosome number comparable between H0 and H1 families, the
root estimate is taken from the CONSTANT-run feature table for every family,
rather than mixing root estimates from different selected models.

Statistics
----------
For each transition, H0 and H1 root chromosome numbers are compared using a
two-sided Welch's independent-samples t-test (equal_var=False). The p-value is
shown directly on the plot and written to a CSV file.

Default behavior
----------------
Generate the analysis for ALL filtered transitions:
    gain, loss, dupl

Default filtered dataset:
    range25_events15_size50

Typical use
-----------
From the project root:

    python -m source_code.analysis.exp_lin_analysis.plot_root_number_by_dependence

Single transition:

    python -m source_code.analysis.exp_lin_analysis.plot_root_number_by_dependence \
        --transition-label dupl

Outputs
-------
analysis/exponential_vs_linear_filtered/<filter>/<transition>/future_work/
    <transition>_root_number_by_dependence.png
    <transition>_root_number_by_dependence_data.csv
    <transition>_root_number_by_dependence_summary.csv
    <transition>_root_number_by_dependence_ttest.csv
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.stats import ttest_ind

from source_code.analysis.analysis_constants import (
    CHOSEN_FUNCTION_LABEL_COL,
    MODEL_COLOR_MAP,
    ROOT_CHROM_NUM_COL,
)
from source_code.analysis.plot_utils import ensure_dir, save_close
from source_code.constants import (
    FAMILY_NAME_COL,
    LABEL_CONSTANT,
    LABEL_DUPL,
    LABEL_EXP,
    LABEL_GAIN,
    LABEL_LINEAR,
    LABEL_LOSS,
)
from source_code.logger import log_run


DEFAULT_FILTER_LABEL = "range25_events15_size50"
DEFAULT_BASELINE_FILTERED_ROOT = Path("analysis/baseline_models_filtered")
DEFAULT_EXP_FILTERED_ROOT = Path("analysis/exponential_vs_linear_filtered")

ALL_TRANSITIONS = (LABEL_GAIN, LABEL_LOSS, LABEL_DUPL)

H0_LABEL = "H0"
H1_LABEL = "H1"
DEPENDENCE_STATUS_COL = "dependence_status"

MODEL_TO_DEPENDENCE = {
    LABEL_CONSTANT: H0_LABEL,
    LABEL_LINEAR: H1_LABEL,
    LABEL_EXP: H1_LABEL,
}

TRANSITION_DISPLAY_NAMES = {
    LABEL_GAIN: "Gain",
    LABEL_LOSS: "Loss",
    LABEL_DUPL: "WGD",
}

MODEL_DISPLAY_NAMES = {
    LABEL_CONSTANT: "Constant",
    LABEL_LINEAR: "Linear",
    LABEL_EXP: "Exponential",
}

GROUP_ORDER = (H0_LABEL, H1_LABEL)
MODEL_ORDER = (LABEL_CONSTANT, LABEL_LINEAR, LABEL_EXP)

FIGSIZE = (7.0, 6.5)
POINT_SIZE = 48
POINT_ALPHA = 0.82
JITTER_WIDTH = 0.14
RANDOM_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot inferred root chromosome number for H0 (Constant) versus "
            "H1 (Linear/Exponential) using filtered families only."
        )
    )
    parser.add_argument(
        "--transition-label",
        default="all",
        choices=[LABEL_GAIN, LABEL_LOSS, LABEL_DUPL, "all"],
        help="Transition to analyze. Default: all.",
    )
    parser.add_argument(
        "--filter-label",
        default=DEFAULT_FILTER_LABEL,
        help=(
            "Filtered-analysis label used under baseline_models_filtered and "
            "exponential_vs_linear_filtered. "
            f"Default: {DEFAULT_FILTER_LABEL}."
        ),
    )
    parser.add_argument(
        "--chosen-model-file",
        type=Path,
        default=None,
        help=(
            "Optional chosen-model CSV for SINGLE-transition use only. "
            "If omitted, the standard filtered path is used."
        ),
    )
    parser.add_argument(
        "--root-features-file",
        type=Path,
        default=None,
        help=(
            "Optional constant-model feature CSV for SINGLE-transition use only. "
            "If omitted, the standard filtered path is used."
        ),
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help=(
            "Optional PNG path for SINGLE-transition use only. "
            "If omitted, the standard filtered future_work path is used."
        ),
    )
    parser.add_argument(
        "--log-note",
        type=str,
        default="",
        help="Optional note describing this run.",
    )
    return parser.parse_args()


def resolve_paths(
    transition_label: str,
    filter_label: str,
    chosen_model_file: Path | None,
    root_features_file: Path | None,
    output_file: Path | None,
) -> tuple[Path, Path, Path]:
    baseline_root = DEFAULT_BASELINE_FILTERED_ROOT / filter_label
    exp_root = DEFAULT_EXP_FILTERED_ROOT / filter_label

    if chosen_model_file is None:
        chosen_model_file = (
            exp_root
            / transition_label
            / "model_selection"
            / f"{transition_label}_chosen_model_with_exp_without_ignore.csv"
        )

    if root_features_file is None:
        root_features_file = (
            baseline_root
            / transition_label
            / f"{transition_label}_constant_features_summary.csv"
        )

    if output_file is None:
        output_file = (
            exp_root
            / transition_label
            / "future_work"
            / f"{transition_label}_root_number_by_dependence.png"
        )

    return chosen_model_file, root_features_file, output_file


def require_columns(
    df: pd.DataFrame,
    required_cols: list[str],
    table_name: str,
) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"{table_name} is missing required columns: {missing}")


def validate_unique_families(
    df: pd.DataFrame,
    table_name: str,
) -> None:
    duplicate_mask = df[FAMILY_NAME_COL].duplicated(keep=False)
    if duplicate_mask.any():
        duplicated = sorted(
            df.loc[duplicate_mask, FAMILY_NAME_COL]
            .astype(str)
            .unique()
            .tolist()
        )
        raise ValueError(
            f"{table_name} contains duplicate family rows: {duplicated}"
        )


def load_chosen_models(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Chosen-model file not found: {file_path}")

    df = pd.read_csv(file_path)
    require_columns(
        df,
        [FAMILY_NAME_COL, CHOSEN_FUNCTION_LABEL_COL],
        "chosen-model table",
    )

    df = df[[FAMILY_NAME_COL, CHOSEN_FUNCTION_LABEL_COL]].copy()
    df[FAMILY_NAME_COL] = df[FAMILY_NAME_COL].astype(str)
    validate_unique_families(df, "chosen-model table")

    observed_models = set(
        df[CHOSEN_FUNCTION_LABEL_COL].dropna().astype(str).unique()
    )
    allowed_models = set(MODEL_TO_DEPENDENCE)
    unexpected_models = sorted(observed_models - allowed_models)

    if unexpected_models:
        raise ValueError(
            "This plot expects the WITHOUT-IGNORE Constant/Linear/Exponential "
            "chosen-model table. "
            f"Unexpected models found: {unexpected_models}"
        )

    df[DEPENDENCE_STATUS_COL] = df[CHOSEN_FUNCTION_LABEL_COL].map(
        MODEL_TO_DEPENDENCE
    )

    if df[DEPENDENCE_STATUS_COL].isna().any():
        bad_rows = df.loc[
            df[DEPENDENCE_STATUS_COL].isna(),
            [FAMILY_NAME_COL, CHOSEN_FUNCTION_LABEL_COL],
        ]
        raise ValueError(
            "Could not classify all selected models as H0/H1:\n"
            f"{bad_rows.to_string(index=False)}"
        )

    return df


def load_constant_root_numbers(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Constant-model feature table not found: {file_path}"
        )

    df = pd.read_csv(file_path)
    require_columns(
        df,
        [FAMILY_NAME_COL, ROOT_CHROM_NUM_COL],
        "constant-model feature table",
    )

    df = df[[FAMILY_NAME_COL, ROOT_CHROM_NUM_COL]].copy()
    df[FAMILY_NAME_COL] = df[FAMILY_NAME_COL].astype(str)
    df[ROOT_CHROM_NUM_COL] = pd.to_numeric(
        df[ROOT_CHROM_NUM_COL],
        errors="coerce",
    )
    validate_unique_families(df, "constant-model feature table")

    return df


def build_plotting_table(
    chosen_df: pd.DataFrame,
    root_df: pd.DataFrame,
) -> pd.DataFrame:
    chosen_families = set(chosen_df[FAMILY_NAME_COL])
    root_families = set(root_df[FAMILY_NAME_COL])

    missing_root_rows = sorted(chosen_families - root_families)
    extra_root_rows = sorted(root_families - chosen_families)

    if missing_root_rows or extra_root_rows:
        raise ValueError(
            "Filtered family sets do not match between model selection and "
            "constant-run root-number table.\n"
            f"Missing from root table: {missing_root_rows or 'NONE'}\n"
            f"Extra in root table: {extra_root_rows or 'NONE'}"
        )

    df = chosen_df.merge(
        root_df,
        on=FAMILY_NAME_COL,
        how="inner",
        validate="one_to_one",
    )

    missing_root_values = int(df[ROOT_CHROM_NUM_COL].isna().sum())
    if missing_root_values:
        print(
            f"[!] Excluding {missing_root_values} families with missing "
            f"{ROOT_CHROM_NUM_COL}."
        )
        df = df.dropna(subset=[ROOT_CHROM_NUM_COL]).copy()

    if df.empty:
        raise ValueError("No families with usable root chromosome numbers.")

    return df.sort_values(
        by=[DEPENDENCE_STATUS_COL, FAMILY_NAME_COL]
    ).reset_index(drop=True)


def get_group_values(
    plot_df: pd.DataFrame,
    status: str,
) -> np.ndarray:
    return (
        plot_df.loc[
            plot_df[DEPENDENCE_STATUS_COL] == status,
            ROOT_CHROM_NUM_COL,
        ]
        .dropna()
        .to_numpy(dtype=float)
    )


def build_summary_table(plot_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for status in GROUP_ORDER:
        values = get_group_values(plot_df, status)

        rows.append(
            {
                DEPENDENCE_STATUS_COL: status,
                "n": int(len(values)),
                "mean_root_chrom_num": (
                    float(np.mean(values)) if len(values) else np.nan
                ),
                "median_root_chrom_num": (
                    float(np.median(values)) if len(values) else np.nan
                ),
                "sd_root_chrom_num": (
                    float(np.std(values, ddof=1))
                    if len(values) > 1
                    else np.nan
                ),
                "q1_root_chrom_num": (
                    float(np.quantile(values, 0.25))
                    if len(values)
                    else np.nan
                ),
                "q3_root_chrom_num": (
                    float(np.quantile(values, 0.75))
                    if len(values)
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def run_welch_t_test(plot_df: pd.DataFrame) -> dict:
    h0_values = get_group_values(plot_df, H0_LABEL)
    h1_values = get_group_values(plot_df, H1_LABEL)

    if len(h0_values) < 2 or len(h1_values) < 2:
        return {
            "test": "Welch two-sided independent-samples t-test",
            "h0_n": int(len(h0_values)),
            "h1_n": int(len(h1_values)),
            "h0_mean": (
                float(np.mean(h0_values)) if len(h0_values) else np.nan
            ),
            "h1_mean": (
                float(np.mean(h1_values)) if len(h1_values) else np.nan
            ),
            "mean_difference_h1_minus_h0": np.nan,
            "t_statistic": np.nan,
            "p_value": np.nan,
            "significant_at_0_05": False,
        }

    result = ttest_ind(
        h1_values,
        h0_values,
        equal_var=False,
        nan_policy="omit",
    )

    return {
        "test": "Welch two-sided independent-samples t-test",
        "h0_n": int(len(h0_values)),
        "h1_n": int(len(h1_values)),
        "h0_mean": float(np.mean(h0_values)),
        "h1_mean": float(np.mean(h1_values)),
        "mean_difference_h1_minus_h0": (
            float(np.mean(h1_values) - np.mean(h0_values))
        ),
        "t_statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "significant_at_0_05": bool(result.pvalue < 0.05),
    }


def format_p_value(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "p = NA"
    if p_value < 0.001:
        return "p < 0.001"
    return f"p = {p_value:.3f}"


def _style_boxplot(boxplot: dict) -> None:
    box_colors = {
        H0_LABEL: MODEL_COLOR_MAP[LABEL_CONSTANT],
        H1_LABEL: MODEL_COLOR_MAP[LABEL_LINEAR],
    }

    for status, box in zip(GROUP_ORDER, boxplot["boxes"]):
        box.set_facecolor(box_colors[status])
        box.set_alpha(0.16)
        box.set_edgecolor(box_colors[status])
        box.set_linewidth(2.0)

    for whisker in boxplot["whiskers"]:
        whisker.set_color("#5A5A5A")
        whisker.set_linewidth(1.4)

    for cap in boxplot["caps"]:
        cap.set_color("#5A5A5A")
        cap.set_linewidth(1.4)

    for median in boxplot["medians"]:
        median.set_color("#222222")
        median.set_linewidth(2.2)


def add_t_test_annotation(
    ax: plt.Axes,
    group_values: list[np.ndarray],
    p_value: float,
) -> None:
    all_values = np.concatenate(group_values)

    data_min = float(np.min(all_values))
    data_max = float(np.max(all_values))
    data_range = data_max - data_min

    if data_range <= 0:
        data_range = max(abs(data_max), 1.0)

    bracket_y = data_max + 0.09 * data_range
    bracket_height = 0.035 * data_range
    text_y = bracket_y + 0.055 * data_range

    ax.plot(
        [0, 0, 1, 1],
        [
            bracket_y,
            bracket_y + bracket_height,
            bracket_y + bracket_height,
            bracket_y,
        ],
        color="#333333",
        linewidth=1.3,
        clip_on=False,
    )

    ax.text(
        0.5,
        text_y,
        f"Welch's t-test: {format_p_value(p_value)}",
        ha="center",
        va="bottom",
        fontsize=10.5,
    )

    current_bottom, _ = ax.get_ylim()
    ax.set_ylim(
        current_bottom,
        data_max + 0.26 * data_range,
    )


def plot_root_number_by_dependence(
    plot_df: pd.DataFrame,
    transition_label: str,
    t_test_result: dict,
    output_file: Path,
) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)

    group_values = [
        get_group_values(plot_df, status)
        for status in GROUP_ORDER
    ]

    if any(len(values) == 0 for values in group_values):
        missing_groups = [
            status
            for status, values in zip(GROUP_ORDER, group_values)
            if len(values) == 0
        ]
        raise ValueError(
            f"Cannot make H0-vs-H1 plot because these groups are empty: "
            f"{missing_groups}"
        )

    positions = np.arange(len(GROUP_ORDER), dtype=float)

    boxplot = ax.boxplot(
        group_values,
        positions=positions,
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        zorder=1,
    )
    _style_boxplot(boxplot)

    rng = np.random.default_rng(RANDOM_SEED)

    for group_index, status in enumerate(GROUP_ORDER):
        group_df = plot_df[
            plot_df[DEPENDENCE_STATUS_COL] == status
        ].copy()

        for model_label in MODEL_ORDER:
            model_df = group_df[
                group_df[CHOSEN_FUNCTION_LABEL_COL] == model_label
            ]
            if model_df.empty:
                continue

            jitter = rng.uniform(
                -JITTER_WIDTH,
                JITTER_WIDTH,
                size=len(model_df),
            )
            x_values = positions[group_index] + jitter

            ax.scatter(
                x_values,
                model_df[ROOT_CHROM_NUM_COL].to_numpy(dtype=float),
                s=POINT_SIZE,
                color=MODEL_COLOR_MAP[model_label],
                alpha=POINT_ALPHA,
                edgecolors="white",
                linewidths=0.7,
                zorder=3,
            )

    group_counts = (
        plot_df[DEPENDENCE_STATUS_COL]
        .value_counts()
        .reindex(GROUP_ORDER, fill_value=0)
    )

    ax.set_xticks(
        positions,
        [
            rf"$H_0$" + "\nNo dependence"
            + f"\n(n={int(group_counts[H0_LABEL])})",
            rf"$H_1$" + "\nDependence"
            + f"\n(n={int(group_counts[H1_LABEL])})",
        ],
    )

    transition_name = TRANSITION_DISPLAY_NAMES[transition_label]
    ax.set_title(
        f"Does ancestral chromosome number predict {transition_name} dependence?",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    ax.set_ylabel("Inferred root chromosome number", fontsize=12)
    ax.set_xlabel("")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.18, linewidth=0.8)
    ax.set_axisbelow(True)

    used_models = [
        model
        for model in MODEL_ORDER
        if model in set(plot_df[CHOSEN_FUNCTION_LABEL_COL])
    ]

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=MODEL_COLOR_MAP[model],
            markeredgecolor="white",
            markersize=8,
            label=MODEL_DISPLAY_NAMES[model],
        )
        for model in used_models
    ]

    legend = ax.legend(
        handles=legend_handles,
        title="Selected model",
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        facecolor="white",
        edgecolor="#555555",
        loc="upper right",
    )
    legend.get_frame().set_linewidth(1.0)

    add_t_test_annotation(
        ax=ax,
        group_values=group_values,
        p_value=t_test_result["p_value"],
    )

    fig.tight_layout()
    ensure_dir(output_file.parent)
    save_close(fig, output_file)


def run_single_transition(
    transition_label: str,
    filter_label: str,
    chosen_model_file: Path | None = None,
    root_features_file: Path | None = None,
    output_file: Path | None = None,
    log_note: str = "",
) -> dict:
    chosen_model_file, root_features_file, output_file = resolve_paths(
        transition_label=transition_label,
        filter_label=filter_label,
        chosen_model_file=chosen_model_file,
        root_features_file=root_features_file,
        output_file=output_file,
    )

    chosen_df = load_chosen_models(chosen_model_file)
    root_df = load_constant_root_numbers(root_features_file)
    plot_df = build_plotting_table(chosen_df, root_df)

    output_dir = output_file.parent
    ensure_dir(output_dir)

    data_file = output_dir / (
        f"{transition_label}_root_number_by_dependence_data.csv"
    )
    summary_file = output_dir / (
        f"{transition_label}_root_number_by_dependence_summary.csv"
    )
    ttest_file = output_dir / (
        f"{transition_label}_root_number_by_dependence_ttest.csv"
    )

    summary_df = build_summary_table(plot_df)
    t_test_result = run_welch_t_test(plot_df)

    plot_df.to_csv(data_file, index=False)
    summary_df.to_csv(summary_file, index=False)

    ttest_row = {
        "transition_label": transition_label,
        **t_test_result,
    }
    pd.DataFrame([ttest_row]).to_csv(ttest_file, index=False)

    plot_root_number_by_dependence(
        plot_df=plot_df,
        transition_label=transition_label,
        t_test_result=t_test_result,
        output_file=output_file,
    )

    outputs = [
        str(output_file),
        str(data_file),
        str(summary_file),
        str(ttest_file),
    ]

    notes = (
        "Filtered families only. H0 = Constant; H1 = Linear or Exponential. "
        "IGNORE is excluded. Root chromosome number is taken from the "
        "Constant-run feature table for every family so H0 and H1 use a "
        "common inference reference. H0 versus H1 was tested using a "
        "two-sided Welch independent-samples t-test."
    )
    if log_note:
        notes += f"\nRun note: {log_note}"

    log_run(
        step="plot_root_number_by_dependence",
        script=Path(__file__),
        params={
            "transition_label": transition_label,
            "filter_label": filter_label,
            "chosen_model_file": str(chosen_model_file),
            "root_features_file": str(root_features_file),
            "n_families_plotted": len(plot_df),
            "h0_n": int(t_test_result["h0_n"]),
            "h1_n": int(t_test_result["h1_n"]),
            "welch_t_statistic": t_test_result["t_statistic"],
            "welch_p_value": t_test_result["p_value"],
        },
        outputs=outputs,
        description=(
            "Compared inferred root chromosome number between H0 and H1 "
            "for the filtered family set and plotted the result."
        ),
        notes=notes,
        log_relative_path=(
            Path("future_work")
            / f"{transition_label}_root_number_by_dependence.log"
        ),
    )

    print(f"[✓] {transition_label}")
    print(
        f"    H0 n={t_test_result['h0_n']}, "
        f"mean={t_test_result['h0_mean']:.3f}"
    )
    print(
        f"    H1 n={t_test_result['h1_n']}, "
        f"mean={t_test_result['h1_mean']:.3f}"
    )
    print(
        f"    Welch t={t_test_result['t_statistic']:.4f}, "
        f"{format_p_value(t_test_result['p_value'])}"
    )
    print(f"    Plot: {output_file}")

    return ttest_row


def main() -> None:
    args = parse_args()

    if args.transition_label == "all":
        if any(
            value is not None
            for value in (
                args.chosen_model_file,
                args.root_features_file,
                args.output_file,
            )
        ):
            raise ValueError(
                "--chosen-model-file, --root-features-file, and --output-file "
                "can only be used when a single transition is selected."
            )

        print(f"Filtered dataset: {args.filter_label}")
        print("Running gain, loss, and duplication...\n")

        all_test_rows = []
        for transition_label in ALL_TRANSITIONS:
            test_row = run_single_transition(
                transition_label=transition_label,
                filter_label=args.filter_label,
                log_note=args.log_note,
            )
            all_test_rows.append(test_row)

        combined_stats_file = (
            DEFAULT_EXP_FILTERED_ROOT
            / args.filter_label
            / "root_number_by_dependence_ttests_all_transitions.csv"
        )
        ensure_dir(combined_stats_file.parent)
        pd.DataFrame(all_test_rows).to_csv(
            combined_stats_file,
            index=False,
        )

        print("\n[✓] All three transitions completed.")
        print(f"Combined t-test summary: {combined_stats_file}")
        return

    run_single_transition(
        transition_label=args.transition_label,
        filter_label=args.filter_label,
        chosen_model_file=args.chosen_model_file,
        root_features_file=args.root_features_file,
        output_file=args.output_file,
        log_note=args.log_note,
    )


if __name__ == "__main__":
    main()
