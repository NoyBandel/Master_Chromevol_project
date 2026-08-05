import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from source_code.analysis.analysis_constants import (
    AICC_COL,
    BASELINE_ANALYSIS_DIR,
    BASELINE_MODELS_LABEL,
    CHOSEN_FUNCTION_LABEL_COL,
    CHOSEN_MODEL_SUFFIX,
    FAMILY_NAME_COL,
    LABEL_CONSTANT,
    LABEL_DUPL,
    LABEL_FUNC_TYPE_COL,
    LABEL_GAIN,
    LABEL_IGNORE,
    LABEL_LINEAR,
    LABEL_LOSS,
    MODEL_COLOR_MAP,
    MODEL_SELECTION_SUBDIR,
    MODELS_SUMMARY_SUFFIX,
)
from source_code.analysis.plot_utils import DEFAULT_DPI, ensure_dir
from source_code.logger import log_run


# -----------------------------------------------------------------------------
# Allowed transitions
# -----------------------------------------------------------------------------
ALLOWED_TRANSITIONS = (LABEL_GAIN, LABEL_LOSS, LABEL_DUPL)


# -----------------------------------------------------------------------------
# Columns created by this script
# -----------------------------------------------------------------------------
CONSTANT_AICC_COL = "constant_AICc"
LINEAR_AICC_COL = "linear_AICc"
SIGNED_DELTA_AICC_COL = "signed_delta_AICc_constant_minus_linear"
SUPPORT_MAGNITUDE_COL = "abs_signed_delta_AICc"
WINNING_MODEL_COL = "winning_model"

TIE_LABEL = "tie"


# -----------------------------------------------------------------------------
# Loading / path helpers
# -----------------------------------------------------------------------------
def build_default_models_summary_file(transition_label: str) -> Path:
    return BASELINE_ANALYSIS_DIR / transition_label / f"{transition_label}_{MODELS_SUMMARY_SUFFIX}"


def build_default_chosen_model_file(transition_label: str) -> Path:
    return BASELINE_ANALYSIS_DIR / transition_label / f"{transition_label}_{CHOSEN_MODEL_SUFFIX}"


def build_default_output_dir(transition_label: str) -> Path:
    return BASELINE_ANALYSIS_DIR / transition_label / MODEL_SELECTION_SUBDIR


def load_models_summary_table(models_summary_file: Path) -> pd.DataFrame:
    if not models_summary_file.exists():
        raise FileNotFoundError(f"Missing models summary table: {models_summary_file}")

    df = pd.read_csv(models_summary_file)

    required_cols = [FAMILY_NAME_COL, LABEL_FUNC_TYPE_COL, AICC_COL]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns in {models_summary_file}: {missing_cols}")

    return df


def load_non_ignore_family_names(chosen_model_file: Path) -> Set[str]:
    """
    Return family names for which the baseline selected model is NOT ignore.

    This filter is applied before computing signed ΔAICc, so the histogram only
    represents families where the actual model-selection result was constant or linear.
    """
    if not chosen_model_file.exists():
        raise FileNotFoundError(f"Missing chosen model table: {chosen_model_file}")

    chosen_df = pd.read_csv(chosen_model_file)

    required_cols = [FAMILY_NAME_COL, CHOSEN_FUNCTION_LABEL_COL]
    missing_cols = [col for col in required_cols if col not in chosen_df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns in {chosen_model_file}: {missing_cols}")

    chosen_df = chosen_df.dropna(subset=[FAMILY_NAME_COL, CHOSEN_FUNCTION_LABEL_COL]).copy()

    chosen_df[FAMILY_NAME_COL] = chosen_df[FAMILY_NAME_COL].astype(str)
    chosen_df[CHOSEN_FUNCTION_LABEL_COL] = chosen_df[CHOSEN_FUNCTION_LABEL_COL].astype(str)

    non_ignore_df = chosen_df[chosen_df[CHOSEN_FUNCTION_LABEL_COL] != LABEL_IGNORE].copy()

    return set(non_ignore_df[FAMILY_NAME_COL])


# -----------------------------------------------------------------------------
# Signed delta construction
# -----------------------------------------------------------------------------
def build_signed_delta_table(
    models_summary_df: pd.DataFrame,
    non_ignore_family_names: Set[str],
) -> pd.DataFrame:
    """
    Builds one row per non-ignore family with:
        signed ΔAICc = AICc_constant - AICc_linear

    Interpretation:
        signed ΔAICc < 0  -> constant has lower AICc, constant preferred
        signed ΔAICc > 0  -> linear has lower AICc, linear preferred
        signed ΔAICc = 0  -> equal support

    Important:
        Families where the baseline chosen model was ignore are excluded before
        this signed ΔAICc table is computed.
    """
    plot_df = models_summary_df[[FAMILY_NAME_COL, LABEL_FUNC_TYPE_COL, AICC_COL]].copy()

    plot_df[FAMILY_NAME_COL] = plot_df[FAMILY_NAME_COL].astype(str)

    # Filter out ignore-selected families BEFORE computing signed ΔAICc.
    plot_df = plot_df[plot_df[FAMILY_NAME_COL].isin(non_ignore_family_names)].copy()

    # Keep only constant and linear rows.
    plot_df = plot_df[plot_df[LABEL_FUNC_TYPE_COL].isin([LABEL_CONSTANT, LABEL_LINEAR])].copy()

    plot_df[AICC_COL] = pd.to_numeric(plot_df[AICC_COL], errors="coerce")
    plot_df = plot_df.dropna(subset=[FAMILY_NAME_COL, LABEL_FUNC_TYPE_COL, AICC_COL])

    if plot_df.empty:
        raise ValueError("No usable constant/linear AICc rows found after excluding ignore-selected families.")

    # If duplicates exist, keep the best AICc for each family x model.
    wide_df = plot_df.pivot_table(
        index=FAMILY_NAME_COL,
        columns=LABEL_FUNC_TYPE_COL,
        values=AICC_COL,
        aggfunc="min",
    ).reset_index()

    missing_models = [label for label in [LABEL_CONSTANT, LABEL_LINEAR] if label not in wide_df.columns]
    if missing_models:
        raise ValueError(f"Missing model columns after pivot: {missing_models}")

    signed_df = wide_df[[FAMILY_NAME_COL, LABEL_CONSTANT, LABEL_LINEAR]].copy()
    signed_df = signed_df.rename(
        columns={
            LABEL_CONSTANT: CONSTANT_AICC_COL,
            LABEL_LINEAR: LINEAR_AICC_COL,
        }
    )

    signed_df = signed_df.dropna(subset=[CONSTANT_AICC_COL, LINEAR_AICC_COL]).reset_index(drop=True)

    if signed_df.empty:
        raise ValueError("No families had both constant and linear AICc values after excluding ignore-selected families.")

    signed_df[SIGNED_DELTA_AICC_COL] = signed_df[CONSTANT_AICC_COL] - signed_df[LINEAR_AICC_COL]
    signed_df[SUPPORT_MAGNITUDE_COL] = signed_df[SIGNED_DELTA_AICC_COL].abs()

    signed_df[WINNING_MODEL_COL] = np.select(
        condlist=[
            signed_df[SIGNED_DELTA_AICC_COL] < 0,
            signed_df[SIGNED_DELTA_AICC_COL] > 0,
        ],
        choicelist=[LABEL_CONSTANT, LABEL_LINEAR],
        default=TIE_LABEL,
    )

    return signed_df.sort_values(by=SIGNED_DELTA_AICC_COL).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------
def build_symmetric_bins(
    values: pd.Series,
    bin_count: int = 40,
    bin_size: Optional[float] = None,
) -> np.ndarray:
    clean_values = pd.to_numeric(values, errors="coerce").dropna()

    if clean_values.empty:
        raise ValueError("Cannot build histogram bins: no numeric values.")

    max_abs = float(np.nanmax(np.abs(clean_values)))

    if max_abs == 0:
        max_abs = 1.0

    if bin_size is not None:
        if bin_size <= 0:
            raise ValueError("bin_size must be positive.")

        max_abs = np.ceil(max_abs / bin_size) * bin_size
        return np.arange(-max_abs, max_abs + bin_size, bin_size)

    # Force an even number of bins so zero is a bin edge.
    if bin_count % 2 != 0:
        bin_count += 1

    return np.linspace(-max_abs, max_abs, bin_count + 1)


def format_median_abs_delta(values: pd.Series) -> str:
    clean_values = pd.to_numeric(values, errors="coerce").dropna().abs()

    if clean_values.empty:
        return "NA"

    return f"{clean_values.median():.2f}"


def build_summary_text(signed_df: pd.DataFrame) -> str:
    constant_values = signed_df.loc[
        signed_df[WINNING_MODEL_COL] == LABEL_CONSTANT,
        SIGNED_DELTA_AICC_COL,
    ]
    linear_values = signed_df.loc[
        signed_df[WINNING_MODEL_COL] == LABEL_LINEAR,
        SIGNED_DELTA_AICC_COL,
    ]
    tie_values = signed_df.loc[
        signed_df[WINNING_MODEL_COL] == TIE_LABEL,
        SIGNED_DELTA_AICC_COL,
    ]

    return "\n".join(
        [
            f"n={len(signed_df)} families",
            f"constant wins: n={len(constant_values)}, median |ΔAICc|={format_median_abs_delta(constant_values)}",
            f"linear wins: n={len(linear_values)}, median |ΔAICc|={format_median_abs_delta(linear_values)}",
            f"ties: n={len(tie_values)}",
        ]
    )


def plot_signed_delta_histogram(
    signed_df: pd.DataFrame,
    transition_label: str,
    out_file: Path,
    bin_count: int = 40,
    bin_size: Optional[float] = None,
    add_support_guides: bool = True,
) -> None:
    if signed_df.empty:
        raise ValueError("No rows available for plotting.")

    values = signed_df[SIGNED_DELTA_AICC_COL]
    bins = build_symmetric_bins(values=values, bin_count=bin_count, bin_size=bin_size)

    constant_values = signed_df.loc[
        signed_df[WINNING_MODEL_COL] == LABEL_CONSTANT,
        SIGNED_DELTA_AICC_COL,
    ]
    tie_values = signed_df.loc[
        signed_df[WINNING_MODEL_COL] == TIE_LABEL,
        SIGNED_DELTA_AICC_COL,
    ]
    linear_values = signed_df.loc[
        signed_df[WINNING_MODEL_COL] == LABEL_LINEAR,
        SIGNED_DELTA_AICC_COL,
    ]

    data: List[pd.Series] = []
    labels: List[str] = []
    colors: List[str] = []

    if not constant_values.empty:
        data.append(constant_values)
        labels.append("constant preferred")
        colors.append(MODEL_COLOR_MAP[LABEL_CONSTANT])

    if not tie_values.empty:
        data.append(tie_values)
        labels.append("equal support")
        colors.append("#9E9E9E")

    if not linear_values.empty:
        data.append(linear_values)
        labels.append("linear preferred")
        colors.append(MODEL_COLOR_MAP[LABEL_LINEAR])

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        data,
        bins=bins,
        stacked=True,
        color=colors,
        edgecolor="black",
        label=labels,
    )

    ax.axvline(
        0,
        color="black",
        linewidth=1.8,
        linestyle="-",
        label="zero line: equal support",
    )

    if add_support_guides:
        ax.axvline(-2, color="black", linewidth=1.0, linestyle=":", alpha=0.7)
        ax.axvline(2, color="black", linewidth=1.0, linestyle=":", alpha=0.7)

        ax.text(
            0.5,
            0.92,
            "weak difference",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
        )

    ax.text(
        0.02,
        0.97,
        "constant preferred\nΔAICc < 0",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color=MODEL_COLOR_MAP[LABEL_CONSTANT],
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="none"),
    )

    ax.text(
        0.98,
        0.97,
        "linear preferred\nΔAICc > 0",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color=MODEL_COLOR_MAP[LABEL_LINEAR],
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="none"),
    )

    ax.text(
        0.98,
        0.78,
        build_summary_text(signed_df),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="black"),
    )

    ax.set_title(f"{transition_label}: strength of support for dependence")
    ax.set_xlabel("Signed ΔAICc = AICc(constant) − AICc(linear)")
    ax.set_ylabel("Number of families")

    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.78), frameon=True)

    fig.tight_layout()
    fig.savefig(out_file, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Running
# -----------------------------------------------------------------------------
def run_transition(
    transition_label: str,
    models_summary_file: Optional[Path] = None,
    chosen_model_file: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    bin_count: int = 40,
    bin_size: Optional[float] = None,
    add_support_guides: bool = True,
) -> Dict[str, object]:
    if transition_label not in ALLOWED_TRANSITIONS:
        raise ValueError(f"Unsupported transition: {transition_label}. Allowed: {ALLOWED_TRANSITIONS}")

    if models_summary_file is None:
        models_summary_file = build_default_models_summary_file(transition_label)

    if chosen_model_file is None:
        chosen_model_file = build_default_chosen_model_file(transition_label)

    if output_dir is None:
        output_dir = build_default_output_dir(transition_label)

    ensure_dir(output_dir)

    models_summary_df = load_models_summary_table(models_summary_file)
    non_ignore_family_names = load_non_ignore_family_names(chosen_model_file)

    signed_df = build_signed_delta_table(
        models_summary_df=models_summary_df,
        non_ignore_family_names=non_ignore_family_names,
    )

    table_file = output_dir / f"{transition_label}_signed_delta_AICc_constant_vs_linear_non_ignore.csv"
    plot_file = output_dir / f"{transition_label}_signed_delta_AICc_constant_vs_linear_non_ignore_hist.png"

    signed_df.to_csv(table_file, index=False)

    plot_signed_delta_histogram(
        signed_df=signed_df,
        transition_label=transition_label,
        out_file=plot_file,
        bin_count=bin_count,
        bin_size=bin_size,
        add_support_guides=add_support_guides,
    )

    counts = signed_df[WINNING_MODEL_COL].value_counts().to_dict()
    outputs = [str(table_file), str(plot_file)]

    log_run(
        step="analysis",
        script=Path(__file__),
        params={
            "transition": transition_label,
            "models_summary_file": str(models_summary_file),
            "chosen_model_file": str(chosen_model_file),
            "output_dir": str(output_dir),
            "n_non_ignore_families_in_chosen_table": len(non_ignore_family_names),
            "n_families_with_constant_and_linear": len(signed_df),
            "winning_model_counts": counts,
            "bin_count": bin_count,
            "bin_size": bin_size,
            "add_support_guides": add_support_guides,
        },
        outputs=outputs,
        description=(
            f"Signed ΔAICc support plot for constant vs linear model selection "
            f"in transition '{transition_label}', excluding ignore-selected families"
        ),
        notes=(
            "Families with chosen_function_label == ignore are excluded before computing signed ΔAICc. "
            "Signed ΔAICc is defined as AICc_constant - AICc_linear; "
            "positive values support linear dependence, negative values support constant."
        ),
        log_relative_path=Path(BASELINE_MODELS_LABEL) / f"{transition_label}.log",
    )

    return {
        "transition": transition_label,
        "n_families": len(signed_df),
        "winning_model_counts": counts,
        "outputs": outputs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot signed ΔAICc = AICc_constant - AICc_linear for constant vs linear support, "
            "excluding families where the baseline selected model is ignore."
        )
    )

    parser.add_argument(
        "--transition",
        required=True,
        choices=list(ALLOWED_TRANSITIONS),
        help="Transition label to plot. Allowed values: gain, loss, dupl.",
    )

    parser.add_argument(
        "--models_summary_file",
        type=Path,
        default=None,
        help=(
            "Optional input CSV. Defaults to "
            "analysis/baseline_models/<transition>/<transition>_models_summary_table.csv."
        ),
    )

    parser.add_argument(
        "--chosen_model_file",
        type=Path,
        default=None,
        help=(
            "Optional chosen-model CSV used to exclude ignore-selected families. Defaults to "
            "analysis/baseline_models/<transition>/<transition>_chosen_model_table.csv."
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to analysis/baseline_models/<transition>/model_selection/.",
    )

    parser.add_argument(
        "--bins",
        type=int,
        default=40,
        help="Number of histogram bins. Forced to even so zero is a bin edge.",
    )

    parser.add_argument(
        "--bin_size",
        type=float,
        default=None,
        help="Optional fixed histogram bin size in ΔAICc units.",
    )

    parser.add_argument(
        "--no_support_guides",
        action="store_true",
        help="Do not draw the ±2 weak-support guide lines.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result = run_transition(
        transition_label=args.transition,
        models_summary_file=args.models_summary_file,
        chosen_model_file=args.chosen_model_file,
        output_dir=args.output_dir,
        bin_count=args.bins,
        bin_size=args.bin_size,
        add_support_guides=not args.no_support_guides,
    )

    print(f"[✓] {result['transition']}: n={result['n_families']}, counts={result['winning_model_counts']}")

    for out in result["outputs"]:
        print(f"    {out}")


if __name__ == "__main__":
    main()