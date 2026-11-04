"""
Create one poster-wide figure containing selected fitted-rate functions
for Whole-Genome Duplication, Loss, and Gain.

Panel order:
    WGD | Loss | Gain

Each panel contains:
- selected H1 fitted functions (Linear / Exponential)
- one summary frame anchored exactly at the top-right corner
- an inner H0/H1 + Linear/Exponential information box
- a large Constant / Linear / Exponential pie chart

Design:
- one shared x-axis label
- one shared y-axis label
- panel titles: WGD / Loss / Gain
- y-axis range: 0-60
- one shared legend for model colors
- large poster-readable fonts
- all summary text kept inside its frame
- large pie charts
- outer summary frames extended slightly downward
- pies positioned slightly lower inside the frame
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

from source_code.analysis.analysis_constants import MODEL_COLOR_MAP
from source_code.analysis.fitted_functions.plot_selected_functions_overlay import (
    add_selected_function_curves,
    build_poster_model_counts,
    load_chosen_models,
    load_fitted_functions,
    prepare_selected_functions,
)
from source_code.analysis.plot_utils import ensure_dir
from source_code.constants import (
    LABEL_CONSTANT,
    LABEL_DUPL,
    LABEL_EXP,
    LABEL_GAIN,
    LABEL_LINEAR,
    LABEL_LOSS,
)


# =============================================================================
# Figure settings
# =============================================================================

FIG_WIDTH_CM: float = 88.33
FIG_HEIGHT_CM: float = 24.0

CM_PER_INCH: float = 2.54
PNG_DPI: int = 300

FULL_X_MAX: float = 150.0
FULL_Y_MAX: float = 60.0


# =============================================================================
# Transition / model definitions
# =============================================================================

SELECTED_MODELS: tuple[str, ...] = (
    LABEL_LINEAR,
    LABEL_EXP,
)

PANEL_ORDER: tuple[str, ...] = (
    LABEL_DUPL,
    LABEL_LOSS,
    LABEL_GAIN,
)

PANEL_TITLES: dict[str, str] = {
    LABEL_DUPL: "WGD",
    LABEL_LOSS: "Loss",
    LABEL_GAIN: "Gain",
}

MODEL_DISPLAY_NAMES: dict[str, str] = {
    LABEL_CONSTANT: "Constant",
    LABEL_LINEAR: "Linear",
    LABEL_EXP: "Exponential",
}


# =============================================================================
# Typography
# =============================================================================

TITLE_FONTSIZE: float = 26
AXIS_LABEL_FONTSIZE: float = 30
TICK_FONTSIZE: float = 19

SUMMARY_HEADER_FONTSIZE: float = 18
SUMMARY_ROW_FONTSIZE: float = 15.5

PIE_AUTOPCT_FONTSIZE: float = 14
LEGEND_FONTSIZE: float = 19


# =============================================================================
# Fitted-function styling
# =============================================================================

LINE_ALPHA: float = 0.65
LINE_WIDTH: float = 1.8


# =============================================================================
# Arguments
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one combined poster figure for "
            "WGD, chromosome loss, and chromosome gain."
        )
    )

    for transition in (
        LABEL_GAIN,
        LABEL_LOSS,
        LABEL_DUPL,
    ):
        parser.add_argument(
            f"--{transition}-fitted-functions-file",
            type=Path,
            required=True,
        )

        parser.add_argument(
            f"--{transition}-chosen-model-file",
            type=Path,
            required=True,
        )

    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="PNG output path.",
    )

    parser.add_argument(
        "--pdf-output-file",
        type=Path,
        default=None,
        help=(
            "Optional PDF output path. "
            "Defaults to the PNG path with .pdf suffix."
        ),
    )

    return parser.parse_args()


# =============================================================================
# Data loading
# =============================================================================

def load_transition_data(
    fitted_file: Path,
    chosen_file: Path,
) -> tuple[pd.DataFrame, pd.Series]:

    fitted_df = load_fitted_functions(fitted_file)
    chosen_df = load_chosen_models(chosen_file)

    selected_df = prepare_selected_functions(
        fitted_df=fitted_df,
        chosen_df=chosen_df,
        selected_models=SELECTED_MODELS,
    )

    model_counts = build_poster_model_counts(chosen_df)

    return selected_df, model_counts


# =============================================================================
# Helpers
# =============================================================================

def relative_luminance(color) -> float:
    """
    Approximate perceived luminance.

    Used to choose black or white text inside each pie slice.
    """

    r, g, b = to_rgb(color)

    return (
        0.2126 * r
        + 0.7152 * g
        + 0.0722 * b
    )


def make_pie_autopct(total: int):
    """
    Show percentage and number of families inside each pie slice.
    """

    def _autopct(pct: float) -> str:
        count = int(round(pct * total / 100.0))

        if count == 0:
            return ""

        return f"{pct:.1f}%\n(n={count})"

    return _autopct


# =============================================================================
# Main panel styling
# =============================================================================

def style_panel(
    ax: plt.Axes,
    panel_title: str,
) -> None:

    ax.set_xlim(0, FULL_X_MAX)
    ax.set_ylim(0, FULL_Y_MAX)

    ax.set_title(
        panel_title,
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
        pad=12,
    )

    # Shared figure-level axis labels.
    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.tick_params(
        axis="both",
        labelsize=TICK_FONTSIZE,
        length=5,
        width=1.0,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# =============================================================================
# Shared model legend
# =============================================================================

def add_shared_model_legend(
    fig: plt.Figure,
) -> None:

    handles = [
        Patch(
            facecolor=MODEL_COLOR_MAP[LABEL_CONSTANT],
            edgecolor="none",
            label=MODEL_DISPLAY_NAMES[LABEL_CONSTANT],
        ),
        Patch(
            facecolor=MODEL_COLOR_MAP[LABEL_LINEAR],
            edgecolor="none",
            label=MODEL_DISPLAY_NAMES[LABEL_LINEAR],
        ),
        Patch(
            facecolor=MODEL_COLOR_MAP[LABEL_EXP],
            edgecolor="none",
            label=MODEL_DISPLAY_NAMES[LABEL_EXP],
        ),
    ]

    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),

        ncol=3,
        frameon=False,

        fontsize=LEGEND_FONTSIZE,

        handlelength=1.5,
        handleheight=0.9,

        columnspacing=2.2,
        handletextpad=0.7,

        borderaxespad=0.0,
    )


# =============================================================================
# Per-panel model-selection summary
# =============================================================================

def add_large_poster_summary(
    ax: plt.Axes,
    model_counts: pd.Series,
) -> None:

    # -------------------------------------------------------------------------
    # Counts
    # -------------------------------------------------------------------------

    total_n = int(model_counts.sum())
    constant_n = int(model_counts[LABEL_CONSTANT])
    linear_n = int(model_counts[LABEL_LINEAR])
    exp_n = int(model_counts[LABEL_EXP])

    h1_n = linear_n + exp_n

    # -------------------------------------------------------------------------
    # Percentages
    # -------------------------------------------------------------------------

    h0_pct = (
        100.0 * constant_n / total_n
        if total_n
        else 0.0
    )

    h1_pct = (
        100.0 * h1_n / total_n
        if total_n
        else 0.0
    )

    linear_pct_h1 = (
        100.0 * linear_n / h1_n
        if h1_n
        else 0.0
    )

    exp_pct_h1 = (
        100.0 * exp_n / h1_n
        if h1_n
        else 0.0
    )

    # =========================================================================
    # OUTER SUMMARY FRAME
    #
    # Previous:
    #     [0.50, 0.16, 0.50, 0.84]
    #
    # New:
    #     [0.50, 0.10, 0.50, 0.90]
    #
    # The top edge remains exactly at:
    #     0.10 + 0.90 = 1.00
    #
    # so the box is still anchored to the upper-right corner,
    # but it now extends farther downward.
    # =========================================================================

    outer_ax = ax.inset_axes(
        [
            0.50,   # left
            0.10,   # bottom
            0.50,   # width
            0.90,   # height
        ]
    )

    outer_ax.set_facecolor("white")
    outer_ax.set_xlim(0, 1)
    outer_ax.set_ylim(0, 1)

    outer_ax.set_xticks([])
    outer_ax.set_yticks([])

    for spine in outer_ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.1)
        spine.set_edgecolor("0.45")

    # =========================================================================
    # INNER SUMMARY BOX
    #
    # Kept close to the top of the enlarged outer frame.
    # =========================================================================

    summary_ax = outer_ax.inset_axes(
        [
            0.04,
            0.72,
            0.92,
            0.25,
        ]
    )

    summary_ax.set_facecolor("white")
    summary_ax.set_xlim(0, 1)
    summary_ax.set_ylim(0, 1)

    summary_ax.set_xticks([])
    summary_ax.set_yticks([])

    for spine in summary_ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_edgecolor("0.55")

    # -------------------------------------------------------------------------
    # H0
    # -------------------------------------------------------------------------

    summary_ax.text(
        0.06,
        0.83,

        f"H₀: {constant_n}/{total_n} "
        f"({h0_pct:.1f}%)",

        ha="left",
        va="center",

        fontsize=SUMMARY_HEADER_FONTSIZE,
        clip_on=True,
    )

    # -------------------------------------------------------------------------
    # H1
    # -------------------------------------------------------------------------

    summary_ax.text(
        0.06,
        0.62,

        f"H₁: {h1_n}/{total_n} "
        f"({h1_pct:.1f}%)",

        ha="left",
        va="center",

        fontsize=SUMMARY_HEADER_FONTSIZE,
        clip_on=True,
    )

    # -------------------------------------------------------------------------
    # Linear row
    # -------------------------------------------------------------------------

    y_linear = 0.38

    summary_ax.plot(
        [0.07, 0.18],
        [y_linear, y_linear],

        color=MODEL_COLOR_MAP[LABEL_LINEAR],

        linewidth=4.0,
        solid_capstyle="butt",

        clip_on=True,
    )

    summary_ax.text(
        0.22,
        y_linear,

        (
            f"Linear: "
            f"{linear_n}/{h1_n} "
            f"({linear_pct_h1:.1f}%)"
        ),

        ha="left",
        va="center",

        fontsize=SUMMARY_ROW_FONTSIZE,
        clip_on=True,
    )

    # -------------------------------------------------------------------------
    # Exponential row
    # -------------------------------------------------------------------------

    y_exp = 0.15

    summary_ax.plot(
        [0.07, 0.18],
        [y_exp, y_exp],

        color=MODEL_COLOR_MAP[LABEL_EXP],

        linewidth=4.0,
        solid_capstyle="butt",

        clip_on=True,
    )

    summary_ax.text(
        0.22,
        y_exp,

        (
            f"Exponential: "
            f"{exp_n}/{h1_n} "
            f"({exp_pct_h1:.1f}%)"
        ),

        ha="left",
        va="center",

        fontsize=SUMMARY_ROW_FONTSIZE,
        clip_on=True,
    )

    # =========================================================================
    # PIE
    #
    # Same pie SIZE as before:
    #     radius = 1.15
    #
    # But the pie axes are shifted slightly downward.
    # =========================================================================

    pie_ax = outer_ax.inset_axes(
        [
            0.00,
            -0.010,
            1.00,
            0.72,
        ]
    )

    pie_ax.set_aspect("equal")
    pie_ax.set_xticks([])
    pie_ax.set_yticks([])

    values = [
        constant_n,
        linear_n,
        exp_n,
    ]

    colors = [
        MODEL_COLOR_MAP[LABEL_CONSTANT],
        MODEL_COLOR_MAP[LABEL_LINEAR],
        MODEL_COLOR_MAP[LABEL_EXP],
    ]

    wedges, _, autotexts = pie_ax.pie(
        values,

        labels=None,

        colors=colors,

        autopct=make_pie_autopct(
            total_n
        ),

        startangle=90,
        counterclock=False,

        # Preserve the previous large pie size.
        radius=1.15,

        pctdistance=0.62,

        textprops={
            "fontsize":
                PIE_AUTOPCT_FONTSIZE,
        },

        wedgeprops={
            "edgecolor":
                "white",

            "linewidth":
                1.0,
        },
    )

    # -------------------------------------------------------------------------
    # Pie percentage/count text
    # -------------------------------------------------------------------------

    for wedge, autotxt in zip(
        wedges,
        autotexts,
    ):

        autotxt.set_fontsize(
            PIE_AUTOPCT_FONTSIZE
        )

        autotxt.set_fontweight(
            "bold"
        )

        autotxt.set_ha(
            "center"
        )

        autotxt.set_va(
            "center"
        )

        luminance = relative_luminance(
            wedge.get_facecolor()
        )

        autotxt.set_color(
            "white"
            if luminance < 0.55
            else "black"
        )

    # Preserve the previous breathing room around the pie.
    pie_ax.set_xlim(
        -1.25,
        1.25,
    )

    pie_ax.set_ylim(
        -1.22,
        1.22,
    )


# =============================================================================
# Figure creation
# =============================================================================

def create_combined_figure(
    data: dict[
        str,
        tuple[pd.DataFrame, pd.Series],
    ],
) -> plt.Figure:

    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,

        figsize=(
            FIG_WIDTH_CM / CM_PER_INCH,
            FIG_HEIGHT_CM / CM_PER_INCH,
        ),

        sharex=True,
        sharey=True,
    )

    # -------------------------------------------------------------------------
    # Draw transition panels
    # -------------------------------------------------------------------------

    for ax, transition in zip(
        axes,
        PANEL_ORDER,
    ):

        selected_df, model_counts = data[
            transition
        ]

        # H1 fitted functions
        add_selected_function_curves(
            ax,
            selected_df,
        )

        # Slightly strengthen fitted curves for poster display.
        for line in ax.lines:
            line.set_alpha(
                LINE_ALPHA
            )

            line.set_linewidth(
                LINE_WIDTH
            )

        # Main panel
        style_panel(
            ax,
            PANEL_TITLES[transition],
        )

        # Summary frame and pie
        add_large_poster_summary(
            ax,
            model_counts,
        )

    # =========================================================================
    # Shared axis labels
    # =========================================================================

    fig.supxlabel(
        "Chromosome number",
        fontsize=AXIS_LABEL_FONTSIZE,
        y=0.045,
    )

    fig.supylabel(
        "Transition rate",
        fontsize=AXIS_LABEL_FONTSIZE,
        x=0.006,
    )

    # =========================================================================
    # Shared model legend
    # =========================================================================

    add_shared_model_legend(
        fig
    )

    # =========================================================================
    # Overall figure layout
    # =========================================================================

    fig.subplots_adjust(
        left=0.045,
        right=0.996,

        bottom=0.14,

        top=0.82,

        wspace=0.08,
    )

    return fig


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    args = parse_args()

    fitted_files = {
        LABEL_DUPL:
            args.dupl_fitted_functions_file,

        LABEL_LOSS:
            args.loss_fitted_functions_file,

        LABEL_GAIN:
            args.gain_fitted_functions_file,
    }

    chosen_files = {
        LABEL_DUPL:
            args.dupl_chosen_model_file,

        LABEL_LOSS:
            args.loss_chosen_model_file,

        LABEL_GAIN:
            args.gain_chosen_model_file,
    }

    # -------------------------------------------------------------------------
    # Load all three transitions
    # -------------------------------------------------------------------------

    data = {
        transition:
            load_transition_data(
                fitted_file=fitted_files[
                    transition
                ],
                chosen_file=chosen_files[
                    transition
                ],
            )

        for transition in PANEL_ORDER
    }

    # -------------------------------------------------------------------------
    # Output paths
    # -------------------------------------------------------------------------

    output_file = (
        args.output_file
    )

    pdf_output_file = (
        args.pdf_output_file
        or output_file.with_suffix(
            ".pdf"
        )
    )

    ensure_dir(
        output_file.parent
    )

    ensure_dir(
        pdf_output_file.parent
    )

    # -------------------------------------------------------------------------
    # Build figure
    # -------------------------------------------------------------------------

    fig = create_combined_figure(
        data
    )

    # Deliberately avoid bbox_inches="tight".
    # This preserves the declared 88.33 cm physical width.
    fig.savefig(
        output_file,
        dpi=PNG_DPI,
        facecolor="white",
    )

    fig.savefig(
        pdf_output_file,
        facecolor="white",
    )

    plt.close(
        fig
    )

    # -------------------------------------------------------------------------
    # Console output
    # -------------------------------------------------------------------------

    print(
        "[✓] Combined WGD / Loss / Gain "
        "summary figure completed."
    )

    print(
        f"Physical size: "
        f"{FIG_WIDTH_CM:.2f} × "
        f"{FIG_HEIGHT_CM:.2f} cm"
    )

    print(
        f"Axes: "
        f"x=0-{FULL_X_MAX:g}, "
        f"y=0-{FULL_Y_MAX:g}"
    )

    print(
        f"PNG: {output_file}"
    )

    print(
        f"PDF: {pdf_output_file}"
    )

    for transition in PANEL_ORDER:

        selected_df, model_counts = data[
            transition
        ]

        total_n = int(
            model_counts.sum()
        )

        h1_n = int(
            model_counts[LABEL_LINEAR]
            + model_counts[LABEL_EXP]
        )

        print(
            f"{PANEL_TITLES[transition]}: "
            f"H1={h1_n}/{total_n} "
            f"({100.0 * h1_n / total_n:.1f}%), "
            f"curves={len(selected_df)}"
        )


if __name__ == "__main__":
    main()