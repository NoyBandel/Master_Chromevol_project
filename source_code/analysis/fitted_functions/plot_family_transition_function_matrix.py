#!/usr/bin/env python3

from pathlib import Path
from math import ceil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(
    "analysis/exponential_vs_linear_filtered/"
    "range25_events15_size50"
)

OUT_PDF = (
    ROOT
    / "all_retained_families_fitted_function_matrix.pdf"
)

TRANSITIONS = (
    "dupl",
    "loss",
    "gain",
)

DISPLAY_NAMES = {
    "dupl": "Duplication",
    "loss": "Descending dysploidy",
    "gain": "Ascending dysploidy",
}

ROWS_PER_PAGE = 7

N_X = 250


# ============================================================
# COLORS
# ============================================================

CONSTANT_COLOR = "#8A8A8A"
LINEAR_COLOR = "#1F77B4"
EXPONENTIAL_COLOR = "#2CA02C"

MODEL_COLORS = {
    "constant": CONSTANT_COLOR,
    "linear": LINEAR_COLOR,
    "exponential": EXPONENTIAL_COLOR,
}

NOT_RETAINED_BACKGROUND = "#F2F2F2"


# ============================================================
# COLUMN HELPERS
# ============================================================

def first_existing_column(df, candidates, required=True):
    """
    Return the first candidate column that exists.
    """

    for col in candidates:
        if col in df.columns:
            return col

    if required:
        raise KeyError(
            "\nNone of these columns were found:\n"
            f"{candidates}\n\n"
            "Available columns:\n"
            + "\n".join(df.columns)
        )

    return None


# ============================================================
# LOAD ONE TRANSITION
# ============================================================

def load_transition(transition):

    # --------------------------------------------------------
    # Main slope/model table
    #
    # Known to contain:
    #   overall_chosen_model
    #   constant_AICc
    #   linear_AICc
    #   exponential_AICc
    #   corrected Linear/Exponential parameters
    #   min_chrom / max_chrom
    # --------------------------------------------------------

    slope_file = (
        ROOT
        / transition
        / "slope_analysis"
        / "tables"
        / f"{transition}_exp_lin_slope_plotting_table.csv"
    )

    # Complete fitted-function parameter table
    param_file = (
        ROOT
        / transition
        / "model_selection"
        / f"{transition}_fitted_function_parameters.csv"
    )

    if not slope_file.exists():
        raise FileNotFoundError(slope_file)

    if not param_file.exists():
        raise FileNotFoundError(param_file)

    slope = pd.read_csv(slope_file)
    params = pd.read_csv(param_file)

    # Normalize family names
    slope["family_key"] = (
        slope["family_name"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    params["family_key"] = (
        params["family_name"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # --------------------------------------------------------
    # Remove Meliaceae defensively
    # --------------------------------------------------------

    slope = slope[
        slope["family_key"] != "meliaceae"
    ].copy()

    params = params[
        params["family_key"] != "meliaceae"
    ].copy()

    # --------------------------------------------------------
    # Identify parameter columns
    # --------------------------------------------------------
    const_p1_col = first_existing_column(
        params,
        [
            "constant_value_tree_scale_corrected",
            "constant_p1_tree_scale_corrected",
            "const_p1_tree_scale_corrected",
            "M0_p1_tree_scale_corrected",
            "constant_rate_tree_scale_corrected",
            "constant_p1",
            "const_p1",
            "M0_p1",
        ],
    )

    # We take Linear and Exponential parameters from the
    # slope table because their corrected columns are known.
    required_slope_columns = [
        "family_name",
        "family_key",
        "overall_chosen_model",
        "constant_AICc",
        "linear_AICc",
        "exponential_AICc",
        "lin_p1_tree_scale_corrected",
        "lin_slope_p2_tree_scale_corrected",
        "exp_p1_tree_scale_corrected",
        "exp_slope_p2",
        "min_chrom",
        "max_chrom",
    ]

    missing = [
        c
        for c in required_slope_columns
        if c not in slope.columns
    ]

    if missing:
        raise KeyError(
            f"\nMissing columns in {slope_file}:\n"
            + "\n".join(missing)
            + "\n\nAvailable columns:\n"
            + "\n".join(slope.columns)
        )

    # --------------------------------------------------------
    # Bring Constant p1 into slope table
    # --------------------------------------------------------

    const = params[
        [
            "family_key",
            const_p1_col,
        ]
    ].drop_duplicates(
        subset="family_key"
    )

    const = const.rename(
        columns={
            const_p1_col:
                "constant_p1_corrected"
        }
    )

    df = slope.merge(
        const,
        on="family_key",
        how="left",
        validate="one_to_one",
    )

    if df["constant_p1_corrected"].isna().any():

        bad = df.loc[
            df["constant_p1_corrected"].isna(),
            "family_name",
        ].tolist()

        raise ValueError(
            f"{transition}: missing Constant p1 for:\n"
            + "\n".join(bad)
        )

    return df


# ============================================================
# FUNCTION EVALUATION
# ============================================================

def evaluate_functions(row):

    xmin = float(row["min_chrom"])
    xmax = float(row["max_chrom"])

    x = np.linspace(
        xmin,
        xmax,
        N_X,
    )

    # Constant
    const_p1 = float(
        row["constant_p1_corrected"]
    )

    y_constant = np.full_like(
        x,
        const_p1,
        dtype=float,
    )

    # Linear
    lin_p1 = float(
        row["lin_p1_tree_scale_corrected"]
    )

    lin_p2 = float(
        row["lin_slope_p2_tree_scale_corrected"]
    )

    y_linear = (
        lin_p1
        + lin_p2 * (x - 1.0)
    )

    # Guard against tiny numerical negatives
    y_linear = np.maximum(
        y_linear,
        0.0,
    )

    # Exponential
    exp_p1 = float(
        row["exp_p1_tree_scale_corrected"]
    )

    exp_p2 = float(
        row["exp_slope_p2"]
    )

    y_exponential = (
        exp_p1
        * np.exp(
            exp_p2 * (x - 1.0)
        )
    )

    return {
        "x": x,
        "constant": y_constant,
        "linear": y_linear,
        "exponential": y_exponential,
    }


# ============================================================
# AICC BOX
# ============================================================

def add_aicc_box(ax, row, selected):

    values = {
        "constant":
            float(row["constant_AICc"]),

        "linear":
            float(row["linear_AICc"]),

        "exponential":
            float(row["exponential_AICc"]),
    }

    labels = {
        "constant": "Constant",
        "linear": "Linear",
        "exponential": "Exponential",
    }

    lines = []

    for model in (
        "constant",
        "linear",
        "exponential",
    ):

        marker = "●" if model == selected else " "

        lines.append(
            f"{marker} {labels[model]}: "
            f"{values[model]:.2f}"
        )

    text = "\n".join(lines)

    ax.text(
        0.98,
        0.97,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.7,
        linespacing=1.20,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="#777777",
            linewidth=0.7,
            alpha=0.92,
        ),
        zorder=10,
    )


# ============================================================
# DRAW ONE CELL
# ============================================================

def plot_cell(
    ax,
    row,
    transition,
    family_name,
):

    curves = evaluate_functions(
        row
    )

    selected = (
        str(
            row["overall_chosen_model"]
        )
        .strip()
        .lower()
    )

    # --------------------------------------------------------
    # Functions
    # --------------------------------------------------------

    for model in (
        "constant",
        "linear",
        "exponential",
    ):

        is_selected = (
            model == selected
        )

        ax.plot(
            curves["x"],
            curves[model],
            color=MODEL_COLORS[model],
            linestyle=(
                "-"
                if is_selected
                else "--"
            ),
            linewidth=(
                2.4
                if is_selected
                else 1.35
            ),
            alpha=(
                1.0
                if is_selected
                else 0.80
            ),
            zorder=(
                4
                if is_selected
                else 2
            ),
        )

    # --------------------------------------------------------
    # AICc box
    # --------------------------------------------------------

    add_aicc_box(
        ax,
        row,
        selected,
    )

    # --------------------------------------------------------
    # Axes
    # --------------------------------------------------------

    ax.set_xlim(
        float(row["min_chrom"]),
        float(row["max_chrom"]),
    )

    ax.set_ylim(
        bottom=0
    )

    ax.tick_params(
        axis="both",
        labelsize=7,
        length=2.5,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(
        axis="y",
        linewidth=0.35,
        alpha=0.25,
    )


# ============================================================
# NOT-RETAINED CELL
# ============================================================

def plot_not_retained(ax):

    ax.set_facecolor(
        NOT_RETAINED_BACKGROUND
    )

    ax.text(
        0.5,
        0.5,
        "Not retained",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9,
        color="#888888",
        fontstyle="italic",
    )

    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)


# ============================================================
# GLOBAL LEGEND
# ============================================================

def add_global_legend(fig):

    handles = [
        Line2D(
            [0],
            [0],
            color=CONSTANT_COLOR,
            linewidth=2.2,
            label="Constant",
        ),

        Line2D(
            [0],
            [0],
            color=LINEAR_COLOR,
            linewidth=2.2,
            label="Linear",
        ),

        Line2D(
            [0],
            [0],
            color=EXPONENTIAL_COLOR,
            linewidth=2.2,
            label="Exponential",
        ),

        Line2D(
            [0],
            [0],
            color="black",
            linewidth=2.2,
            linestyle="-",
            label="Selected",
        ),

        Line2D(
            [0],
            [0],
            color="black",
            linewidth=1.4,
            linestyle="--",
            label="Not selected",
        ),
    ]

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(
            0.5,
            0.006,
        ),
    )


# ============================================================
# MAIN PDF
# ============================================================

def create_pdf():

    # --------------------------------------------------------
    # Load all three transition datasets
    # --------------------------------------------------------

    data = {
        t: load_transition(t)
        for t in TRANSITIONS
    }

    # --------------------------------------------------------
    # Union of families retained for >=1 transition
    # --------------------------------------------------------

    family_display = {}

    all_family_keys = set()

    for t in TRANSITIONS:

        for _, row in data[t].iterrows():

            key = row["family_key"]

            all_family_keys.add(
                key
            )

            family_display[
                key
            ] = row["family_name"]

    families = sorted(
        all_family_keys,
        key=lambda x:
            family_display[x].lower(),
    )

    print(
        f"Families retained for >=1 transition: "
        f"{len(families)}"
    )

    # Lookup tables
    lookup = {}

    for t in TRANSITIONS:

        lookup[t] = (
            data[t]
            .set_index(
                "family_key",
                drop=False,
            )
        )

    # --------------------------------------------------------
    # Multipage PDF
    # --------------------------------------------------------

    n_pages = ceil(
        len(families)
        / ROWS_PER_PAGE
    )

    with PdfPages(
        OUT_PDF
    ) as pdf:

        for page_idx in range(
            n_pages
        ):

            start = (
                page_idx
                * ROWS_PER_PAGE
            )

            end = min(
                start
                + ROWS_PER_PAGE,
                len(families),
            )

            page_families = (
                families[
                    start:end
                ]
            )

            n_rows = len(
                page_families
            )

            fig, axes = plt.subplots(
                n_rows,
                3,
                figsize=(
                    15,
                    2.55 * n_rows + 1.3,
                ),
                squeeze=False,
            )

            # -----------------------------------------------
            # Column titles
            # -----------------------------------------------

            for j, transition in enumerate(
                TRANSITIONS
            ):

                axes[0, j].set_title(
                    DISPLAY_NAMES[
                        transition
                    ],
                    fontsize=13,
                    fontweight="bold",
                    pad=10,
                )

            # -----------------------------------------------
            # Cells
            # -----------------------------------------------

            for i, family_key in enumerate(
                page_families
            ):

                family_name = (
                    family_display[
                        family_key
                    ]
                )

                for j, transition in enumerate(
                    TRANSITIONS
                ):

                    ax = axes[i, j]

                    if (
                        family_key
                        not in
                        lookup[
                            transition
                        ].index
                    ):
                        plot_not_retained(
                            ax
                        )

                    else:

                        row = (
                            lookup[
                                transition
                            ].loc[
                                family_key
                            ]
                        )

                        # Defensive handling
                        # if duplicated somehow
                        if isinstance(
                            row,
                            pd.DataFrame,
                        ):
                            row = row.iloc[0]

                        plot_cell(
                            ax,
                            row,
                            transition,
                            family_name,
                        )

                    # Row labels
                    if j == 0:

                        ax.text(
                            -0.22,
                            0.5,
                            family_name,
                            transform=ax.transAxes,
                            ha="right",
                            va="center",
                            fontsize=9,
                            fontweight="bold",
                            clip_on=False,
                        )

                    # Y label only first column
                    if j == 0:
                        ax.set_ylabel(
                            "Transition rate",
                            fontsize=8,
                        )

                    # X label every cell
                    ax.set_xlabel(
                        "Chromosome number",
                        fontsize=8,
                    )

            # -----------------------------------------------
            # Page title / legend
            # -----------------------------------------------

            fig.suptitle(
                "Fitted chromosome-number-dependent functions",
                fontsize=15,
                fontweight="bold",
                y=0.995,
            )

            fig.text(
                0.995,
                0.995,
                f"Page {page_idx + 1}/{n_pages}",
                ha="right",
                va="top",
                fontsize=8,
            )

            add_global_legend(
                fig
            )

            fig.subplots_adjust(
                left=0.16,
                right=0.985,
                top=0.94,
                bottom=0.055,
                hspace=0.48,
                wspace=0.25,
            )

            pdf.savefig(
                fig,
                bbox_inches="tight",
            )

            plt.close(
                fig
            )

    print(
        f"\nSaved:\n{OUT_PDF}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    create_pdf()