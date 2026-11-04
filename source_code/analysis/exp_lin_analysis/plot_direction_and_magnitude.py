#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from source_code.analysis.exp_lin_analysis.exp_lin_analysis_constants import (
    EXP_EFFECTIVE_SLOPE_COL,
    LINEAR_P2_COL,
    NEGATIVE_SLOPE_LABEL,
    OVERALL_CHOSEN_MODEL_COL,
    POSITIVE_SLOPE_LABEL,
    SLOPE_ANALYSIS_SUBDIR,
    SLOPE_SIGN_COLOR_MAP,
    TABLES_SUBDIR,
)
from source_code.analysis.parameter_utils import (
    CORRECTED_PARAMETER_SCALE,
    resolve_parameter_column,
)
from source_code.constants import (
    LABEL_DUPL,
    LABEL_EXP,
    LABEL_GAIN,
    LABEL_LINEAR,
    LABEL_LOSS,
)

from source_code.logger import log_run


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(
    "analysis/exponential_vs_linear_filtered/"
    "range25_events15_size50"
)

OUT_PNG = (
    ROOT / "direction_by_functional_form.png"
)

OUT_PDF = (
    ROOT / "direction_by_functional_form.pdf"
)

OUT_CSV = (
    ROOT / "direction_by_functional_form_summary.csv"
)

TRANSITIONS = (
    LABEL_DUPL,
    LABEL_LOSS,
    LABEL_GAIN,
)

DISPLAY_NAMES = {
    LABEL_DUPL: "Duplication",
    LABEL_LOSS: "Descending dysploidy",
    LABEL_GAIN: "Ascending dysploidy",
}

PARAMETER_SCALE = CORRECTED_PARAMETER_SCALE

MODEL_HATCH = {
    LABEL_LINEAR: "",
    LABEL_EXP: "//",
}

BAR_HEIGHT = 0.58

plt.rcParams["hatch.linewidth"] = 0.75


# ============================================================
# INPUT
# ============================================================

def get_slope_table_path(transition):

    return (
        ROOT
        / transition
        / SLOPE_ANALYSIS_SUBDIR
        / TABLES_SUBDIR
        / f"{transition}_exp_lin_slope_plotting_table.csv"
    )


def load_selected_h1_data():

    linear_slope_col = resolve_parameter_column(
        LINEAR_P2_COL,
        PARAMETER_SCALE,
    )

    exp_slope_col = resolve_parameter_column(
        EXP_EFFECTIVE_SLOPE_COL,
        PARAMETER_SCALE,
    )

    all_rows = []

    for transition in TRANSITIONS:

        path = get_slope_table_path(
            transition
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"Missing slope table:\n{path}"
            )

        df = pd.read_csv(
            path
        )

        required_columns = [
            OVERALL_CHOSEN_MODEL_COL,
            linear_slope_col,
            exp_slope_col,
        ]

        missing_columns = [
            col
            for col in required_columns
            if col not in df.columns
        ]

        if missing_columns:
            raise KeyError(
                f"{path} is missing columns: "
                f"{missing_columns}"
            )

        df = df[
            df[
                OVERALL_CHOSEN_MODEL_COL
            ].isin(
                [
                    LABEL_LINEAR,
                    LABEL_EXP,
                ]
            )
        ].copy()

        df[linear_slope_col] = (
            pd.to_numeric(
                df[linear_slope_col],
                errors="coerce",
            )
        )

        df[exp_slope_col] = (
            pd.to_numeric(
                df[exp_slope_col],
                errors="coerce",
            )
        )

        def get_selected_slope(row):

            if (
                row[
                    OVERALL_CHOSEN_MODEL_COL
                ]
                == LABEL_LINEAR
            ):
                return row[
                    linear_slope_col
                ]

            return row[
                exp_slope_col
            ]

        df["selected_slope"] = (
            df.apply(
                get_selected_slope,
                axis=1,
            )
        )

        df = df[
            df["selected_slope"].notna()
            & (df["selected_slope"] != 0)
        ].copy()

        df["direction"] = np.where(
            df["selected_slope"] > 0,
            POSITIVE_SLOPE_LABEL,
            NEGATIVE_SLOPE_LABEL,
        )

        df["transition"] = transition

        all_rows.append(
            df
        )

    return pd.concat(
        all_rows,
        ignore_index=True,
    )


# ============================================================
# SUMMARY
# ============================================================

def build_summary(df):

    rows = []

    for transition in TRANSITIONS:

        transition_df = df[
            df["transition"] == transition
        ]

        total_n = len(
            transition_df
        )

        for direction in (
            POSITIVE_SLOPE_LABEL,
            NEGATIVE_SLOPE_LABEL,
        ):

            for model in (
                LABEL_LINEAR,
                LABEL_EXP,
            ):

                n = len(
                    transition_df[
                        (
                            transition_df[
                                "direction"
                            ]
                            == direction
                        )
                        & (
                            transition_df[
                                OVERALL_CHOSEN_MODEL_COL
                            ]
                            == model
                        )
                    ]
                )

                pct = (
                    100 * n / total_n
                    if total_n
                    else 0
                )

                rows.append(
                    {
                        "transition":
                            transition,
                        "direction":
                            direction,
                        "model":
                            model,
                        "n":
                            n,
                        "total_n":
                            total_n,
                        "pct":
                            pct,
                    }
                )

    return pd.DataFrame(
        rows
    )


# ============================================================
# PLOT
# ============================================================

def plot_direction_by_form(
    summary,
):

    fig, ax = plt.subplots(
        figsize=(10.4, 5.3)
    )

    y_positions = np.arange(
        len(TRANSITIONS)
    )

    # Increasing first, then decreasing.
    # Within each direction:
    # Linear first, Exponential second.
    segment_order = (
        (
            POSITIVE_SLOPE_LABEL,
            LABEL_LINEAR,
        ),
        (
            POSITIVE_SLOPE_LABEL,
            LABEL_EXP,
        ),
        (
            NEGATIVE_SLOPE_LABEL,
            LABEL_LINEAR,
        ),
        (
            NEGATIVE_SLOPE_LABEL,
            LABEL_EXP,
        ),
    )

    for y, transition in zip(
        y_positions,
        TRANSITIONS,
    ):

        transition_df = summary[
            summary["transition"]
            == transition
        ]

        left = 0.0

        segments = []

        # --------------------------------------------
        # Draw bar segments
        # --------------------------------------------

        for direction, model in segment_order:

            row = transition_df[
                (
                    transition_df[
                        "direction"
                    ]
                    == direction
                )
                & (
                    transition_df[
                        "model"
                    ]
                    == model
                )
            ].iloc[0]

            width = float(
                row["pct"]
            )

            count = int(
                row["n"]
            )

            if width <= 0:
                continue

            center = (
                left
                + width / 2
            )

            ax.barh(
                y,
                width,
                left=left,
                height=BAR_HEIGHT,
                color=SLOPE_SIGN_COLOR_MAP[
                    direction
                ],
                hatch=MODEL_HATCH[
                    model
                ],
                edgecolor="#4A4A4A",
                linewidth=0.65,
                zorder=3,
            )

            segments.append(
                {
                    "center":
                        center,
                    "width":
                        width,
                    "count":
                        count,
                    "direction":
                        direction,
                    "model":
                        model,
                }
            )

            left += width

        # --------------------------------------------
        # Draw labels below bar
        # --------------------------------------------

        # Separate handling for very small adjacent segments.
        # These would otherwise overlap even when vertically staggered.
        small_segments = [
            s for s in segments
            if s["width"] < 7.0
        ]

        normal_segments = [
            s for s in segments
            if s["width"] >= 7.0
        ]

        # --------------------------------------------
        # Normal-sized segments:
        # label directly below segment center
        # --------------------------------------------

        for segment in normal_segments:

            label_y = (
                y
                + BAR_HEIGHT / 2
                + 0.12
            )

            ax.text(
                segment["center"],
                label_y,
                (
                    f"{segment['width']:.1f}%\n"
                    f"({segment['count']})"
                ),
                ha="center",
                va="top",
                fontsize=8.7,
                color="#222222",
                linespacing=0.90,
                clip_on=False,
                zorder=6,
            )

        # --------------------------------------------
        # Small adjacent segments:
        # spread labels horizontally and connect with
        # short leader lines
        # --------------------------------------------

        if small_segments:

            small_segments = sorted(
                small_segments,
                key=lambda s: s["center"],
            )

            # Horizontal spacing between displayed labels.
            # Keeps the labels readable even if the actual
            # segment centers are only a few percent apart.
            label_spacing = 6.0

            original_centers = np.array(
                [
                    s["center"]
                    for s in small_segments
                ]
            )

            mean_center = (
                original_centers.mean()
            )

            n_small = len(
                small_segments
            )

            label_x_positions = (
                mean_center
                + (
                    np.arange(n_small)
                    - (n_small - 1) / 2
                )
                * label_spacing
            )

            # Prevent labels from extending outside plotting range.
            label_x_positions = np.clip(
                label_x_positions,
                2.5,
                97.5,
            )

            label_y = (
                y
                + BAR_HEIGHT / 2
                + 0.13
            )

            line_start_y = (
                y
                + BAR_HEIGHT / 2
                + 0.015
            )

            line_end_y = (
                label_y - 0.025
            )

            for (
                segment,
                label_x,
            ) in zip(
                small_segments,
                label_x_positions,
            ):

                # Leader line from true segment center
                # to the relocated label.
                ax.annotate(
                    "",
                    xy=(
                        segment["center"],
                        line_start_y,
                    ),
                    xytext=(
                        label_x,
                        line_end_y,
                    ),
                    arrowprops={
                        "arrowstyle": "-",
                        "color": "#666666",
                        "linewidth": 0.7,
                    },
                    annotation_clip=False,
                    zorder=5,
                )

                ax.text(
                    label_x,
                    label_y,
                    (
                        f"{segment['width']:.1f}%\n"
                        f"({segment['count']})"
                    ),
                    ha="center",
                    va="top",
                    fontsize=8.5,
                    color="#222222",
                    linespacing=0.90,
                    clip_on=False,
                    zorder=6,
                )

    # ========================================================
    # AXES
    # ========================================================

    ax.set_xlim(
        0,
        100,
    )

    ax.set_xticks(
        [
            0,
            25,
            50,
            75,
            100,
        ]
    )

    ax.set_xlabel(
        "H1 families (%)",
        fontsize=11.5,
    )

    ax.set_yticks(
        y_positions
    )

    y_labels = []

    for transition in TRANSITIONS:

        transition_df = summary[
            summary["transition"]
            == transition
        ]

        total_n = int(
            transition_df[
                "total_n"
            ].iloc[0]
        )

        y_labels.append(
            (
                f"{DISPLAY_NAMES[transition]} "
                f"(n={total_n})"
            )
        )

    ax.set_yticklabels(
        y_labels,
        fontsize=10.5,
    )

    # Extra vertical space below each bar for labels.
    ax.set_ylim(
        len(TRANSITIONS) - 0.05,
        -0.72,
    )

    ax.grid(
        axis="x",
        linestyle="--",
        linewidth=0.65,
        color="#DDDDDD",
        zorder=0,
    )

    ax.set_axisbelow(
        True
    )

    # ========================================================
    # LEGENDS
    # ========================================================

    direction_handles = [
        Patch(
            facecolor=SLOPE_SIGN_COLOR_MAP[
                POSITIVE_SLOPE_LABEL
            ],
            edgecolor="#4A4A4A",
            label="Increasing",
        ),
        Patch(
            facecolor=SLOPE_SIGN_COLOR_MAP[
                NEGATIVE_SLOPE_LABEL
            ],
            edgecolor="#4A4A4A",
            label="Decreasing",
        ),
    ]

    form_handles = [
        Patch(
            facecolor="white",
            edgecolor="#4A4A4A",
            hatch=MODEL_HATCH[
                LABEL_LINEAR
            ],
            label="Linear",
        ),
        Patch(
            facecolor="white",
            edgecolor="#4A4A4A",
            hatch=MODEL_HATCH[
                LABEL_EXP
            ],
            label="Exponential",
        ),
    ]

    direction_legend = ax.legend(
        handles=direction_handles,
        title="Direction",
        loc="lower center",
        bbox_to_anchor=(
            0.36,
            1.03,
        ),
        ncol=2,
        frameon=False,
        fontsize=9.2,
        title_fontsize=9.2,
        handlelength=1.9,
        columnspacing=1.4,
    )

    ax.add_artist(
        direction_legend
    )

    ax.legend(
        handles=form_handles,
        title="Functional form",
        loc="lower center",
        bbox_to_anchor=(
            0.73,
            1.03,
        ),
        ncol=2,
        frameon=False,
        fontsize=9.2,
        title_fontsize=9.2,
        handlelength=1.9,
        columnspacing=1.4,
    )

    # ========================================================
    # STYLE
    # ========================================================

    ax.spines["top"].set_visible(
        False
    )

    ax.spines["right"].set_visible(
        False
    )

    ax.spines["left"].set_visible(
        False
    )

    ax.spines["bottom"].set_color(
        "#666666"
    )

    ax.tick_params(
        axis="y",
        length=0,
        pad=7,
    )

    ax.tick_params(
        axis="x",
        labelsize=9.8,
    )

    fig.subplots_adjust(
        left=0.27,
        right=0.985,
        bottom=0.13,
        top=0.80,
    )

    fig.savefig(
        OUT_PNG,
        dpi=400,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        OUT_PDF,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(
        fig
    )


# ============================================================
# CONSOLE SUMMARY
# ============================================================

def print_summary(
    summary,
):

    print()
    print(
        "Direction × functional form"
    )
    print(
        "=" * 72
    )

    for transition in TRANSITIONS:

        transition_df = summary[
            summary["transition"]
            == transition
        ]

        total_n = int(
            transition_df[
                "total_n"
            ].iloc[0]
        )

        print()
        print(
            f"{DISPLAY_NAMES[transition]} "
            f"(n={total_n})"
        )

        for direction in (
            POSITIVE_SLOPE_LABEL,
            NEGATIVE_SLOPE_LABEL,
        ):

            display_direction = (
                "Increasing"
                if direction
                == POSITIVE_SLOPE_LABEL
                else "Decreasing"
            )

            direction_df = (
                transition_df[
                    transition_df[
                        "direction"
                    ]
                    == direction
                ]
            )

            direction_n = int(
                direction_df[
                    "n"
                ].sum()
            )

            direction_pct = (
                100
                * direction_n
                / total_n
            )

            print(
                f"  {display_direction}: "
                f"{direction_n}/{total_n} "
                f"({direction_pct:.1f}%)"
            )

            for model in (
                LABEL_LINEAR,
                LABEL_EXP,
            ):

                row = direction_df[
                    direction_df[
                        "model"
                    ]
                    == model
                ].iloc[0]

                print(
                    f"    {model:12s}: "
                    f"{int(row['n']):2d} "
                    f"({float(row['pct']):5.1f}%)"
                )


# ============================================================
# MAIN
# ============================================================

def main():

    selected_df = (
        load_selected_h1_data()
    )

    summary = (
        build_summary(
            selected_df
        )
    )

    summary.to_csv(
        OUT_CSV,
        index=False,
    )

    plot_direction_by_form(
        summary
    )

    print_summary(
        summary
    )

    print()
    print("Saved:")
    print(
        f"  {OUT_PNG}"
    )
    print(
        f"  {OUT_PDF}"
    )
    print(
        f"  {OUT_CSV}"
    )
    log_run(
        step="analysis",
        script=Path(__file__),
        params={
            "analysis_root": str(ROOT),
            "parameter_scale": str(PARAMETER_SCALE),
            "transitions": list(TRANSITIONS),
            "input_slope_tables": [
                str(get_slope_table_path(t))
                for t in TRANSITIONS
            ],
            "selected_h1_families": {
                t: int((selected_df["transition"] == t).sum())
                for t in TRANSITIONS
            },
        },
        outputs=[
            str(OUT_PNG),
            str(OUT_PDF),
            str(OUT_CSV),
        ],
        description=(
            "Generated the direction-by-functional-form analysis "
            "for chromosome-number-dependent transition models, "
            "using tree-scale-corrected slope estimates."
        ),
        notes=(
            "Meliaceae excluded following model-diagnostic analyses "
            "indicating unstable parameter estimates and sensitivity "
            "to an extreme chromosome count (n=180). "
            "The filtered family sets contain 60 duplication, "
            "51 loss, and 29 gain families."
        ),
        log_relative_path=Path(
            "analysis/exp_lin_analysis/direction_and_magnitude.log"
        ),
    )


if __name__ == "__main__":
    main()