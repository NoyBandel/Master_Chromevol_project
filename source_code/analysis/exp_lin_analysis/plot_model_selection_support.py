#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from source_code.logger import log_run


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(
    "analysis/exponential_vs_linear_filtered/"
    "range25_events15_size50"
)

OUT_PNG_COLOR = ROOT / "constant_vs_best_h1_delta_aicc.png"
OUT_PNG_BW = ROOT / "constant_vs_best_h1_delta_aicc_bw.png"

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


# ============================================================
# COLOR PALETTES
# ============================================================

# ---- color version ----
COLOR_STYLE = {
    "constant": {
        "facecolor": "#8A8A8A",
        "marker": "o",
        "label": "Constant favored",
    },
    "linear": {
        "facecolor": "#1F77B4",
        "marker": "o",
        "label": "Linear H1 favored",
    },
    "exponential": {
        "facecolor": "#2CA02C",
        "marker": "o",
        "label": "Exponential H1 favored",
    },
    "constant_background": "#F3F3F3",
    "dependence_background": "#F4F8FB",
}

# ---- black-and-white version ----
# Use both grayscale + shape differences
BW_STYLE = {
    "constant": {
        "facecolor": "#BDBDBD",
        "marker": "o",
        "label": "Constant favored",
    },
    "linear": {
        "facecolor": "#6F6F6F",
        "marker": "^",
        "label": "Linear H1 favored",
    },
    "exponential": {
        "facecolor": "#1A1A1A",
        "marker": "s",
        "label": "Exponential H1 favored",
    },
    "constant_background": "#F2F2F2",
    "dependence_background": "#FAFAFA",
}


# ============================================================
# SUPPORT THRESHOLDS
# ============================================================

SUPPORT_THRESHOLDS = (
    2,
    4,
    10,
)


# ============================================================
# DATA LOADING
# ============================================================

def load_model_selection_data():
    """
    Load evaluable families and calculate signed ΔAICc:

        delta_h1 =
            AICc(Constant)
            - AICc(best dependent model)

    Interpretation:
        delta_h1 > 0  -> best dependent model favored
        delta_h1 < 0  -> Constant favored
    """

    rows = []

    for transition in TRANSITIONS:

        path = (
            ROOT
            / transition
            / "model_selection"
            / (
                f"{transition}_core_table_"
                "with_dependence_detection_with_ignore.csv"
            )
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"Missing file:\n{path}"
            )

        df = pd.read_csv(path)

        # ----------------------------------------------------
        # Exclude Ignore-selected cases
        # ----------------------------------------------------

        df = df[
            df["chosen_function_label"]
            .astype(str)
            .str.lower()
            != "ignore"
        ].copy()

        # ----------------------------------------------------
        # Best H1 model
        # ----------------------------------------------------

        df["best_dependent_AICc"] = df[
            [
                "M1_linear_AICc",
                "M1_exp_AICc",
            ]
        ].min(axis=1)

        df["best_dependent_model"] = np.where(
            df["M1_linear_AICc"]
            <= df["M1_exp_AICc"],
            "Linear",
            "Exponential",
        )

        # ----------------------------------------------------
        # Signed Constant-vs-H1 ΔAICc
        # Positive -> H1 favored
        # Negative -> Constant favored
        # ----------------------------------------------------

        df["delta_h1"] = (
            df["M0_AICc"]
            - df["best_dependent_AICc"]
        )

        # ----------------------------------------------------
        # Existing H1 classification
        # ----------------------------------------------------

        df["is_h1"] = (
            df["dependence_detection_status"]
            .astype(str)
            .str.startswith("Dependence supported")
        )

        # ----------------------------------------------------
        # Collect rows
        # ----------------------------------------------------

        for _, row in df.iterrows():
            rows.append(
                {
                    "transition": transition,
                    "family_name": row["family_name"],
                    "delta_h1": row["delta_h1"],
                    "best_dependent_model": row["best_dependent_model"],
                    "is_h1": row["is_h1"],
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# SUMMARY OUTPUT
# ============================================================

def print_summary(df):

    print()
    print("=" * 78)
    print("CONSTANT vs BEST H1 MODEL")
    print("=" * 78)

    for transition in TRANSITIONS:

        sub = df[
            df["transition"] == transition
        ].copy()

        h1 = sub[
            sub["is_h1"]
        ].copy()

        h0 = sub[
            ~sub["is_h1"]
        ].copy()

        n_linear = (
            h1["best_dependent_model"] == "Linear"
        ).sum()

        n_exp = (
            h1["best_dependent_model"] == "Exponential"
        ).sum()

        print()
        print("-" * 78)
        print(DISPLAY_NAMES[transition])
        print("-" * 78)

        print(f"Evaluable families: {len(sub)}")
        print(f"H1 families: {len(h1)}")
        print(f"Constant families: {len(h0)}")
        print(f"Median signed ΔAICc: {sub['delta_h1'].median():.2f}")

        if len(h1) > 0:
            print(
                f"H1 Linear: {n_linear}/{len(h1)} "
                f"({100 * n_linear / len(h1):.1f}%)"
            )
            print(
                f"H1 Exponential: {n_exp}/{len(h1)} "
                f"({100 * n_exp / len(h1):.1f}%)"
            )


# ============================================================
# MEDIAN + IQR MARKER
# ============================================================

def add_median_iqr(ax, position, values):

    q1 = values.quantile(0.25)
    median = values.median()
    q3 = values.quantile(0.75)

    # IQR vertical line
    ax.plot(
        [position, position],
        [q1, q3],
        color="black",
        linewidth=3,
        zorder=5,
    )

    # Median horizontal line
    ax.plot(
        [position - 0.10, position + 0.10],
        [median, median],
        color="black",
        linewidth=3,
        zorder=5,
    )


# ============================================================
# POINT STYLE
# ============================================================

def get_point_style(row, style_map):
    """
    Return marker-face color and marker shape for a row.
    """

    if row["delta_h1"] < 0:
        return style_map["constant"]

    if row["best_dependent_model"] == "Linear":
        return style_map["linear"]

    return style_map["exponential"]


# ============================================================
# SHARED PLOTTER
# ============================================================

def plot_constant_vs_best_h1(
    df,
    out_path,
    style_map,
):
    fig, ax = plt.subplots(figsize=(8.8, 6.2))

    positions = np.arange(
        1,
        len(TRANSITIONS) + 1
    )

    # --------------------------------------------------------
    # Y-axis limits
    # --------------------------------------------------------

    y_min = df["delta_h1"].min()
    y_max = df["delta_h1"].max()
    y_range = y_max - y_min
    margin = 0.08 * y_range

    lower = y_min - margin
    upper = y_max + margin

    # --------------------------------------------------------
    # Background shading
    # --------------------------------------------------------

    ax.axhspan(
        lower,
        0,
        color=style_map["constant_background"],
        zorder=0,
    )

    ax.axhspan(
        0,
        upper,
        color=style_map["dependence_background"],
        zorder=0,
    )

    # --------------------------------------------------------
    # Family-level points
    # --------------------------------------------------------

    rng = np.random.default_rng(42)

    for position, transition in zip(
        positions,
        TRANSITIONS,
    ):

        sub = df[
            df["transition"] == transition
        ].copy()

        jitter = rng.uniform(
            -0.16,
            0.16,
            size=len(sub),
        )

        for dx, (_, row) in zip(
            jitter,
            sub.iterrows(),
        ):
            point_style = get_point_style(
                row,
                style_map,
            )

            ax.scatter(
                position + dx,
                row["delta_h1"],
                s=52,
                facecolor=point_style["facecolor"],
                marker=point_style["marker"],
                edgecolor="black",
                linewidth=0.55,
                alpha=0.95,
                zorder=3,
            )

        add_median_iqr(
            ax,
            position,
            sub["delta_h1"],
        )

    # --------------------------------------------------------
    # Equal-support line
    # --------------------------------------------------------

    ax.axhline(
        0,
        color="black",
        linewidth=2.0,
        zorder=2,
    )

    # --------------------------------------------------------
    # Support thresholds
    # --------------------------------------------------------

    for threshold in SUPPORT_THRESHOLDS:
        ax.axhline(
            threshold,
            color="gray",
            linestyle="--",
            linewidth=0.8,
            alpha=0.6,
            zorder=1,
        )

    # --------------------------------------------------------
    # Axes
    # --------------------------------------------------------

    ax.set_xlim(0.55, 3.45)
    ax.set_ylim(lower, upper)

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [DISPLAY_NAMES[t] for t in TRANSITIONS],
        fontsize=10.5,
    )

    ax.set_ylabel(
        "Signed ΔAICc\n"
        "(Constant − best H1 model)",
        fontsize=11.5,
    )

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    handles = [
        Line2D(
            [0], [0],
            marker=style_map["constant"]["marker"],
            linestyle="",
            markerfacecolor=style_map["constant"]["facecolor"],
            markeredgecolor="black",
            markersize=7,
            label=style_map["constant"]["label"],
        ),
        Line2D(
            [0], [0],
            marker=style_map["linear"]["marker"],
            linestyle="",
            markerfacecolor=style_map["linear"]["facecolor"],
            markeredgecolor="black",
            markersize=7,
            label=style_map["linear"]["label"],
        ),
        Line2D(
            [0], [0],
            marker=style_map["exponential"]["marker"],
            linestyle="",
            markerfacecolor=style_map["exponential"]["facecolor"],
            markeredgecolor="black",
            markersize=7,
            label=style_map["exponential"]["label"],
        ),
        Line2D(
            [0], [0],
            color="black",
            linewidth=3,
            label="Median and IQR",
        ),
    ]

    ax.legend(
        handles=handles,
        loc="upper right",
        frameon=True,
        fancybox=False,
        facecolor="white",
        edgecolor="#808080",
        framealpha=1,
        fontsize=8.8,
        borderpad=0.7,
        labelspacing=0.5,
    )

    # --------------------------------------------------------
    # Style
    # --------------------------------------------------------

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        axis="y",
        labelsize=10,
    )

    ax.tick_params(
        axis="x",
        length=0,
    )

    fig.tight_layout(pad=1.2)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    fig.savefig(
        out_path,
        dpi=400,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print()
    print(f"Saved: {out_path}")


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_model_selection_data()

    # Verify revised dataset after Meliaceae exclusion
    expected = {
        "dupl": 60,
        "loss": 51,
        "gain": 29,
    }

    for transition, n in expected.items():
        subset = df[
            df["transition"] == transition
        ]

        assert len(subset) == n, (
            f"{transition}: expected {n} families, "
            f"got {len(subset)}"
        )

        assert not subset["family_name"].astype(str).str.lower().eq(
            "meliaceae"
        ).any(), f"Meliaceae remains in {transition}"

    print_summary(df)

    # Color plot
    plot_constant_vs_best_h1(
        df=df,
        out_path=OUT_PNG_COLOR,
        style_map=COLOR_STYLE,
    )

    # Black-and-white plot
    plot_constant_vs_best_h1(
        df=df,
        out_path=OUT_PNG_BW,
        style_map=BW_STYLE,
    )

    input_files = [
        ROOT / transition / "model_selection"
        / f"{transition}_core_table_with_dependence_detection_with_ignore.csv"
        for transition in TRANSITIONS
    ]

    log_run(
        step="analysis",
        script=Path(__file__),
        params={
            "analysis_root": str(ROOT),
            "input_files": [str(p) for p in input_files],
            "transitions": list(TRANSITIONS),
            "support_thresholds": list(SUPPORT_THRESHOLDS),
            "family_counts": expected,
            "output_files": [
                str(OUT_PNG_COLOR),
                str(OUT_PNG_BW),
            ],
        },
        outputs=[
            str(OUT_PNG_COLOR),
            str(OUT_PNG_BW),
        ],
        description=(
            "Generated color and black-and-white signed ΔAICc figures "
            "comparing the Constant model with the best "
            "chromosome-number-dependent model "
            "(Linear or Exponential)."
        ),
        notes=(
            "Meliaceae excluded following model-diagnostic analyses "
            "indicating unstable parameter estimates and sensitivity "
            "to an extreme chromosome count (n=180). "
            "Analysis uses 60 duplication, 51 loss, and 29 gain families."
        ),
        log_relative_path=Path(
            "analysis/exp_lin_analysis/constant_vs_best_h1.log"
        ),
    )


if __name__ == "__main__":
    main()