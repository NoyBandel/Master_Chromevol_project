from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch, Rectangle

from source_code.analysis.exp_lin_analysis.exp_lin_analysis_constants import (
    EXP_EFFECTIVE_SLOPE_COL,
    LINEAR_P2_COL,
    OVERALL_CHOSEN_MODEL_COL,
    SLOPE_ANALYSIS_SUBDIR,
    TABLES_SUBDIR,
)
from source_code.analysis.parameter_utils import (
    CORRECTED_PARAMETER_SCALE,
    VALID_PARAMETER_SCALES,
    resolve_parameter_column,
)
from source_code.analysis.plot_utils import ensure_dir
from source_code.constants import (
    LABEL_DUPL,
    LABEL_EXP,
    LABEL_GAIN,
    LABEL_LINEAR,
    LABEL_LOSS,
)
from source_code.logger import log_run


# ============================================================
# Configuration
# ============================================================

TRANSITION_ORDER: tuple[str, str, str] = (
    LABEL_DUPL,
    LABEL_LOSS,
    LABEL_GAIN,
)

TRANSITION_DISPLAY: dict[str, str] = {
    LABEL_DUPL: "Duplication",
    LABEL_LOSS: "Loss",
    LABEL_GAIN: "Gain",
}

MODEL_DISPLAY: dict[str, str] = {
    LABEL_LINEAR: "Linear",
    LABEL_EXP: "Exponential",
}

H1_LABEL: str = "H₁"


# ============================================================
# Bar layout
# ============================================================

# H1 remains visually more prominent
H1_BAR_HEIGHT: float = 0.78
MODEL_BAR_HEIGHT: float = 0.46

# Space between different transition groups
TRANSITION_GROUP_GAP: float = 0.95


def build_y_positions() -> dict[str, tuple[float, float, float]]:
    """
    Construct y positions so that H1, Linear, and Exponential bars
    touch exactly within each transition, while leaving a larger
    whitespace gap between transition groups.
    """

    positions: dict[str, tuple[float, float, float]] = {}

    current_h1_y = 0.0

    for transition in TRANSITION_ORDER:
        # H1 and Linear touch:
        # distance between centers = half-height(H1) + half-height(model)
        linear_y = current_h1_y + (
            H1_BAR_HEIGHT + MODEL_BAR_HEIGHT
        ) / 2

        # Linear and Exponential have equal heights, so their centers
        # differ by exactly one model-bar height.
        exp_y = linear_y + MODEL_BAR_HEIGHT

        positions[transition] = (
            current_h1_y,
            linear_y,
            exp_y,
        )

        # Bottom edge of current Exponential bar
        current_group_bottom = exp_y + MODEL_BAR_HEIGHT / 2

        # Top edge of next H1 bar should be separated by
        # TRANSITION_GROUP_GAP.
        current_h1_y = (
            current_group_bottom
            + TRANSITION_GROUP_GAP
            + H1_BAR_HEIGHT / 2
        )

    return positions


Y_POSITIONS = build_y_positions()


# ============================================================
# Colors
# ============================================================

POSITIVE_COLOR = "#EDC353"
NEGATIVE_COLOR = "#B090E0"

# Both fills are light enough that dark text provides better contrast
BAR_TEXT_COLOR = "#222222"

# H1 emphasis
H1_OUTLINE_COLOR = "#1F1F1F"
H1_OUTLINE_WIDTH = 2.4


# ============================================================
# Layout / typography
# ============================================================

FIGSIZE = (11.4, 6.4)

TITLE_FONTSIZE = 18
LEGEND_FONTSIZE = 11
TRANSITION_LABEL_FONTSIZE = 13

H1_ROW_LABEL_FONTSIZE = 12.0
MODEL_ROW_LABEL_FONTSIZE = 10.2

AXIS_LABEL_FONTSIZE = 12
TICK_LABEL_FONTSIZE = 10.5

TRANSITION_LABEL_X = -0.35
ROW_LABEL_X = -0.05


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the main combined poster figure showing the "
            "direction of chromosome-number dependence."
        )
    )

    parser.add_argument(
        "--exp-root",
        type=Path,
        required=True,
        help=(
            "Root EXP-vs-LINEAR analysis directory, e.g. "
            "analysis/exponential_vs_linear_filtered/"
            "range25_events15_size50"
        ),
    )

    parser.add_argument(
        "--parameter-scale",
        choices=VALID_PARAMETER_SCALES,
        default=CORRECTED_PARAMETER_SCALE,
        help="Parameter scale to use. Poster figure should use corrected.",
    )

    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Output PNG file path.",
    )

    parser.add_argument(
        "--pdf-output-file",
        type=Path,
        default=None,
        help="Optional PDF output path.",
    )

    parser.add_argument(
        "--log-note",
        type=str,
        default="",
        help="Optional note to append to log_run().",
    )

    return parser.parse_args()


# ============================================================
# Input
# ============================================================

def get_slope_table_path(
    exp_root: Path,
    transition: str,
) -> Path:
    return (
        exp_root
        / transition
        / SLOPE_ANALYSIS_SUBDIR
        / TABLES_SUBDIR
        / f"{transition}_exp_lin_slope_plotting_table.csv"
    )


def load_slope_table(
    exp_root: Path,
    transition: str,
    parameter_scale: str,
) -> tuple[pd.DataFrame, Path, str, str]:
    input_file = get_slope_table_path(
        exp_root=exp_root,
        transition=transition,
    )

    if not input_file.is_file():
        raise FileNotFoundError(
            "Slope plotting table not found:\n"
            f"{input_file}"
        )

    linear_slope_col = resolve_parameter_column(
        LINEAR_P2_COL,
        parameter_scale,
    )

    exp_slope_col = resolve_parameter_column(
        EXP_EFFECTIVE_SLOPE_COL,
        parameter_scale,
    )

    df = pd.read_csv(input_file)

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
            f"{input_file} is missing required columns: {missing_columns}"
        )

    df = df.copy()

    df[linear_slope_col] = pd.to_numeric(
        df[linear_slope_col],
        errors="coerce",
    )

    df[exp_slope_col] = pd.to_numeric(
        df[exp_slope_col],
        errors="coerce",
    )

    return df, input_file, linear_slope_col, exp_slope_col


# ============================================================
# Summaries
# ============================================================

def summarize_selected_model(
    df: pd.DataFrame,
    transition: str,
    model: str,
    slope_col: str,
) -> dict[str, object]:
    selected_df = df.loc[
        df[OVERALL_CHOSEN_MODEL_COL] == model
    ].copy()

    selected_n = len(selected_df)

    valid_df = selected_df.loc[
        selected_df[slope_col].notna()
        & (selected_df[slope_col] != 0)
    ].copy()

    positive_n = int((valid_df[slope_col] > 0).sum())
    negative_n = int((valid_df[slope_col] < 0).sum())
    valid_n = positive_n + negative_n

    return {
        "transition": transition,
        "row_type": "model",
        "model": model,
        "positive_n": positive_n,
        "negative_n": negative_n,
        "valid_n": valid_n,
        "selected_n": selected_n,
        "excluded_n": selected_n - valid_n,
    }


def add_percentages(
    row: dict[str, object],
) -> dict[str, object]:
    total = int(row["valid_n"])
    positive_n = int(row["positive_n"])
    negative_n = int(row["negative_n"])

    if total == 0:
        positive_pct = 0.0
        negative_pct = 0.0
    else:
        positive_pct = 100.0 * positive_n / total
        negative_pct = 100.0 * negative_n / total

    row["positive_pct"] = positive_pct
    row["negative_pct"] = negative_pct

    return row


def build_transition_rows(
    df: pd.DataFrame,
    transition: str,
    linear_slope_col: str,
    exp_slope_col: str,
) -> list[dict[str, object]]:
    linear_row = summarize_selected_model(
        df=df,
        transition=transition,
        model=LABEL_LINEAR,
        slope_col=linear_slope_col,
    )

    exp_row = summarize_selected_model(
        df=df,
        transition=transition,
        model=LABEL_EXP,
        slope_col=exp_slope_col,
    )

    h1_row = {
        "transition": transition,
        "row_type": "total",
        "model": "H1",
        "positive_n": (
            int(linear_row["positive_n"])
            + int(exp_row["positive_n"])
        ),
        "negative_n": (
            int(linear_row["negative_n"])
            + int(exp_row["negative_n"])
        ),
        "valid_n": (
            int(linear_row["valid_n"])
            + int(exp_row["valid_n"])
        ),
        "selected_n": (
            int(linear_row["selected_n"])
            + int(exp_row["selected_n"])
        ),
        "excluded_n": (
            int(linear_row["excluded_n"])
            + int(exp_row["excluded_n"])
        ),
    }

    return [
        add_percentages(h1_row),
        add_percentages(linear_row),
        add_percentages(exp_row),
    ]


def build_summary_table(
    exp_root: Path,
    parameter_scale: str,
) -> tuple[pd.DataFrame, list[Path], dict[str, str]]:
    rows: list[dict[str, object]] = []
    input_files: list[Path] = []
    slope_columns: dict[str, str] = {}

    for transition in TRANSITION_ORDER:
        (
            transition_df,
            input_file,
            linear_slope_col,
            exp_slope_col,
        ) = load_slope_table(
            exp_root=exp_root,
            transition=transition,
            parameter_scale=parameter_scale,
        )

        input_files.append(input_file)

        slope_columns["linear"] = linear_slope_col
        slope_columns["exponential"] = exp_slope_col

        rows.extend(
            build_transition_rows(
                df=transition_df,
                transition=transition,
                linear_slope_col=linear_slope_col,
                exp_slope_col=exp_slope_col,
            )
        )

    return pd.DataFrame(rows), input_files, slope_columns


# ============================================================
# Plot helpers
# ============================================================

def get_row_y(
    transition: str,
    row_type: str,
    model: str,
) -> float:
    h1_y, linear_y, exp_y = Y_POSITIONS[transition]

    if row_type == "total":
        return h1_y

    if model == LABEL_LINEAR:
        return linear_y

    if model == LABEL_EXP:
        return exp_y

    raise ValueError(
        f"Unexpected row specification: "
        f"{transition}, {row_type}, {model}"
    )


def build_row_label(row: pd.Series) -> str:
    n = int(row["valid_n"])

    if row["row_type"] == "total":
        return f"{H1_LABEL} ({n})"

    return f"{MODEL_DISPLAY[str(row['model'])]} ({n})"


def label_text_math(
    percentage: float,
    count: int,
) -> str:
    return rf"$\mathbf{{{percentage:.0f}\%}}$ ({count})"


def add_bar_label(
    ax: plt.Axes,
    *,
    left: float,
    width: float,
    y: float,
    count: int,
    percentage: float,
    text_color: str,
) -> None:
    if width <= 0:
        return

    text = label_text_math(
        percentage=percentage,
        count=count,
    )

    if width >= 15:
        ax.text(
            left + width / 2,
            y,
            text,
            ha="center",
            va="center",
            fontsize=10.5,
            color=text_color,
        )
        return

    if width >= 8:
        ax.text(
            left + width / 2,
            y,
            text,
            ha="center",
            va="center",
            fontsize=8.7,
            color=text_color,
        )
        return

    if width >= 4.5:
        ax.text(
            left + width / 2,
            y,
            text,
            ha="center",
            va="center",
            fontsize=7.8,
            color=text_color,
        )
        return

    x_anchor = left + width / 2
    x_text = min(max(x_anchor, 2.5), 97.5)

    ax.annotate(
        text,
        xy=(x_anchor, y),
        xytext=(x_text, y - 0.32),
        ha="center",
        va="bottom",
        fontsize=7.6,
        color="#333333",
        arrowprops={
            "arrowstyle": "-",
            "linewidth": 0.65,
            "color": "#777777",
        },
    )


def add_full_h1_outline(
    ax: plt.Axes,
    y: float,
    bar_height: float,
) -> None:
    rect = Rectangle(
        (0, y - bar_height / 2),
        100,
        bar_height,
        fill=False,
        edgecolor=H1_OUTLINE_COLOR,
        linewidth=H1_OUTLINE_WIDTH,
        zorder=4,
    )

    ax.add_patch(rect)


# ============================================================
# Plot
# ============================================================

def plot_combined_direction_figure(
    summary_df: pd.DataFrame,
    output_file: Path,
    pdf_output_file: Path | None,
) -> list[str]:
    ensure_dir(output_file.parent)

    if pdf_output_file is not None:
        ensure_dir(pdf_output_file.parent)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.set_yticks([])

    for _, row in summary_df.iterrows():
        transition = str(row["transition"])
        row_type = str(row["row_type"])
        model = str(row["model"])

        y = get_row_y(
            transition=transition,
            row_type=row_type,
            model=model,
        )

        is_h1 = row_type == "total"

        positive_pct = float(row["positive_pct"])
        negative_pct = float(row["negative_pct"])

        bar_height = (
            H1_BAR_HEIGHT
            if is_h1
            else MODEL_BAR_HEIGHT
        )

        # Positive section
        ax.barh(
            y,
            positive_pct,
            height=bar_height,
            color=POSITIVE_COLOR,
            edgecolor="white",
            linewidth=0.8,
            zorder=2,
        )

        # Negative section
        ax.barh(
            y,
            negative_pct,
            left=positive_pct,
            height=bar_height,
            color=NEGATIVE_COLOR,
            edgecolor="white",
            linewidth=0.8,
            zorder=2,
        )

        # Strong outline around H1 only
        if is_h1:
            add_full_h1_outline(
                ax=ax,
                y=y,
                bar_height=bar_height,
            )

        # Bar labels
        add_bar_label(
            ax=ax,
            left=0.0,
            width=positive_pct,
            y=y,
            count=int(row["positive_n"]),
            percentage=positive_pct,
            text_color=BAR_TEXT_COLOR,
        )

        add_bar_label(
            ax=ax,
            left=positive_pct,
            width=negative_pct,
            y=y,
            count=int(row["negative_n"]),
            percentage=negative_pct,
            text_color=BAR_TEXT_COLOR,
        )

        # Row labels
        row_label = build_row_label(row)

        if is_h1:
            row_fontsize = H1_ROW_LABEL_FONTSIZE
            row_fontweight = "bold"
        else:
            row_fontsize = MODEL_ROW_LABEL_FONTSIZE
            row_fontweight = "normal"

        ax.text(
            ROW_LABEL_X,
            y,
            row_label,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=row_fontsize,
            fontweight=row_fontweight,
            color="#111111",
        )

    # ========================================================
    # Transition labels
    # ========================================================

    for transition in TRANSITION_ORDER:
        h1_y, _, exp_y = Y_POSITIONS[transition]

        group_top = h1_y - H1_BAR_HEIGHT / 2
        group_bottom = exp_y + MODEL_BAR_HEIGHT / 2
        center_y = (group_top + group_bottom) / 2

        ax.text(
            TRANSITION_LABEL_X,
            center_y,
            TRANSITION_DISPLAY[transition],
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=TRANSITION_LABEL_FONTSIZE,
            fontweight="bold",
            color="#111111",
        )

    # ========================================================
    # Separators between transitions
    # ========================================================

    for current_transition, next_transition in zip(
        TRANSITION_ORDER[:-1],
        TRANSITION_ORDER[1:],
    ):
        current_exp_y = Y_POSITIONS[current_transition][2]
        next_h1_y = Y_POSITIONS[next_transition][0]

        current_bottom = (
            current_exp_y
            + MODEL_BAR_HEIGHT / 2
        )

        next_top = (
            next_h1_y
            - H1_BAR_HEIGHT / 2
        )

        separator_y = (
            current_bottom + next_top
        ) / 2

        ax.axhline(
            separator_y,
            color="#D8D8D8",
            linewidth=0.9,
            zorder=1,
        )

    # ========================================================
    # Axes
    # ========================================================

    ax.set_xlim(0, 100)

    ax.set_xticks(
        [0, 25, 50, 75, 100]
    )

    ax.set_xlabel(
        "Families (%)",
        fontsize=AXIS_LABEL_FONTSIZE,
    )

    ax.invert_yaxis()

    ax.grid(
        axis="x",
        color="#EBEBEB",
        linestyle="--",
        linewidth=0.8,
        zorder=0,
    )

    ax.set_axisbelow(True)

    # ========================================================
    # Legend
    # ========================================================

    legend_handles = [
        Patch(
            facecolor=POSITIVE_COLOR,
            label="Positive",
        ),
        Patch(
            facecolor=NEGATIVE_COLOR,
            label="Negative",
        ),
    ]

    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=2,
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        handlelength=2.1,
        columnspacing=2.4,
    )

    # ========================================================
    # Title
    # ========================================================

    fig.suptitle(
        "Direction of chromosome-number dependence",
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
        y=0.975,
    )

    # ========================================================
    # Styling
    # ========================================================

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.spines["bottom"].set_color(
        "#777777"
    )

    ax.tick_params(
        axis="x",
        labelsize=TICK_LABEL_FONTSIZE,
        colors="#333333",
    )

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    fig.subplots_adjust(
        left=0.28,
        right=0.98,
        top=0.79,
        bottom=0.13,
    )

    # ========================================================
    # Save
    # ========================================================

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    output_paths = [
        output_file.as_posix()
    ]

    if pdf_output_file is not None:
        fig.savefig(
            pdf_output_file,
            bbox_inches="tight",
            facecolor="white",
        )

        output_paths.append(
            pdf_output_file.as_posix()
        )

    plt.close(fig)

    return output_paths


# ============================================================
# Console summary
# ============================================================

def print_summary(
    summary_df: pd.DataFrame,
) -> None:
    print()
    print("Combined slope-direction summary")
    print("=" * 82)

    for transition in TRANSITION_ORDER:
        transition_df = summary_df.loc[
            summary_df["transition"] == transition
        ]

        print()
        print(
            TRANSITION_DISPLAY[transition]
        )

        for _, row in transition_df.iterrows():
            label = (
                H1_LABEL
                if row["row_type"] == "total"
                else MODEL_DISPLAY[str(row["model"])]
            )

            print(
                f"  {label:12s} "
                f"Positive: {int(row['positive_n']):2d} "
                f"({float(row['positive_pct']):5.1f}%)   "
                f"Negative: {int(row['negative_n']):2d} "
                f"({float(row['negative_pct']):5.1f}%)   "
                f"n={int(row['valid_n'])}"
            )

            excluded_n = int(
                row["excluded_n"]
            )

            if excluded_n > 0:
                print(
                    f"    WARNING: {excluded_n} "
                    "selected families had missing/zero "
                    "direction and were excluded."
                )

    print()


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()

    (
        summary_df,
        input_files,
        slope_columns,
    ) = build_summary_table(
        exp_root=args.exp_root,
        parameter_scale=args.parameter_scale,
    )

    print_summary(summary_df)

    output_paths = plot_combined_direction_figure(
        summary_df=summary_df,
        output_file=args.output_file,
        pdf_output_file=args.pdf_output_file,
    )

    notes = (
        "Generated the main combined slope-direction poster figure. "
        "Within each transition, H1, Linear, and Exponential bars "
        "touch directly and therefore form one visual group. "
        "Larger whitespace separates Duplication, Loss, and Gain. "
        "H1 pools counts across overall-chosen Linear and Exponential "
        "families before computing percentages. "
        "H1 remains visually prominent through a thicker bar, "
        "bold label, and thick dark outline. "
        "Labels show bold percentages followed by counts in parentheses. "
        "Transition order top-to-bottom: Duplication, Loss, Gain. "
        f"Parameter scale: {args.parameter_scale}. "
        f"Positive color: {POSITIVE_COLOR}. "
        f"Negative color: {NEGATIVE_COLOR}. "
        f"Transition-group gap: {TRANSITION_GROUP_GAP}."
    )

    if args.log_note:
        notes += (
            f"\nRun note: {args.log_note}"
        )

    log_run(
        step="poster_combined_slope_direction",
        script=Path(__file__),
        params={
            "exp_root": args.exp_root,
            "parameter_scale": args.parameter_scale,
            "linear_slope_column": slope_columns["linear"],
            "exp_effective_slope_column": slope_columns["exponential"],
            "input_files": [
                path.as_posix()
                for path in input_files
            ],
            "output_file": args.output_file,
            "pdf_output_file": args.pdf_output_file,
        },
        outputs=output_paths,
        description=(
            "Generated the combined poster figure summarizing "
            "positive versus negative chromosome-number dependence "
            "for H1 families together with the Linear and Exponential "
            "breakdown for duplication, loss, and gain."
        ),
        notes=notes,
    )

    print(
        "[✓] Combined slope-direction poster figure completed."
    )

    print(
        f"Parameter scale: {args.parameter_scale}"
    )

    for output_path in output_paths:
        print(output_path)


if __name__ == "__main__":
    main()