"""
Plot selected fitted transition-rate functions.

Each run writes three figures:
1. full overlay-only figure: x = 0-150, y = 0-150
2. full poster figure: x = 0-150, y = 0-150
3. zoomed overlay-only figure: x = 0-25, y = 0-25

The poster figure contains one outer summary panel in the upper-right.
Inside it:
- one box containing H0, H1, Linear, and Exponential summaries
- a large Constant / Linear / Exponential pie chart

The chosen-model table is the source of truth for model selection.
All model colors come directly from MODEL_COLOR_MAP.
"""

import argparse
from pathlib import Path
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from source_code.analysis.analysis_constants import CHOSEN_FUNCTION_LABEL_COL, MODEL_COLOR_MAP
from source_code.analysis.fitted_functions.fitted_functions_utils import (
    FITTED_PARAMETERS_JSON_COL,
    PARAMETER_SCALE_COL,
    evaluate_fitted_function,
    parse_parameters,
    require_columns,
)
from source_code.analysis.plot_utils import build_chromosome_grid, ensure_dir, save_close, to_float
from source_code.constants import (
    FAMILY_NAME_COL,
    LABEL_CONSTANT,
    LABEL_DUPL,
    LABEL_EXP,
    LABEL_FUNC_TYPE_COL,
    LABEL_GAIN,
    LABEL_LINEAR,
    LABEL_LOSS,
    LABEL_TESTED_TRANSITION_COL,
    MAX_CHROM_COL,
    MIN_CHROM_COL,
)
from source_code.logger import log_run


SUPPORTED_SELECTED_MODELS: tuple[str, ...] = (LABEL_LINEAR, LABEL_EXP)
POSTER_MODEL_ORDER: tuple[str, ...] = (LABEL_CONSTANT, LABEL_LINEAR, LABEL_EXP)

TRANSITION_DISPLAY_NAMES: dict[str, str] = {
    LABEL_GAIN: "Chromosome Gain",
    LABEL_LOSS: "Chromosome Loss",
    LABEL_DUPL: "Whole-Genome Duplication",
}

MODEL_DISPLAY_NAMES: dict[str, str] = {
    LABEL_CONSTANT: "Constant",
    LABEL_LINEAR: "Linear",
    LABEL_EXP: "Exponential",
}

FIGSIZE: tuple[float, float] = (11.0, 7.0)
LINE_ALPHA: float = 0.55
LINE_WIDTH: float = 1.4

FULL_X_MAX: float = 150.0
FULL_Y_MAX: float = 80.0
ZOOM_X_MAX: float = 25.0
ZOOM_Y_MAX: float = 25.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot selected fitted Linear/Exponential transition-rate functions.")
    parser.add_argument("--fitted-functions-file", type=Path, required=True)
    parser.add_argument("--chosen-model-file", type=Path, required=True)
    parser.add_argument("--selected-models", nargs="+", choices=SUPPORTED_SELECTED_MODELS, required=True)
    parser.add_argument("--output-file", type=Path, required=True, help="Full-range overlay-only PNG path.")
    parser.add_argument("--poster-output-file", type=Path, default=None, help="Full-range poster PNG path.")
    parser.add_argument("--zoom-output-file", type=Path, default=None, help="Zoomed overlay-only PNG path.")
    return parser.parse_args()


def load_chosen_models(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    require_columns(df, [FAMILY_NAME_COL, CHOSEN_FUNCTION_LABEL_COL], "chosen-model table")
    df = df[[FAMILY_NAME_COL, CHOSEN_FUNCTION_LABEL_COL]].copy()

    if df[FAMILY_NAME_COL].duplicated().any():
        duplicates = sorted(df.loc[df[FAMILY_NAME_COL].duplicated(keep=False), FAMILY_NAME_COL].astype(str).unique().tolist())
        raise ValueError(f"Duplicate families in chosen-model table: {duplicates}")

    return df


def load_fitted_functions(file_path: Path) -> pd.DataFrame:
    required_columns = [
        FAMILY_NAME_COL,
        LABEL_TESTED_TRANSITION_COL,
        LABEL_FUNC_TYPE_COL,
        MIN_CHROM_COL,
        MAX_CHROM_COL,
        FITTED_PARAMETERS_JSON_COL,
        PARAMETER_SCALE_COL,
    ]

    df = pd.read_csv(file_path)
    require_columns(df, required_columns, "fitted-functions table")

    if df.duplicated([FAMILY_NAME_COL, LABEL_TESTED_TRANSITION_COL, LABEL_FUNC_TYPE_COL]).any():
        raise ValueError("Duplicate family × transition × function rows in fitted-functions table.")

    return df[required_columns].copy()


def prepare_selected_functions(
    fitted_df: pd.DataFrame,
    chosen_df: pd.DataFrame,
    selected_models: Sequence[str],
) -> pd.DataFrame:
    chosen_selected = chosen_df[chosen_df[CHOSEN_FUNCTION_LABEL_COL].isin(set(selected_models))].copy()

    if chosen_selected.empty:
        raise ValueError(f"No families selected one of the requested models: {sorted(set(selected_models))}")

    merged = chosen_selected.merge(fitted_df, on=FAMILY_NAME_COL, how="left", validate="one_to_many")
    selected = merged[merged[LABEL_FUNC_TYPE_COL] == merged[CHOSEN_FUNCTION_LABEL_COL]].copy()

    missing = sorted(set(chosen_selected[FAMILY_NAME_COL].astype(str)) - set(selected[FAMILY_NAME_COL].astype(str)))
    if missing:
        raise ValueError(f"Selected families missing their selected fitted function: {missing}")
    if selected[FAMILY_NAME_COL].duplicated().any():
        raise ValueError("More than one selected fitted-function row found for at least one family.")

    for column in (MIN_CHROM_COL, MAX_CHROM_COL):
        selected[column] = pd.to_numeric(selected[column], errors="coerce")

    invalid_range = selected[MIN_CHROM_COL].isna() | selected[MAX_CHROM_COL].isna() | (selected[MAX_CHROM_COL] < selected[MIN_CHROM_COL])
    if invalid_range.any():
        families = sorted(selected.loc[invalid_range, FAMILY_NAME_COL].astype(str).tolist())
        raise ValueError(f"Invalid chromosome ranges for families: {families}")

    return selected.sort_values(FAMILY_NAME_COL).reset_index(drop=True)


def get_single_value(df: pd.DataFrame, column: str, context: str) -> str:
    values = df[column].dropna().astype(str).unique().tolist()
    if len(values) != 1:
        raise ValueError(f"{context}: expected one '{column}' value, found {values}")
    return values[0]


def build_plot_title(transition_label: str) -> str:
    transition_name = TRANSITION_DISPLAY_NAMES.get(transition_label, transition_label.replace("_", " ").title())
    return f"Selected {transition_name} Rate Functions"


def evaluate_selected_curve(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    family_name = str(row[FAMILY_NAME_COL])
    model_label = str(row[CHOSEN_FUNCTION_LABEL_COL])
    min_chrom, max_chrom = to_float(row[MIN_CHROM_COL]), to_float(row[MAX_CHROM_COL])

    chromosome_grid = build_chromosome_grid(min_chrom, max_chrom)
    if chromosome_grid is None:
        raise ValueError(f"Could not construct chromosome grid for {family_name}: min={min_chrom}, max={max_chrom}")

    rates = np.asarray(
        evaluate_fitted_function(model_label, chromosome_grid, parse_parameters(row[FITTED_PARAMETERS_JSON_COL])),
        dtype=float,
    )

    if not np.isfinite(rates).all():
        raise ValueError(f"Non-finite fitted rates for {family_name}: {model_label}")

    return chromosome_grid, rates


def add_selected_function_curves(ax: plt.Axes, selected_df: pd.DataFrame) -> None:
    for _, row in selected_df.iterrows():
        chromosome_grid, rates = evaluate_selected_curve(row)
        model_label = str(row[CHOSEN_FUNCTION_LABEL_COL])
        ax.plot(chromosome_grid, rates, color=MODEL_COLOR_MAP[model_label], alpha=LINE_ALPHA, linewidth=LINE_WIDTH)


def build_overlay_legend_handles(selected_df: pd.DataFrame, selected_models: Sequence[str]) -> list[Line2D]:
    model_order = [model for model in SUPPORTED_SELECTED_MODELS if model in set(selected_models)]
    counts = selected_df[CHOSEN_FUNCTION_LABEL_COL].value_counts().reindex(model_order, fill_value=0)
    total = len(selected_df)

    return [
        Line2D(
            [0],
            [0],
            color=MODEL_COLOR_MAP[model],
            linewidth=2.4,
            label=f"{MODEL_DISPLAY_NAMES[model]} (n={int(counts[model])}, {100 * int(counts[model]) / total:.1f}%)",
        )
        for model in model_order
    ]


def style_axes(
    ax: plt.Axes,
    transition_label: str,
    x_max: float,
    y_max: float,
    selected_df: pd.DataFrame,
    selected_models: Sequence[str],
    show_legend: bool,
) -> None:
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_title(build_plot_title(transition_label))
    ax.set_xlabel("Chromosome number")
    ax.set_ylabel("Transition rate")

    if show_legend:
        ax.legend(
            handles=build_overlay_legend_handles(selected_df, selected_models),
            title=f"Selected models (n={len(selected_df)})",
            loc="upper right",
            frameon=True,
        )


def build_poster_model_counts(chosen_df: pd.DataFrame) -> pd.Series:
    labels = set(chosen_df[CHOSEN_FUNCTION_LABEL_COL].dropna().astype(str))
    unexpected = sorted(labels - set(POSTER_MODEL_ORDER))

    if unexpected:
        raise ValueError(
            "Poster figure requires a Constant/Linear/Exponential chosen-model table; "
            f"unexpected categories found: {unexpected}. Use the output without Ignore."
        )

    return chosen_df[CHOSEN_FUNCTION_LABEL_COL].value_counts().reindex(POSTER_MODEL_ORDER, fill_value=0).astype(int)


def build_pie_autopct(model_counts: pd.Series) -> Callable[[float], str]:
    total = int(model_counts.sum())

    def format_slice(pct: float) -> str:
        count = int(round(pct * total / 100))
        return f"{pct:.1f}%\n(n={count})" if count else ""

    return format_slice


def add_poster_summary(ax: plt.Axes, selected_df: pd.DataFrame, model_counts: pd.Series) -> None:
    total_n = int(model_counts.sum())
    h0_n = int(model_counts[LABEL_CONSTANT])
    h1_n = int(model_counts[LABEL_LINEAR] + model_counts[LABEL_EXP])

    h0_pct = 100 * h0_n / total_n if total_n else 0.0
    h1_pct = 100 * h1_n / total_n if total_n else 0.0

    h1_counts = (
        selected_df[CHOSEN_FUNCTION_LABEL_COL]
        .value_counts()
        .reindex(SUPPORTED_SELECTED_MODELS, fill_value=0)
    )

    # Outer frame containing both summary box and pie.
    outer_ax = ax.inset_axes([0.63, 0.30, 0.36, 0.67])
    outer_ax.set_facecolor("white")
    outer_ax.set_xlim(0, 1)
    outer_ax.set_ylim(0, 1)
    outer_ax.set_xticks([])
    outer_ax.set_yticks([])

    for spine in outer_ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.9)
        spine.set_edgecolor("black")

    # Single inner box containing H0, H1, Linear, and Exponential.
    summary_ax = outer_ax.inset_axes([0.06, 0.69, 0.88, 0.27])
    summary_ax.set_facecolor("white")
    summary_ax.set_xlim(0, 1)
    summary_ax.set_ylim(0, 1)
    summary_ax.set_xticks([])
    summary_ax.set_yticks([])

    for spine in summary_ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_edgecolor("black")

    summary_ax.text(0.06, 0.80, rf"$H_0$: {h0_n}/{total_n} ({h0_pct:.1f}%)", ha="left", va="center", fontsize=10.2)
    summary_ax.text(0.06, 0.58, rf"$H_1$: {h1_n}/{total_n} ({h1_pct:.1f}%)", ha="left", va="center", fontsize=10.2)

    for model, y_pos in ((LABEL_LINEAR, 0.34), (LABEL_EXP, 0.12)):
        count = int(h1_counts[model])
        percentage = 100 * count / h1_n if h1_n else 0.0

        summary_ax.plot(
            [0.07, 0.18],
            [y_pos, y_pos],
            color=MODEL_COLOR_MAP[model],
            linewidth=3.2,
            solid_capstyle="butt",
        )
        summary_ax.text(
            0.22,
            y_pos,
            f"{MODEL_DISPLAY_NAMES[model]} (n={count}/{h1_n}, {percentage:.1f}%)",
            ha="left",
            va="center",
            fontsize=9.1,
        )

    # Large pie with explicit margins so outside labels stay inside the outer frame.
    pie_ax = outer_ax.inset_axes([0.03, 0.02, 0.94, 0.66])

    pie_ax.pie(
        model_counts.to_numpy(dtype=int),
        labels=[MODEL_DISPLAY_NAMES[model] for model in POSTER_MODEL_ORDER],
        colors=[MODEL_COLOR_MAP[model] for model in POSTER_MODEL_ORDER],
        autopct=build_pie_autopct(model_counts),
        startangle=90,
        counterclock=False,
        radius=0.88,
        pctdistance=0.60,
        labeldistance=1.05,
        textprops={"fontsize": 9.2},
        wedgeprops={"edgecolor": "white", "linewidth": 1.0},
    )

    # Reserve enough room for Constant / Linear / Exponential labels.
    pie_ax.set_xlim(-1.35, 1.35)
    pie_ax.set_ylim(-1.18, 1.18)


def resolve_output_file(base_file: Path, explicit_file: Path | None, suffix: str) -> Path:
    if explicit_file is not None:
        return explicit_file
    extension = base_file.suffix or ".png"
    return base_file.with_name(f"{base_file.stem}_{suffix}{extension}")


def plot_overlay(
    selected_df: pd.DataFrame,
    selected_models: Sequence[str],
    output_file: Path,
    x_max: float,
    y_max: float,
) -> None:
    transition_label = get_single_value(selected_df, LABEL_TESTED_TRANSITION_COL, "Selected-function overlay")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    add_selected_function_curves(ax, selected_df)
    style_axes(ax, transition_label, x_max, y_max, selected_df, selected_models, show_legend=True)

    ensure_dir(output_file.parent)
    save_close(fig, output_file)


def plot_poster(
    selected_df: pd.DataFrame,
    chosen_df: pd.DataFrame,
    selected_models: Sequence[str],
    output_file: Path,
) -> None:
    transition_label = get_single_value(selected_df, LABEL_TESTED_TRANSITION_COL, "Selected-function poster")
    model_counts = build_poster_model_counts(chosen_df)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    add_selected_function_curves(ax, selected_df)
    style_axes(ax, transition_label, FULL_X_MAX, FULL_Y_MAX, selected_df, selected_models, show_legend=False)
    add_poster_summary(ax, selected_df, model_counts)

    ensure_dir(output_file.parent)
    save_close(fig, output_file)


def run_selected_functions_overlay(
    fitted_functions_file: Path,
    chosen_model_file: Path,
    selected_models: Sequence[str],
    output_file: Path,
    poster_output_file: Path | None = None,
    zoom_output_file: Path | None = None,
) -> tuple[pd.DataFrame, list[Path]]:
    fitted_df = load_fitted_functions(fitted_functions_file)
    chosen_df = load_chosen_models(chosen_model_file)
    selected_df = prepare_selected_functions(fitted_df, chosen_df, selected_models)

    poster_file = resolve_output_file(output_file, poster_output_file, "poster_combined")
    zoom_file = resolve_output_file(output_file, zoom_output_file, "zoom_x25")

    plot_overlay(selected_df, selected_models, output_file, FULL_X_MAX, FULL_Y_MAX)
    plot_poster(selected_df, chosen_df, selected_models, poster_file)

    # Zoom is explicitly restricted to x = 0-25 AND y = 0-25.
    plot_overlay(selected_df, selected_models, zoom_file, ZOOM_X_MAX, ZOOM_Y_MAX)

    return selected_df, [output_file, poster_file, zoom_file]


def main() -> None:
    args = parse_args()
    selected_models = tuple(dict.fromkeys(args.selected_models))

    selected_df, output_files = run_selected_functions_overlay(
        fitted_functions_file=args.fitted_functions_file,
        chosen_model_file=args.chosen_model_file,
        selected_models=selected_models,
        output_file=args.output_file,
        poster_output_file=args.poster_output_file,
        zoom_output_file=args.zoom_output_file,
    )

    transition_label = get_single_value(selected_df, LABEL_TESTED_TRANSITION_COL, "Selected-function overlay")
    parameter_scale = get_single_value(selected_df, PARAMETER_SCALE_COL, "Selected-function overlay")
    counts = {
        model: int((selected_df[CHOSEN_FUNCTION_LABEL_COL] == model).sum())
        for model in selected_models
    }

    print("[✓] Selected fitted-function plots completed.")
    print(f"Transition: {transition_label}")
    print(f"Selected models: {list(selected_models)}")
    print(f"Families plotted: {len(selected_df)}")

    for model in selected_models:
        print(f"  {MODEL_DISPLAY_NAMES[model]}: {counts[model]}")

    print(f"Parameter scale: {parameter_scale}")
    print(f"Full axes: x=0-{FULL_X_MAX:g}, y=0-{FULL_Y_MAX:g}")
    print(f"Zoom axes: x=0-{ZOOM_X_MAX:g}, y=0-{ZOOM_Y_MAX:g}")

    for output_file in output_files:
        print(output_file)

    # log_run(
    #     step="selected_fitted_functions_overlay",
    #     script=Path(__file__),
    #     params={
    #         "fitted_functions_file": str(args.fitted_functions_file),
    #         "chosen_model_file": str(args.chosen_model_file),
    #         "selected_models": list(selected_models),
    #         "transition_label": transition_label,
    #         "parameter_scale": parameter_scale,
    #         "number_of_families": len(selected_df),
    #         "model_counts": counts,
    #         "full_x_max": FULL_X_MAX,
    #         "full_y_max": FULL_Y_MAX,
    #         "zoom_x_max": ZOOM_X_MAX,
    #         "zoom_y_max": ZOOM_Y_MAX,
    #     },
    #     outputs=[file.as_posix() for file in output_files],
    #     description=(
    #         "Plotted a full selected-function overlay, a full poster figure "
    #         "with boxed model-selection context, and a zoomed overlay."
    #     ),
    #     notes=(
    #         "All figures use MODEL_COLOR_MAP. Full figures use x=0-150 and y=0-150. "
    #         "The zoom uses x=0-25 and y=0-25. The poster contains one outer box around "
    #         "a single H0/H1/Linear/Exponential summary box and an enlarged model-selection "
    #         "pie. The pie axis includes explicit margins so its labels remain within "
    #         "the outer summary frame."
    #     ),
    #     log_relative_path=Path("fitted_functions") / f"{args.output_file.stem}.log",
    # )


if __name__ == "__main__":
    main()