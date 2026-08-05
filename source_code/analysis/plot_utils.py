from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import re

from source_code.analysis.analysis_constants import *
from source_code.constants import *


DEFAULT_FIGSIZE: tuple[int, int] = (9, 6)
DEFAULT_DPI: int = 300

RateFunction: TypeAlias = Callable[[float | np.ndarray, float, float], float | np.ndarray]


# =========================================================
# File and text helpers
# =========================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_close(fig: Figure, out_file: Path, dpi: int = DEFAULT_DPI) -> None:
    fig.tight_layout()
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_text(text: str, out_file: Path) -> None:
    out_file.write_text(text, encoding="utf-8")


# =========================================================
# Numeric helpers
# =========================================================

def to_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def to_float(value: object) -> float:
    numeric_value: object = pd.to_numeric(value, errors="coerce")
    return float("nan") if pd.isna(numeric_value) else float(numeric_value)


def build_stats(series: pd.Series) -> dict[str, float | int]:
    values: pd.Series = to_numeric_series(series).dropna()

    if values.empty:
        return {"n": float("nan"), "mean": float("nan"), "median": float("nan"), "std": float("nan"), "min": float("nan"), "q25": float("nan"), "q75": float("nan"), "max": float("nan")}

    return {
        "n": len(values),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "std": float(values.std()),
        "min": float(values.min()),
        "q25": float(values.quantile(0.25)),
        "q75": float(values.quantile(0.75)),
        "max": float(values.max()),
    }


def get_ordered_counts(series: pd.Series, order: Iterable[str] | None = None) -> pd.Series:
    counts: pd.Series = series.value_counts(dropna=False)
    return counts if order is None else pd.Series({label: int(counts.get(label, 0)) for label in order})


def build_common_bin_edges(series_map: dict[str, pd.Series], bins: int) -> np.ndarray | None:
    clean_values: list[pd.Series] = [to_numeric_series(series).dropna() for series in series_map.values()]
    clean_values = [values for values in clean_values if not values.empty]

    if not clean_values:
        return None

    pooled_values: pd.Series = pd.concat(clean_values, ignore_index=True)
    _, bin_edges = pd.cut(pooled_values, bins=bins, retbins=True, duplicates="drop")
    return bin_edges


def resolve_hist_bins(series: pd.Series, bin_count: int | None = None, bin_size: float | None = None) -> int | np.ndarray | None:
    values: pd.Series = to_numeric_series(series).dropna()

    if values.empty:
        return None

    if bin_count is not None and bin_size is not None:
        raise ValueError("Use either bin_count or bin_size, not both.")

    if bin_size is not None:
        if bin_size <= 0:
            raise ValueError("bin_size must be positive.")

        start: float = np.floor(float(values.min()) / bin_size) * bin_size
        end: float = np.ceil(float(values.max()) / bin_size) * bin_size + bin_size
        return np.arange(start, end, bin_size)

    return 30 if bin_count is None else bin_count


# =========================================================
# Annotation helpers
# =========================================================

def add_class_summary_box(ax: Axes, rows: list[tuple[int, str, int]], class_color_map: dict[int, str], loc: str = "upper right") -> None:
    if not rows:
        return

    if loc == "upper left":
        box_x, box_y, horizontal_alignment = 0.03, 0.97, "left"
    else:
        box_x, box_y, horizontal_alignment = 0.97, 0.97, "right"

    line_height: float = 0.055
    box_width: float = 0.26
    box_height: float = line_height * len(rows) + 0.02
    rectangle_x: float = box_x - box_width if horizontal_alignment == "right" else box_x
    rectangle_y: float = box_y - box_height

    background = plt.Rectangle(
        (rectangle_x, rectangle_y),
        box_width,
        box_height,
        transform=ax.transAxes,
        facecolor="white",
        edgecolor="black",
        alpha=0.9,
        zorder=5,
    )
    ax.add_patch(background)

    swatch_x: float = rectangle_x + 0.015
    text_x: float = rectangle_x + 0.055
    start_y: float = box_y - 0.035

    for row_index, (class_value, label_text, count) in enumerate(rows):
        y_position: float = start_y - row_index * line_height
        color: str = class_color_map.get(class_value, "#BDBDBD")

        swatch = plt.Rectangle((swatch_x, y_position - 0.018), 0.025, 0.025, transform=ax.transAxes, facecolor=color, edgecolor="black", zorder=6)
        ax.add_patch(swatch)
        ax.text(text_x, y_position, f"({label_text})  {count}", transform=ax.transAxes, ha="left", va="center", fontsize=8.5, zorder=7)


# =========================================================
# General plotters
# =========================================================

def plot_pie_counts(counts: pd.Series, title: str, out_file: Path, color_map: dict[str, str] | None = None, show_counts: bool = False, total_n: int | None = None) -> None:
    counts = counts[counts > 0]

    if counts.empty:
        return

    total_value: float = float(counts.sum())
    colors: list[str | None] | None = [color_map.get(label) for label in counts.index] if color_map is not None else None

    def format_percentage(percentage: float) -> str:
        count: int = int(round(percentage * total_value / 100.0))
        return f"{percentage:.1f}%\n(n={count})" if show_counts else f"{percentage:.1f}%"

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    ax.pie(counts.values, labels=counts.index, autopct=format_percentage, startangle=90, colors=colors)
    ax.set_title(title)
    ax.axis("equal")

    if total_n is not None:
        ax.text(0.98, 0.98, f"n={total_n}", transform=ax.transAxes, ha="right", va="top", fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

    save_close(fig, out_file)


def plot_bar_counts(counts: pd.Series, title: str, xlabel: str, ylabel: str, out_file: Path, rotation: int = 0, color_map: dict[str, str] | None = None) -> None:
    labels = counts.index.astype(str)
    colors: list[str | None] | None = None if color_map is None else [color_map.get(label) for label in labels]

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    ax.bar(labels, counts.values, color=colors)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    for index, value in enumerate(counts.values):
        ax.text(index, value, str(value), ha="center", va="bottom")

    ax.tick_params(axis="x", labelrotation=rotation)
    save_close(fig, out_file)


def plot_hist_overlay(series_map: dict[str, pd.Series], title: str, xlabel: str, ylabel: str, out_file: Path, bins: int = 30, density: bool = False, color_map: dict[str, str] | None = None) -> None:
    clean_series_map: dict[str, pd.Series] = {label: to_numeric_series(series).dropna() for label, series in series_map.items()}
    clean_series_map = {label: values for label, values in clean_series_map.items() if not values.empty}

    if not clean_series_map:
        return

    bin_edges: np.ndarray | None = build_common_bin_edges(clean_series_map, bins)

    if bin_edges is None:
        return

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    for label, values in clean_series_map.items():
        ax.hist(values, bins=bin_edges, alpha=0.5, edgecolor="black", label=label, density=density, color=None if color_map is None else color_map.get(label))

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    save_close(fig, out_file)


def plot_single_hist(series: pd.Series, title: str, xlabel: str, ylabel: str, out_file: Path, bin_count: int | None = 30, bin_size: float | None = None, color: str | None = None, vline_x: float | None = None) -> None:
    values: pd.Series = to_numeric_series(series).dropna()

    if values.empty:
        return

    bins: int | np.ndarray | None = resolve_hist_bins(values, bin_count=bin_count, bin_size=bin_size)

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    ax.hist(values, bins=bins, edgecolor="black", color=color)

    if vline_x is not None:
        ax.axvline(vline_x, linestyle="--", linewidth=1.5, color="black")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    save_close(fig, out_file)


def plot_scatter_by_group(df: pd.DataFrame, x_col: str, y_col: str, group_col: str, title: str, xlabel: str, ylabel: str, out_file: Path, color_map: dict[str, str] | None = None) -> None:
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    plotted_any: bool = False

    for group_name, group_df in df.groupby(group_col):
        x_values: pd.Series = pd.to_numeric(group_df[x_col], errors="coerce")
        y_values: pd.Series = pd.to_numeric(group_df[y_col], errors="coerce")
        valid_mask: pd.Series = x_values.notna() & y_values.notna()

        if not valid_mask.any():
            continue

        ax.scatter(x_values[valid_mask], y_values[valid_mask], alpha=0.75, label=str(group_name), color=None if color_map is None else color_map.get(str(group_name)))
        plotted_any = True

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if plotted_any:
        ax.legend()

    save_close(fig, out_file)


def plot_box_by_group(df: pd.DataFrame, value_col: str, group_col: str, title: str, xlabel: str, ylabel: str, out_file: Path, group_order: list[object] | None = None) -> None:
    plot_df: pd.DataFrame = df[[value_col, group_col]].copy()
    plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[value_col, group_col])

    if plot_df.empty:
        return

    if group_order is None:
        group_order = list(plot_df[group_col].unique())

    data: list[pd.Series] = []
    labels: list[str] = []

    for group in group_order:
        values: pd.Series = plot_df.loc[plot_df[group_col] == group, value_col].dropna()

        if values.empty:
            continue

        data.append(values)
        labels.append(str(group))

    if not data:
        return

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    ax.boxplot(data, tick_labels=labels)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    save_close(fig, out_file)


def plot_stacked_bar_counts(counts_df: pd.DataFrame, title: str, xlabel: str, ylabel: str, out_file: Path, color_map: dict[str, str] | None = None, show_segment_labels: bool = True) -> None:
    if counts_df.empty:
        return

    categories: list[str] = counts_df.index.astype(str).tolist()
    model_labels: list[str] = counts_df.columns.astype(str).tolist()
    bottoms: list[float] = [0.0] * len(categories)

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    for model_label in model_labels:
        values: list[float] = counts_df[model_label].tolist()
        bars = ax.bar(categories, values, bottom=bottoms, label=model_label, color=None if color_map is None else color_map.get(model_label))

        if show_segment_labels:
            for index, (bar, value) in enumerate(zip(bars, values)):
                total: float = float(counts_df.iloc[index].sum())

                if value <= 0 or total <= 0:
                    continue

                percentage: float = 100.0 * value / total
                y_position: float = bottoms[index] + value / 2.0
                ax.text(bar.get_x() + bar.get_width() / 2.0, y_position, f"{value}\n{percentage:.1f}%", ha="center", va="center", fontsize=9)

        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

    for index, total in enumerate(counts_df.sum(axis=1).tolist()):
        ax.text(index, total, f"n={int(total)}", ha="center", va="bottom")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    save_close(fig, out_file)


# =========================================================
# Class-colored histogram
# =========================================================

def plot_hist_by_class(df: pd.DataFrame, value_col: str, class_col: str, title: str, xlabel: str, ylabel: str, out_file: Path, class_order: list[int], class_color_map: dict[int, str], bin_size: float, summary_rows: list[tuple[int, str, int]] | None = None, summary_loc: str = "upper right") -> None:
    plot_df: pd.DataFrame = df[[value_col, class_col]].copy()
    plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors="coerce")
    plot_df[class_col] = pd.to_numeric(plot_df[class_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[value_col, class_col])

    if plot_df.empty:
        return

    minimum_value: float = float(plot_df[value_col].min())
    maximum_value: float = float(plot_df[value_col].max())
    start: float = np.floor(minimum_value / bin_size) * bin_size
    end: float = np.ceil(maximum_value / bin_size) * bin_size + bin_size
    bins: np.ndarray = np.arange(start, end + bin_size, bin_size)

    data: list[pd.Series] = []
    colors: list[str] = []

    for class_value in class_order:
        values: pd.Series = plot_df.loc[plot_df[class_col] == class_value, value_col].dropna()

        if values.empty:
            continue

        data.append(values)
        colors.append(class_color_map.get(class_value, "#BDBDBD"))

    if not data:
        return

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    ax.hist(data, bins=bins, stacked=True, color=colors, edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if summary_rows:
        add_class_summary_box(ax, summary_rows, class_color_map, summary_loc)

    save_close(fig, out_file)


# =========================================================
# Function-rate helpers
# =========================================================

def infer_slope_sign(slope: float) -> str | None:
    if pd.isna(slope) or slope == 0:
        return None

    return POSITIVE_SLOPE_LABEL if slope > 0 else NEGATIVE_SLOPE_LABEL


def compute_constant_rate(chrom_num: float | np.ndarray, value: float) -> float | np.ndarray:
    if np.isscalar(chrom_num):
        return float(value)

    return np.full_like(chrom_num, value, dtype=float)


def compute_linear_rate(chrom_num: float | np.ndarray, p1: float, p2: float) -> float | np.ndarray:
    return p1 + p2 * (chrom_num - 1)


def compute_exp_rate(chrom_num: float | np.ndarray, p1: float, p2: float) -> float | np.ndarray:
    return p1 * np.exp(p2 * (chrom_num - 1))


def build_chromosome_grid(min_chr: float, max_chr: float, padding: float = 0.0, n_points: int = 200) -> np.ndarray | None:
    if pd.isna(min_chr) or pd.isna(max_chr) or max_chr < min_chr:
        return None

    return np.linspace(max(1.0, min_chr - padding), max_chr + padding, n_points)


def compute_effective_slope(rate_at_min: float, rate_at_max: float, min_chr: float, max_chr: float) -> float:
    if pd.isna(rate_at_min) or pd.isna(rate_at_max) or pd.isna(min_chr) or pd.isna(max_chr) or max_chr == min_chr:
        return np.nan

    return (rate_at_max - rate_at_min) / (max_chr - min_chr)


# =========================================================
# Function-curve overlay
# =========================================================

def plot_direction_colored_function_overlay(df: pd.DataFrame, family_col: str, p1_col: str, p2_col: str, min_chrom_col: str, max_chrom_col: str, rate_function: RateFunction, direction_col: str, title: str, ylabel: str, out_file: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    sign_counts: dict[str, int] = {POSITIVE_SLOPE_LABEL: 0, NEGATIVE_SLOPE_LABEL: 0}

    for _, row in df.sort_values(family_col).iterrows():
        p1: float = to_float(row[p1_col])
        p2: float = to_float(row[p2_col])
        direction_value: float = to_float(row[direction_col])
        min_chr: float = to_float(row[min_chrom_col])
        max_chr: float = to_float(row[max_chrom_col])
        chromosome_grid: np.ndarray | None = build_chromosome_grid(min_chr, max_chr)
        slope_sign: str | None = infer_slope_sign(direction_value)

        if chromosome_grid is None or pd.isna(p1) or pd.isna(p2) or slope_sign is None:
            continue

        rates: np.ndarray = np.asarray(rate_function(chromosome_grid, p1, p2), dtype=float)

        if not np.isfinite(rates).all():
            continue

        sign_counts[slope_sign] += 1
        ax.plot(chromosome_grid, rates, color=SLOPE_SIGN_COLOR_MAP[slope_sign], alpha=0.65, linewidth=1.4)

    total_plotted: int = sum(sign_counts.values())

    if total_plotted == 0:
        plt.close(fig)
        return

    legend_handles: list[Line2D] = [
        Line2D([0], [0], color=SLOPE_SIGN_COLOR_MAP[POSITIVE_SLOPE_LABEL], linewidth=2, label=f"Positive (n={sign_counts[POSITIVE_SLOPE_LABEL]})"),
        Line2D([0], [0], color=SLOPE_SIGN_COLOR_MAP[NEGATIVE_SLOPE_LABEL], linewidth=2, label=f"Negative (n={sign_counts[NEGATIVE_SLOPE_LABEL]})"),
    ]

    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_title(f"{title} (n={total_plotted})")
    ax.set_xlabel("Chromosome number")
    ax.set_ylabel(ylabel)
    ax.legend(handles=legend_handles)

    save_close(fig, out_file)

def safe_name_for_file(value: object) -> str:
    clean_name: str = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return clean_name if clean_name else "unknown"