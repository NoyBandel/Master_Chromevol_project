from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_FIGSIZE = (9, 6)
DEFAULT_DPI = 300


# -------- file / text helpers --------
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_close(fig: plt.Figure, out_file: Path, dpi: int = DEFAULT_DPI) -> None:
    fig.tight_layout()
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_text(text: str, out_file: Path) -> None:
    out_file.write_text(text, encoding="utf-8")


# -------- numeric helpers --------
def to_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def build_stats(series: pd.Series) -> Dict[str, float]:
    s: pd.Series = to_numeric_series(series).dropna()

    if s.empty:
        return {
            "n": float("nan"),
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "q25": float("nan"),
            "q75": float("nan"),
            "max": float("nan"),
        }

    return {
        "n": len(s),
        "mean": s.mean(),
        "median": s.median(),
        "std": s.std(),
        "min": s.min(),
        "q25": s.quantile(0.25),
        "q75": s.quantile(0.75),
        "max": s.max(),
    }


def get_ordered_counts(series: pd.Series, order: Optional[Iterable[str]] = None) -> pd.Series:
    counts: pd.Series = series.value_counts(dropna=False)

    if order is None:
        return counts

    return pd.Series({label: int(counts.get(label, 0)) for label in order})


def build_common_bin_edges(series_map: Dict[str, pd.Series], bins: int) -> Optional[np.ndarray]:
    clean_vals: list[pd.Series] = []

    for series in series_map.values():
        vals: pd.Series = to_numeric_series(series).dropna()
        if not vals.empty:
            clean_vals.append(vals)

    if not clean_vals:
        return None

    pooled: pd.Series = pd.concat(clean_vals, ignore_index=True)
    _, bin_edges = pd.cut(pooled, bins=bins, retbins=True, duplicates="drop")

    return bin_edges


def resolve_hist_bins(series: pd.Series, bin_count: Optional[int] = None, bin_size: Optional[float] = None):
    vals: pd.Series = to_numeric_series(series).dropna()

    if vals.empty:
        return None

    if bin_count is not None and bin_size is not None:
        raise ValueError("Use either bin_count or bin_size, not both.")

    if bin_size is not None:
        if bin_size <= 0:
            raise ValueError("bin_size must be positive.")

        min_val: float = vals.min()
        max_val: float = vals.max()
        start: float = np.floor(min_val / bin_size) * bin_size
        end: float = np.ceil(max_val / bin_size) * bin_size + bin_size

        return np.arange(start, end, bin_size)

    if bin_count is None:
        bin_count = 30

    return bin_count


# -------- annotation helpers --------
def add_class_summary_box(
    ax,
    rows: list[tuple[int, str, int]],
    class_color_map: Dict[int, str],
    loc: str = "upper right",
) -> None:
    if not rows:
        return

    # positioning
    if loc == "upper right":
        box_x, box_y = 0.97, 0.97   # slightly inward
        ha = "right"
    elif loc == "upper left":
        box_x, box_y = 0.03, 0.97
        ha = "left"
    else:
        box_x, box_y = 0.97, 0.97
        ha = "right"

    # smaller layout
    line_h = 0.055          # was ~0.075 → tighter
    box_w = 0.26            # was ~0.34 → narrower
    box_h = line_h * len(rows) + 0.02

    rect_x = box_x - box_w if ha == "right" else box_x
    rect_y = box_y - box_h

    # background
    bg = plt.Rectangle(
        (rect_x, rect_y),
        box_w,
        box_h,
        transform=ax.transAxes,
        facecolor="white",
        edgecolor="black",
        alpha=0.9,
        zorder=5,
    )
    ax.add_patch(bg)

    # tighter spacing
    swatch_x = rect_x + 0.015
    text_x = rect_x + 0.055
    start_y = box_y - 0.035

    for i, (class_value, label_text, count) in enumerate(rows):
        y = start_y - i * line_h
        color = class_color_map.get(class_value, "#BDBDBD")

        # smaller color square
        swatch = plt.Rectangle(
            (swatch_x, y - 0.018),
            0.025,
            0.025,
            transform=ax.transAxes,
            facecolor=color,
            edgecolor="black",
            zorder=6,
        )
        ax.add_patch(swatch)

        ax.text(
            text_x,
            y,
            f"({label_text})  {count}",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=8.5,   # slightly smaller text
            zorder=7,
        )

# -------- general plotters --------
def plot_pie_counts(
    counts: pd.Series,
    title: str,
    out_file: Path,
    color_map: Optional[Dict[str, str]] = None,
    show_counts: bool = False,
    total_n: Optional[int] = None,
) -> None:
    counts = counts[counts > 0]

    if counts.empty:
        return

    total_val: float = float(counts.sum())
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    colors = [color_map.get(label) for label in counts.index] if color_map is not None else None

    def autopct_func(pct: float) -> str:
        if show_counts:
            val = int(round(pct * total_val / 100.0))
            return f"{pct:.1f}%\n(n={val})"
        return f"{pct:.1f}%"

    ax.pie(
        counts.values,
        labels=counts.index,
        autopct=autopct_func,
        startangle=90,
        colors=colors,
    )
    ax.set_title(title)
    ax.axis("equal")

    if total_n is not None:
        ax.text(
            0.98,
            0.98,
            f"n={total_n}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        )

    save_close(fig, out_file)


def plot_bar_counts(
    counts: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    out_file: Path,
    rotation: int = 0,
    color_map: Optional[Dict[str, str]] = None,
) -> None:
    if counts.empty:
        return

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    labels = counts.index.astype(str)
    colors = None if color_map is None else [color_map.get(label, None) for label in labels]

    ax.bar(labels, counts.values, color=colors)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    for i, val in enumerate(counts.values):
        ax.text(i, val, str(val), ha="center", va="bottom")

    plt.xticks(rotation=rotation)
    save_close(fig, out_file)


def plot_hist_overlay(
    series_map: Dict[str, pd.Series],
    title: str,
    xlabel: str,
    ylabel: str,
    out_file: Path,
    bins: int = 30,
    density: bool = False,
    color_map: Optional[Dict[str, str]] = None,
) -> None:
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    clean_series_map: Dict[str, pd.Series] = {}
    for label, series in series_map.items():
        vals: pd.Series = to_numeric_series(series).dropna()
        if not vals.empty:
            clean_series_map[label] = vals

    if not clean_series_map:
        plt.close(fig)
        return

    bin_edges = build_common_bin_edges(clean_series_map, bins=bins)
    if bin_edges is None:
        plt.close(fig)
        return

    plotted_any: bool = False
    for label, vals in clean_series_map.items():
        ax.hist(
            vals,
            bins=bin_edges,
            alpha=0.5,
            edgecolor="black",
            label=label,
            density=density,
            color=None if color_map is None else color_map.get(label),
        )
        plotted_any = True

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if plotted_any:
        ax.legend()

    save_close(fig, out_file)


def plot_single_hist(
    series: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    out_file: Path,
    bin_count: Optional[int] = 30,
    bin_size: Optional[float] = None,
    color: Optional[str] = None,
    vline_x: Optional[float] = None,
) -> None:
    vals: pd.Series = to_numeric_series(series).dropna()
    if vals.empty:
        return

    bins = resolve_hist_bins(vals, bin_count=bin_count, bin_size=bin_size)

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    ax.hist(vals, bins=bins, edgecolor="black", color=color)

    if vline_x is not None:
        ax.axvline(vline_x, linestyle="--", linewidth=1.5, color="black")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    save_close(fig, out_file)


def plot_scatter_by_group(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    out_file: Path,
    color_map: Optional[Dict[str, str]] = None,
) -> None:
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    plotted_any: bool = False
    for group_name, sub_df in df.groupby(group_col):
        x: pd.Series = pd.to_numeric(sub_df[x_col], errors="coerce")
        y: pd.Series = pd.to_numeric(sub_df[y_col], errors="coerce")
        mask: pd.Series = x.notna() & y.notna()

        if mask.sum() == 0:
            continue

        ax.scatter(
            x[mask],
            y[mask],
            alpha=0.75,
            label=str(group_name),
            color=None if color_map is None else color_map.get(group_name),
        )
        plotted_any = True

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if plotted_any:
        ax.legend()

    save_close(fig, out_file)


def plot_box_by_group(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    out_file: Path,
    group_order: Optional[list] = None,
) -> None:
    plot_df: pd.DataFrame = df[[value_col, group_col]].copy()
    plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[value_col, group_col])

    if plot_df.empty:
        return

    if group_order is None:
        group_order = list(plot_df[group_col].dropna().unique())

    data: list[pd.Series] = []
    labels: list[str] = []

    for group in group_order:
        vals: pd.Series = plot_df.loc[plot_df[group_col] == group, value_col].dropna()
        if vals.empty:
            continue
        data.append(vals)
        labels.append(group)

    if not data:
        return

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    ax.boxplot(data, tick_labels=labels)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    save_close(fig, out_file)


def plot_stacked_bar_counts(
    counts_df: pd.DataFrame,
    title: str,
    xlabel: str,
    ylabel: str,
    out_file: Path,
    color_map: Optional[Dict[str, str]] = None,
    show_segment_labels: bool = True,
) -> None:
    if counts_df.empty:
        return

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    categories = counts_df.index.astype(str).tolist()
    model_labels = counts_df.columns.tolist()
    bottoms = [0] * len(categories)

    for model_label in model_labels:
        vals = counts_df[model_label].tolist()
        color = None if color_map is None else color_map.get(model_label)
        bars = ax.bar(categories, vals, bottom=bottoms, label=model_label, color=color)

        if show_segment_labels:
            for i, (bar, val) in enumerate(zip(bars, vals)):
                total = counts_df.iloc[i].sum()
                if val <= 0 or total <= 0:
                    continue

                pct = 100.0 * val / total
                y = bottoms[i] + (val / 2.0)
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    y,
                    f"{val}\n{pct:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                )

        bottoms = [b + v for b, v in zip(bottoms, vals)]

    for i, total in enumerate(counts_df.sum(axis=1).tolist()):
        ax.text(i, total, f"n={int(total)}", ha="center", va="bottom")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

    save_close(fig, out_file)


# -------- class-colored histogram --------
def plot_hist_by_class(
    df: pd.DataFrame,
    value_col: str,
    class_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    out_file: Path,
    class_order: list[int],
    class_color_map: Dict[int, str],
    bin_size: float,
    summary_rows: Optional[list[tuple[int, str, int]]] = None,
    summary_loc: str = "upper right",
) -> None:
    plot_df: pd.DataFrame = df[[value_col, class_col]].copy()
    plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors="coerce")
    plot_df[class_col] = pd.to_numeric(plot_df[class_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[value_col, class_col])

    if plot_df.empty:
        return

    min_val: float = plot_df[value_col].min()
    max_val: float = plot_df[value_col].max()
    start: float = np.floor(min_val / bin_size) * bin_size
    end: float = np.ceil(max_val / bin_size) * bin_size + bin_size
    bins: np.ndarray = np.arange(start, end + bin_size, bin_size)

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    data: list[pd.Series] = []
    colors: list[str] = []

    for class_value in class_order:
        vals: pd.Series = plot_df.loc[plot_df[class_col] == class_value, value_col].dropna()
        if vals.empty:
            continue
        data.append(vals)
        colors.append(class_color_map.get(class_value, "#BDBDBD"))

    if not data:
        plt.close(fig)
        return

    ax.hist(data, bins=bins, stacked=True, color=colors, edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if summary_rows:
        add_class_summary_box(ax, summary_rows, class_color_map=class_color_map, loc=summary_loc)

    save_close(fig, out_file)