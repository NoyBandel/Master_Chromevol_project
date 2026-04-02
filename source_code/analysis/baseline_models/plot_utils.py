from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_FIGSIZE = (9, 6)
DEFAULT_DPI = 300


# Plot/output helpers
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_close(fig: plt.Figure, out_file: Path, dpi: int = DEFAULT_DPI) -> None:
    fig.tight_layout()
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_text(text: str, out_file: Path) -> None:
    out_file.write_text(text, encoding="utf-8")


# Data helpers
def to_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def build_stats(series: pd.Series) -> Dict[str, float]:
    s = to_numeric_series(series).dropna()

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
    counts = series.value_counts(dropna=False)

    if order is None:
        return counts

    return pd.Series({label: int(counts.get(label, 0)) for label in order})


def build_common_bin_edges(series_map: Dict[str, pd.Series], bins: int):
    clean_vals = []

    for series in series_map.values():
        vals = to_numeric_series(series).dropna()
        if not vals.empty:
            clean_vals.append(vals)

    if not clean_vals:
        return None

    pooled = pd.concat(clean_vals, ignore_index=True)
    _, bin_edges = pd.cut(pooled, bins=bins, retbins=True, duplicates="drop")
    return bin_edges


# Plotters
def plot_bar_counts(
    counts: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    out_file: Path,
    rotation: int = 0,
    color_map: Optional[Dict[str, str]] = None,
) -> None:
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

    clean_series_map = {}
    for label, series in series_map.items():
        vals = to_numeric_series(series).dropna()
        if not vals.empty:
            clean_series_map[label] = vals

    if not clean_series_map:
        plt.close(fig)
        return

    bin_edges = build_common_bin_edges(clean_series_map, bins=bins)
    if bin_edges is None:
        plt.close(fig)
        return

    plotted_any = False
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
    bins: int = 30,
    color: Optional[str] = None,
    vline_x: Optional[float] = None,
) -> None:
    vals = to_numeric_series(series).dropna()
    if vals.empty:
        return

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

    plotted_any = False
    for group_name, sub_df in df.groupby(group_col):
        x = pd.to_numeric(sub_df[x_col], errors="coerce")
        y = pd.to_numeric(sub_df[y_col], errors="coerce")
        mask = x.notna() & y.notna()

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
    plot_df = df[[value_col, group_col]].copy()
    plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[value_col, group_col])

    if plot_df.empty:
        return

    if group_order is None:
        group_order = list(plot_df[group_col].dropna().unique())

    data = []
    labels = []

    for group in group_order:
        vals = plot_df.loc[plot_df[group_col] == group, value_col].dropna()
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
        colors = None if color_map is None else color_map.get(model_label)

        bars = ax.bar(categories, vals, bottom=bottoms, label=model_label, color=colors)

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