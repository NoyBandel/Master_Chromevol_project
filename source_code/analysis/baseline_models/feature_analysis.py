import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind, ttest_rel

from source_code.analysis.analysis_constants import *
from source_code.analysis.baseline_models.plot_utils import ensure_dir
from source_code.logger import log_run


INFERRED_FEATURE_COLS = [
    ROOT_CHROM_NUM_COL,
    BASE_CHROM_NUM_COL,
    NUM_OF_EVENTS_COL,
]

METADATA_FEATURE_COLS = [
    col for col in FEATURE_ANALYSIS_COLS
    if col not in INFERRED_FEATURE_COLS
]


# -------- loading --------
def load_feature_table(transition_label: str, function_label: str) -> pd.DataFrame:
    file_path: Path = BASELINE_ANALYSIS_DIR / transition_label / f"{transition_label}_{function_label}_{FEATURES_SUMMARY_SUFFIX}"

    if not file_path.exists():
        raise FileNotFoundError(f"Missing feature table: {file_path}")

    return pd.read_csv(file_path)


def load_chosen_table(transition_label: str) -> pd.DataFrame:
    file_path: Path = BASELINE_ANALYSIS_DIR / transition_label / f"{transition_label}_{CHOSEN_MODEL_SUFFIX}"

    if not file_path.exists():
        raise FileNotFoundError(f"Missing chosen model table: {file_path}")

    return pd.read_csv(file_path)


# -------- analysis dfs --------
def build_chosen_analysis_df(transition_label: str) -> pd.DataFrame:
    constant_df: pd.DataFrame = load_feature_table(transition_label, LABEL_CONSTANT)
    linear_df: pd.DataFrame = load_feature_table(transition_label, LABEL_LINEAR)
    chosen_df: pd.DataFrame = load_chosen_table(transition_label)

    constant_df = constant_df[constant_df[CHOSEN_MODEL_COL] == 1].copy()
    linear_df = linear_df[linear_df[CHOSEN_MODEL_COL] == 1].copy()

    constant_df[CHOSEN_FUNCTION_LABEL_COL] = LABEL_CONSTANT
    linear_df[CHOSEN_FUNCTION_LABEL_COL] = LABEL_LINEAR

    df: pd.DataFrame = pd.concat([constant_df, linear_df], ignore_index=True)

    chosen_info_df: pd.DataFrame = chosen_df[
        [
            FAMILY_NAME_COL,
            CHOSEN_FUNCTION_LABEL_COL,
            DELTA_BEST_VS_SECOND_COL,
            BEST_AKAIKE_WEIGHT_COL,
            DELTA_SUPPORT_CLASS_COL,
            WEIGHT_SUPPORT_CLASS_COL,
        ]
    ].copy()

    df = df.merge(
        chosen_info_df,
        on=[FAMILY_NAME_COL, CHOSEN_FUNCTION_LABEL_COL],
        how="left",
    )

    return df


def build_paired_inferred_df(transition_label: str) -> pd.DataFrame:
    constant_df: pd.DataFrame = load_feature_table(transition_label, LABEL_CONSTANT).copy()
    linear_df: pd.DataFrame = load_feature_table(transition_label, LABEL_LINEAR).copy()

    constant_df = constant_df[[FAMILY_NAME_COL] + INFERRED_FEATURE_COLS].copy()
    linear_df = linear_df[[FAMILY_NAME_COL] + INFERRED_FEATURE_COLS].copy()

    constant_df = constant_df.rename(
        columns={col: f"{col}_{LABEL_CONSTANT}" for col in INFERRED_FEATURE_COLS}
    )
    linear_df = linear_df.rename(
        columns={col: f"{col}_{LABEL_LINEAR}" for col in INFERRED_FEATURE_COLS}
    )

    paired_df: pd.DataFrame = constant_df.merge(
        linear_df,
        on=FAMILY_NAME_COL,
        how="inner",
    )

    for feature_col in INFERRED_FEATURE_COLS:
        paired_df[f"{feature_col}_{LABEL_CONSTANT}"] = pd.to_numeric(
            paired_df[f"{feature_col}_{LABEL_CONSTANT}"],
            errors="coerce",
        )
        paired_df[f"{feature_col}_{LABEL_LINEAR}"] = pd.to_numeric(
            paired_df[f"{feature_col}_{LABEL_LINEAR}"],
            errors="coerce",
        )

    return paired_df


# -------- stats helpers --------
def to_numeric_clean(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").dropna()


def compute_cohens_d(linear_vals: pd.Series, constant_vals: pd.Series) -> float:
    linear_vals = to_numeric_clean(linear_vals)
    constant_vals = to_numeric_clean(constant_vals)

    if len(linear_vals) < 2 or len(constant_vals) < 2:
        return np.nan

    linear_var: float = float(linear_vals.var(ddof=1))
    constant_var: float = float(constant_vals.var(ddof=1))
    pooled_std: float = float(
        np.sqrt(
            ((len(linear_vals) - 1) * linear_var + (len(constant_vals) - 1) * constant_var)
            / (len(linear_vals) + len(constant_vals) - 2)
        )
    )

    if pooled_std == 0:
        return np.nan

    return float((linear_vals.mean() - constant_vals.mean()) / pooled_std)


def compute_paired_cohens_dz(linear_vals: pd.Series, constant_vals: pd.Series) -> float:
    paired_df: pd.DataFrame = pd.DataFrame(
        {
            "linear": pd.to_numeric(linear_vals, errors="coerce"),
            "constant": pd.to_numeric(constant_vals, errors="coerce"),
        }
    ).dropna()

    if len(paired_df) < 2:
        return np.nan

    diffs: pd.Series = paired_df["linear"] - paired_df["constant"]
    diff_std: float = float(diffs.std(ddof=1))

    if diff_std == 0:
        return np.nan

    return float(diffs.mean() / diff_std)


def get_feature_groups(df: pd.DataFrame, feature_col: str) -> Tuple[pd.Series, pd.Series]:
    sub_df: pd.DataFrame = df[[feature_col, CHOSEN_FUNCTION_LABEL_COL]].copy()
    sub_df[feature_col] = pd.to_numeric(sub_df[feature_col], errors="coerce")
    sub_df = sub_df.dropna(subset=[feature_col, CHOSEN_FUNCTION_LABEL_COL])

    constant_vals: pd.Series = sub_df.loc[sub_df[CHOSEN_FUNCTION_LABEL_COL] == LABEL_CONSTANT, feature_col]
    linear_vals: pd.Series = sub_df.loc[sub_df[CHOSEN_FUNCTION_LABEL_COL] == LABEL_LINEAR, feature_col]

    return constant_vals, linear_vals


def get_paired_feature_groups(paired_df: pd.DataFrame, feature_col: str) -> Tuple[pd.Series, pd.Series]:
    const_col: str = f"{feature_col}_{LABEL_CONSTANT}"
    lin_col: str = f"{feature_col}_{LABEL_LINEAR}"

    sub_df: pd.DataFrame = paired_df[[const_col, lin_col]].copy()
    sub_df[const_col] = pd.to_numeric(sub_df[const_col], errors="coerce")
    sub_df[lin_col] = pd.to_numeric(sub_df[lin_col], errors="coerce")
    sub_df = sub_df.dropna()

    constant_vals: pd.Series = sub_df[const_col]
    linear_vals: pd.Series = sub_df[lin_col]

    return constant_vals, linear_vals


def build_unpaired_feature_stats_row(df: pd.DataFrame, feature_col: str) -> Dict[str, object]:
    constant_vals, linear_vals = get_feature_groups(df, feature_col)

    if constant_vals.empty or linear_vals.empty:
        return {
            "feature": feature_col,
            "comparison_type": "metadata_unpaired_chosen",
            "n_constant": len(constant_vals),
            "n_linear": len(linear_vals),
            "n_pairs": np.nan,
            "mean_constant": np.nan,
            "mean_linear": np.nan,
            "std_constant": np.nan,
            "std_linear": np.nan,
            "mean_diff_linear_minus_constant": np.nan,
            "welch_t_stat": np.nan,
            "welch_p_value": np.nan,
            "paired_t_stat": np.nan,
            "paired_p_value": np.nan,
            "mannwhitney_u": np.nan,
            "mannwhitney_p_value": np.nan,
            "cohens_d": np.nan,
            "paired_cohens_dz": np.nan,
        }

    t_stat, t_p = ttest_ind(linear_vals, constant_vals, equal_var=False, nan_policy="omit")
    u_stat, u_p = mannwhitneyu(linear_vals, constant_vals, alternative="two-sided")

    return {
        "feature": feature_col,
        "comparison_type": "metadata_unpaired_chosen",
        "n_constant": len(constant_vals),
        "n_linear": len(linear_vals),
        "n_pairs": np.nan,
        "mean_constant": float(constant_vals.mean()),
        "mean_linear": float(linear_vals.mean()),
        "std_constant": float(constant_vals.std(ddof=1)),
        "std_linear": float(linear_vals.std(ddof=1)),
        "mean_diff_linear_minus_constant": float(linear_vals.mean() - constant_vals.mean()),
        "welch_t_stat": float(t_stat),
        "welch_p_value": float(t_p),
        "paired_t_stat": np.nan,
        "paired_p_value": np.nan,
        "mannwhitney_u": float(u_stat),
        "mannwhitney_p_value": float(u_p),
        "cohens_d": float(compute_cohens_d(linear_vals, constant_vals)),
        "paired_cohens_dz": np.nan,
    }


def build_paired_feature_stats_row(paired_df: pd.DataFrame, feature_col: str) -> Dict[str, object]:
    constant_vals, linear_vals = get_paired_feature_groups(paired_df, feature_col)

    if constant_vals.empty or linear_vals.empty:
        return {
            "feature": feature_col,
            "comparison_type": "inferred_paired_all_models",
            "n_constant": len(constant_vals),
            "n_linear": len(linear_vals),
            "n_pairs": len(constant_vals),
            "mean_constant": np.nan,
            "mean_linear": np.nan,
            "std_constant": np.nan,
            "std_linear": np.nan,
            "mean_diff_linear_minus_constant": np.nan,
            "welch_t_stat": np.nan,
            "welch_p_value": np.nan,
            "paired_t_stat": np.nan,
            "paired_p_value": np.nan,
            "mannwhitney_u": np.nan,
            "mannwhitney_p_value": np.nan,
            "cohens_d": np.nan,
            "paired_cohens_dz": np.nan,
        }

    paired_t_stat, paired_t_p = ttest_rel(linear_vals, constant_vals, nan_policy="omit")

    return {
        "feature": feature_col,
        "comparison_type": "inferred_paired_all_models",
        "n_constant": len(constant_vals),
        "n_linear": len(linear_vals),
        "n_pairs": len(constant_vals),
        "mean_constant": float(constant_vals.mean()),
        "mean_linear": float(linear_vals.mean()),
        "std_constant": float(constant_vals.std(ddof=1)),
        "std_linear": float(linear_vals.std(ddof=1)),
        "mean_diff_linear_minus_constant": float((linear_vals - constant_vals).mean()),
        "welch_t_stat": np.nan,
        "welch_p_value": np.nan,
        "paired_t_stat": float(paired_t_stat),
        "paired_p_value": float(paired_t_p),
        "mannwhitney_u": np.nan,
        "mannwhitney_p_value": np.nan,
        "cohens_d": np.nan,
        "paired_cohens_dz": float(compute_paired_cohens_dz(linear_vals, constant_vals)),
    }


def build_unpaired_feature_stats_box_lines(df: pd.DataFrame, feature_col: str) -> List[str]:
    stats_row: Dict[str, object] = build_unpaired_feature_stats_row(df, feature_col)

    return [
        f"n const = {stats_row['n_constant']}",
        f"n lin = {stats_row['n_linear']}",
        f"μ const = {stats_row['mean_constant']:.2f}" if pd.notna(stats_row["mean_constant"]) else "μ const = NA",
        f"μ lin = {stats_row['mean_linear']:.2f}" if pd.notna(stats_row["mean_linear"]) else "μ lin = NA",
        f"Δμ = {stats_row['mean_diff_linear_minus_constant']:.2f}" if pd.notna(stats_row["mean_diff_linear_minus_constant"]) else "Δμ = NA",
        f"Welch p = {stats_row['welch_p_value']:.3g}" if pd.notna(stats_row["welch_p_value"]) else "Welch p = NA",
        f"MWU p = {stats_row['mannwhitney_p_value']:.3g}" if pd.notna(stats_row["mannwhitney_p_value"]) else "MWU p = NA",
        f"d = {stats_row['cohens_d']:.2f}" if pd.notna(stats_row["cohens_d"]) else "d = NA",
    ]


def build_paired_feature_stats_box_lines(paired_df: pd.DataFrame, feature_col: str) -> List[str]:
    stats_row: Dict[str, object] = build_paired_feature_stats_row(paired_df, feature_col)

    return [
        f"n pairs = {stats_row['n_pairs']}",
        f"μ const = {stats_row['mean_constant']:.2f}" if pd.notna(stats_row["mean_constant"]) else "μ const = NA",
        f"μ lin = {stats_row['mean_linear']:.2f}" if pd.notna(stats_row["mean_linear"]) else "μ lin = NA",
        f"mean paired Δ = {stats_row['mean_diff_linear_minus_constant']:.2f}" if pd.notna(stats_row["mean_diff_linear_minus_constant"]) else "mean paired Δ = NA",
        f"paired p = {stats_row['paired_p_value']:.3g}" if pd.notna(stats_row["paired_p_value"]) else "paired p = NA",
        f"dz = {stats_row['paired_cohens_dz']:.2f}" if pd.notna(stats_row["paired_cohens_dz"]) else "dz = NA",
    ]


def add_stats_box(ax, lines: List[str], loc: str = "upper right") -> None:
    text: str = "\n".join(lines)

    if loc == "upper right":
        x, y, ha = 0.98, 0.98, "right"
    elif loc == "upper left":
        x, y, ha = 0.02, 0.98, "left"
    else:
        x, y, ha = 0.98, 0.98, "right"

    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )


# -------- threshold scan --------
def build_threshold_scan_table(df: pd.DataFrame, feature_col: str) -> pd.DataFrame:
    scan_df: pd.DataFrame = df[
        [
            feature_col,
            CHOSEN_FUNCTION_LABEL_COL,
            DELTA_BEST_VS_SECOND_COL,
            BEST_AKAIKE_WEIGHT_COL,
            DELTA_SUPPORT_CLASS_COL,
            WEIGHT_SUPPORT_CLASS_COL,
        ]
    ].copy()

    scan_df[feature_col] = pd.to_numeric(scan_df[feature_col], errors="coerce")
    scan_df[DELTA_BEST_VS_SECOND_COL] = pd.to_numeric(scan_df[DELTA_BEST_VS_SECOND_COL], errors="coerce")
    scan_df[BEST_AKAIKE_WEIGHT_COL] = pd.to_numeric(scan_df[BEST_AKAIKE_WEIGHT_COL], errors="coerce")
    scan_df[DELTA_SUPPORT_CLASS_COL] = pd.to_numeric(scan_df[DELTA_SUPPORT_CLASS_COL], errors="coerce")
    scan_df = scan_df.dropna(subset=[feature_col])

    if scan_df.empty:
        return pd.DataFrame()

    unique_vals: np.ndarray = np.sort(scan_df[feature_col].unique())
    thresholds: List[float] = (
        [float(v) for v in np.unique(scan_df[feature_col].quantile(np.linspace(0.05, 0.95, 19)).values)]
        if len(unique_vals) > 20
        else [float(v) for v in unique_vals]
    )

    rows: List[Dict[str, object]] = []

    for threshold in thresholds:
        high_df: pd.DataFrame = scan_df[scan_df[feature_col] >= threshold].copy()

        if high_df.empty:
            continue

        n_high: int = len(high_df)
        n_linear: int = int((high_df[CHOSEN_FUNCTION_LABEL_COL] == LABEL_LINEAR).sum())
        n_constant: int = int((high_df[CHOSEN_FUNCTION_LABEL_COL] == LABEL_CONSTANT).sum())
        constant_strong_delta: int = int(
            (
                (high_df[CHOSEN_FUNCTION_LABEL_COL] == LABEL_CONSTANT)
                & (high_df[DELTA_SUPPORT_CLASS_COL] >= 3)
            ).sum()
        )

        rows.append(
            {
                "threshold": float(threshold),
                "n_above_threshold": n_high,
                "n_linear": n_linear,
                "n_constant": n_constant,
                "pct_linear": 100.0 * n_linear / n_high,
                "pct_constant": 100.0 * n_constant / n_high,
                "n_constant_strong_delta": constant_strong_delta,
                "pct_constant_strong_delta": 100.0 * constant_strong_delta / n_high,
                "mean_delta_best_vs_second": float(high_df[DELTA_BEST_VS_SECOND_COL].mean()) if high_df[DELTA_BEST_VS_SECOND_COL].notna().any() else np.nan,
                "mean_best_akaike_weight": float(high_df[BEST_AKAIKE_WEIGHT_COL].mean()) if high_df[BEST_AKAIKE_WEIGHT_COL].notna().any() else np.nan,
            }
        )

    return pd.DataFrame(rows)


def build_threshold_box_lines(threshold_df: pd.DataFrame) -> List[str]:
    if threshold_df.empty:
        return ["No valid thresholds"]

    max_linear_idx: int = int(threshold_df["pct_linear"].idxmax())
    max_linear_row: pd.Series = threshold_df.loc[max_linear_idx]

    threshold_50_df: pd.DataFrame = threshold_df[threshold_df["pct_linear"] >= 50]
    threshold_70_df: pd.DataFrame = threshold_df[threshold_df["pct_linear"] >= 70]

    threshold_50_text: str = f"{float(threshold_50_df.iloc[0]['threshold']):.2f}" if not threshold_50_df.empty else "NA"
    threshold_70_text: str = f"{float(threshold_70_df.iloc[0]['threshold']):.2f}" if not threshold_70_df.empty else "NA"

    return [
        f"max %lin = {float(max_linear_row['pct_linear']):.1f}",
        f"at thr = {float(max_linear_row['threshold']):.2f}",
        f"%lin >= 50 at {threshold_50_text}",
        f"%lin >= 70 at {threshold_70_text}",
        f"max n const strong Δ = {int(threshold_df['n_constant_strong_delta'].max())}",
        f"max mean ΔAICc = {float(threshold_df['mean_delta_best_vs_second'].max()):.2f}" if threshold_df["mean_delta_best_vs_second"].notna().any() else "max mean ΔAICc = NA",
    ]


# -------- plots: metadata / chosen --------
def plot_feature_boxplot(df: pd.DataFrame, feature_col: str, out_file: Path) -> None:
    plot_df: pd.DataFrame = df[[feature_col, CHOSEN_FUNCTION_LABEL_COL]].copy()
    plot_df[feature_col] = pd.to_numeric(plot_df[feature_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[feature_col, CHOSEN_FUNCTION_LABEL_COL])

    if plot_df.empty:
        return

    group_order: List[str] = [LABEL_CONSTANT, LABEL_LINEAR]
    data: List[pd.Series] = [
        plot_df.loc[plot_df[CHOSEN_FUNCTION_LABEL_COL] == group, feature_col].dropna()
        for group in group_order
    ]
    data = [vals for vals in data if not vals.empty]
    labels: List[str] = [
        group for group in group_order
        if not plot_df.loc[plot_df[CHOSEN_FUNCTION_LABEL_COL] == group, feature_col].dropna().empty
    ]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.boxplot(data, tick_labels=labels)
    ax.set_title(f"{feature_col}: constant vs linear chosen")
    ax.set_xlabel("Chosen model")
    ax.set_ylabel(feature_col)
    add_stats_box(ax, build_unpaired_feature_stats_box_lines(df, feature_col), loc="upper right")
    fig.tight_layout()
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_feature_hist(df: pd.DataFrame, feature_col: str, out_file: Path) -> None:
    constant_vals, linear_vals = get_feature_groups(df, feature_col)

    if constant_vals.empty and linear_vals.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    vals_list: List[pd.Series] = []
    labels: List[str] = []
    colors: List[str] = []

    if not constant_vals.empty:
        vals_list.append(constant_vals)
        labels.append(LABEL_CONSTANT)
        colors.append(MODEL_COLOR_MAP[LABEL_CONSTANT])

    if not linear_vals.empty:
        vals_list.append(linear_vals)
        labels.append(LABEL_LINEAR)
        colors.append(MODEL_COLOR_MAP[LABEL_LINEAR])

    pooled: pd.Series = pd.concat(vals_list, ignore_index=True)
    bins = 20 if pooled.nunique() > 20 else min(10, max(3, pooled.nunique()))

    for vals, label, color in zip(vals_list, labels, colors):
        ax.hist(vals, bins=bins, alpha=0.5, edgecolor="black", label=label, color=color)

    ax.set_title(f"{feature_col}: constant vs linear chosen")
    ax.set_xlabel(feature_col)
    ax.set_ylabel("Number of families")
    ax.legend()
    add_stats_box(ax, build_unpaired_feature_stats_box_lines(df, feature_col), loc="upper right")
    fig.tight_layout()
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_feature_delta_scatter(df: pd.DataFrame, feature_col: str, out_file: Path) -> None:
    plot_df: pd.DataFrame = df[[feature_col, DELTA_BEST_VS_SECOND_COL, CHOSEN_FUNCTION_LABEL_COL]].copy()
    plot_df[feature_col] = pd.to_numeric(plot_df[feature_col], errors="coerce")
    plot_df[DELTA_BEST_VS_SECOND_COL] = pd.to_numeric(plot_df[DELTA_BEST_VS_SECOND_COL], errors="coerce")
    plot_df = plot_df.dropna(subset=[feature_col, DELTA_BEST_VS_SECOND_COL, CHOSEN_FUNCTION_LABEL_COL])

    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    for label in [LABEL_CONSTANT, LABEL_LINEAR]:
        sub_df: pd.DataFrame = plot_df[plot_df[CHOSEN_FUNCTION_LABEL_COL] == label]
        if sub_df.empty:
            continue

        ax.scatter(
            sub_df[feature_col],
            sub_df[DELTA_BEST_VS_SECOND_COL],
            alpha=0.75,
            label=label,
            color=MODEL_COLOR_MAP[label],
        )

    ax.set_title(f"{feature_col} vs {DELTA_BEST_VS_SECOND_COL}")
    ax.set_xlabel(feature_col)
    ax.set_ylabel(DELTA_BEST_VS_SECOND_COL)
    ax.legend()

    corr_df: pd.DataFrame = plot_df[[feature_col, DELTA_BEST_VS_SECOND_COL]].dropna()
    pearson_r: float = float(corr_df[feature_col].corr(corr_df[DELTA_BEST_VS_SECOND_COL], method="pearson")) if len(corr_df) >= 3 else np.nan
    spearman_r: float = float(corr_df[feature_col].corr(corr_df[DELTA_BEST_VS_SECOND_COL], method="spearman")) if len(corr_df) >= 3 else np.nan

    add_stats_box(
        ax,
        [
            f"n = {len(corr_df)}",
            f"pearson r = {pearson_r:.2f}" if pd.notna(pearson_r) else "pearson r = NA",
            f"spearman r = {spearman_r:.2f}" if pd.notna(spearman_r) else "spearman r = NA",
            f"max ΔAICc = {float(plot_df[DELTA_BEST_VS_SECOND_COL].max()):.2f}",
        ],
        loc="upper right",
    )

    fig.tight_layout()
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -------- plots: inferred / paired --------
def plot_paired_feature_boxplot(paired_df: pd.DataFrame, feature_col: str, out_file: Path) -> None:
    constant_vals, linear_vals = get_paired_feature_groups(paired_df, feature_col)

    if constant_vals.empty or linear_vals.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.boxplot([constant_vals, linear_vals], tick_labels=[LABEL_CONSTANT, LABEL_LINEAR])
    ax.set_title(f"{feature_col}: all constant vs all linear (paired)")
    ax.set_xlabel("Model")
    ax.set_ylabel(feature_col)
    add_stats_box(ax, build_paired_feature_stats_box_lines(paired_df, feature_col), loc="upper right")
    fig.tight_layout()
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_paired_feature_hist(paired_df: pd.DataFrame, feature_col: str, out_file: Path) -> None:
    constant_vals, linear_vals = get_paired_feature_groups(paired_df, feature_col)

    if constant_vals.empty or linear_vals.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    pooled: pd.Series = pd.concat([constant_vals, linear_vals], ignore_index=True)
    bins = 20 if pooled.nunique() > 20 else min(10, max(3, pooled.nunique()))

    ax.hist(
        constant_vals,
        bins=bins,
        alpha=0.5,
        edgecolor="black",
        label=LABEL_CONSTANT,
        color=MODEL_COLOR_MAP[LABEL_CONSTANT],
    )
    ax.hist(
        linear_vals,
        bins=bins,
        alpha=0.5,
        edgecolor="black",
        label=LABEL_LINEAR,
        color=MODEL_COLOR_MAP[LABEL_LINEAR],
    )

    ax.set_title(f"{feature_col}: all constant vs all linear (paired)")
    ax.set_xlabel(feature_col)
    ax.set_ylabel("Number of families")
    ax.legend()
    add_stats_box(ax, build_paired_feature_stats_box_lines(paired_df, feature_col), loc="upper right")
    fig.tight_layout()
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_paired_feature_difference_hist(paired_df: pd.DataFrame, feature_col: str, out_file: Path) -> None:
    constant_vals, linear_vals = get_paired_feature_groups(paired_df, feature_col)

    if constant_vals.empty or linear_vals.empty:
        return

    diffs: pd.Series = linear_vals - constant_vals
    diffs = diffs.dropna()

    if diffs.empty:
        return

    bins = 20 if diffs.nunique() > 20 else min(10, max(3, diffs.nunique()))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(diffs, bins=bins, alpha=0.7, edgecolor="black")
    ax.axvline(0, linestyle="--", linewidth=1.5, color="black")

    ax.set_title(f"{feature_col}: linear - constant")
    ax.set_xlabel(f"{feature_col} difference")
    ax.set_ylabel("Number of families")

    mean_diff: float = float(diffs.mean()) if not diffs.empty else np.nan
    median_diff: float = float(diffs.median()) if not diffs.empty else np.nan

    add_stats_box(
        ax,
        [
            f"n pairs = {len(diffs)}",
            f"mean Δ = {mean_diff:.2f}" if pd.notna(mean_diff) else "mean Δ = NA",
            f"median Δ = {median_diff:.2f}" if pd.notna(median_diff) else "median Δ = NA",
            f"min Δ = {float(diffs.min()):.2f}" if not diffs.empty else "min Δ = NA",
            f"max Δ = {float(diffs.max()):.2f}" if not diffs.empty else "max Δ = NA",
        ],
        loc="upper right",
    )

    fig.tight_layout()
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_summary(threshold_df: pd.DataFrame, feature_col: str, out_file: Path) -> None:
    if threshold_df.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(threshold_df["threshold"], threshold_df["pct_linear"], marker="o", label="% linear", color=MODEL_COLOR_MAP[LABEL_LINEAR])
    ax.plot(threshold_df["threshold"], threshold_df["pct_constant"], marker="o", label="% constant", color=MODEL_COLOR_MAP[LABEL_CONSTANT])

    ax.set_title(f"{feature_col}: threshold scan")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Percent")
    ax.legend()

    add_stats_box(ax, build_threshold_box_lines(threshold_df), loc="upper right")

    fig.tight_layout()
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metadata_feature_outputs(df: pd.DataFrame, feature_col: str, feature_out_dir: Path, output_paths: List[str]) -> None:
    boxplot_file: Path = feature_out_dir / f"{feature_col}_boxplot.png"
    hist_file: Path = feature_out_dir / f"{feature_col}_hist.png"
    scatter_file: Path = feature_out_dir / f"{feature_col}_delta_scatter.png"

    plot_feature_boxplot(df, feature_col, boxplot_file)
    plot_feature_hist(df, feature_col, hist_file)
    plot_feature_delta_scatter(df, feature_col, scatter_file)

    if boxplot_file.exists():
        output_paths.append(str(boxplot_file))
    if hist_file.exists():
        output_paths.append(str(hist_file))
    if scatter_file.exists():
        output_paths.append(str(scatter_file))


def plot_inferred_feature_outputs(paired_df: pd.DataFrame, feature_col: str, feature_out_dir: Path, output_paths: List[str]) -> None:
    boxplot_file: Path = feature_out_dir / f"{feature_col}_boxplot.png"
    hist_file: Path = feature_out_dir / f"{feature_col}_hist.png"
    diff_hist_file: Path = feature_out_dir / f"{feature_col}_difference_hist.png"

    plot_paired_feature_boxplot(paired_df, feature_col, boxplot_file)
    plot_paired_feature_hist(paired_df, feature_col, hist_file)
    plot_paired_feature_difference_hist(paired_df, feature_col, diff_hist_file)

    if boxplot_file.exists():
        output_paths.append(str(boxplot_file))
    if hist_file.exists():
        output_paths.append(str(hist_file))
    if diff_hist_file.exists():
        output_paths.append(str(diff_hist_file))


# -------- run --------
def run_transition(transition_label: str) -> Dict[str, object]:
    out_dir: Path = BASELINE_ANALYSIS_DIR / transition_label / FEATURE_ANALYSIS_SUBDIR
    chosen_df: pd.DataFrame = build_chosen_analysis_df(transition_label)
    paired_inferred_df: pd.DataFrame = build_paired_inferred_df(transition_label)

    output_paths: List[str] = []
    stats_rows: List[Dict[str, object]] = []

    ensure_dir(out_dir)

    for feature_col in FEATURE_ANALYSIS_COLS:
        feature_out_dir: Path = out_dir / feature_col
        ensure_dir(feature_out_dir)

        if feature_col in INFERRED_FEATURE_COLS:
            plot_inferred_feature_outputs(
                paired_inferred_df,
                feature_col,
                feature_out_dir,
                output_paths,
            )
            stats_rows.append(build_paired_feature_stats_row(paired_inferred_df, feature_col))
        else:
            plot_metadata_feature_outputs(
                chosen_df,
                feature_col,
                feature_out_dir,
                output_paths,
            )
            threshold_df: pd.DataFrame = build_threshold_scan_table(chosen_df, feature_col)
            threshold_file: Path = feature_out_dir / f"{feature_col}_threshold_scan.csv"
            threshold_plot_file: Path = feature_out_dir / f"{feature_col}_threshold_summary.png"

            threshold_df.to_csv(threshold_file, index=False)
            plot_threshold_summary(threshold_df, feature_col, threshold_plot_file)

            output_paths.append(str(threshold_file))
            if threshold_plot_file.exists():
                output_paths.append(str(threshold_plot_file))

            stats_rows.append(build_unpaired_feature_stats_row(chosen_df, feature_col))

    stats_df: pd.DataFrame = pd.DataFrame(stats_rows)
    stats_file: Path = out_dir / f"{transition_label}_feature_stats.csv"
    stats_df.to_csv(stats_file, index=False)
    output_paths.append(str(stats_file))

    model_counts: pd.Series = chosen_df[CHOSEN_FUNCTION_LABEL_COL].value_counts()

    return {
        "transition": transition_label,
        "n_families_total": len(chosen_df),
        "n_constant": int(model_counts.get(LABEL_CONSTANT, 0)),
        "n_linear": int(model_counts.get(LABEL_LINEAR, 0)),
        "features": FEATURE_ANALYSIS_COLS,
        "outputs": output_paths,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Constant vs linear feature analysis.")
    parser.add_argument("--transition", required=True, type=str, choices=list(LABEL_TRANSITIONS_ORDERED))
    return parser.parse_args()


def main() -> None:
    args: argparse.Namespace = parse_args()
    result: Dict[str, object] = run_transition(args.transition)

    # log_run(
    #     step="analysis",
    #     script=Path(__file__),
    #     params={
    #         "transition": result["transition"],
    #         "n_families_total": result["n_families_total"],
    #         "n_constant": result["n_constant"],
    #         "n_linear": result["n_linear"],
    #         "features": result["features"],
    #     },
    #     outputs=result["outputs"],
    #     description=f"Feature analysis for transition '{result['transition']}'",
    #     notes="Metadata features are analyzed as chosen-model unpaired comparisons; inferred features are analyzed as paired all-constant vs all-linear comparisons.",
    #     log_relative_path=Path(BASELINE_MODELS_LABEL) / f"{result['transition']}.log",
    # )


if __name__ == "__main__":
    main()