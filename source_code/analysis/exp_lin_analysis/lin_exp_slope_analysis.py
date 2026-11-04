"""Analyze LINEAR and EXPONENTIAL slopes and summarize directional consistency.

Outputs per transition:
- EXP-LINEAR slope plotting table
- LINEAR/EXP sign-agreement pie
- 100% stacked sign-direction bar
- positive/negative direction bar for overall-chosen LINEAR and EXP families
- positive/negative slope-sign pie for overall-chosen EXP families
- effective-slope scatter
- chosen-model slope histograms
- LINEAR/EXP overlay histograms

Once gain, loss, and duplication sign-direction tables all exist, the script also
creates combined 3-transition summaries for both sign agreement and overall-chosen direction.

For corrected analysis, LINEAR p1/p2 and EXP p1 use tree-scale-corrected values.
EXP p2 is unchanged because it is a shape parameter; the corrected EXP effective
slope is recomputed across each family's observed chromosome-number range.
"""

import argparse
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from source_code.analysis.exp_lin_analysis.exp_lin_analysis_constants import *
from source_code.analysis.parameter_utils import CORRECTED_PARAMETER_SCALE, VALID_PARAMETER_SCALES, resolve_parameter_column
from source_code.analysis.plot_utils import compute_effective_slope, compute_exp_rate, ensure_dir, get_ordered_counts, plot_pie_counts, save_close, to_float
from source_code.constants import LABEL_DUPL, LABEL_GAIN, LABEL_LOSS
from source_code.logger import log_run


TREE_SCALING_FACTOR_COL: str = "tree_scaling_factor"
SIGN_DIRECTION_COL: str = "sign_direction"
SLOPE_SUMMARY_DIR_NAME: str = "slope_analysis_summary"

VALID_DIRECTION_LABELS: tuple[str, str] = (POSITIVE_SLOPE_LABEL, NEGATIVE_SLOPE_LABEL)
SIGN_AGREEMENT_ORDER: tuple[str, str] = (SIGN_AGREEMENT_LABEL, SIGN_DISAGREEMENT_LABEL)
TRANSITION_ORDER: tuple[str, str, str] = (LABEL_GAIN, LABEL_LOSS, LABEL_DUPL)
TRANSITION_DISPLAY_NAMES: dict[str, str] = {LABEL_GAIN: "Gain", LABEL_LOSS: "Loss", LABEL_DUPL: "Duplication"}

AGREE_POSITIVE_LABEL: str = "Agree: LINEAR + / EXP +"
AGREE_NEGATIVE_LABEL: str = "Agree: LINEAR - / EXP -"
DISAGREE_LINEAR_POSITIVE_LABEL: str = "Disagree: LINEAR + / EXP -"
DISAGREE_LINEAR_NEGATIVE_LABEL: str = "Disagree: LINEAR - / EXP +"
SIGN_DIRECTION_ORDER: tuple[str, str, str, str] = (
    AGREE_POSITIVE_LABEL, AGREE_NEGATIVE_LABEL, DISAGREE_LINEAR_POSITIVE_LABEL, DISAGREE_LINEAR_NEGATIVE_LABEL
)
SIGN_DIRECTION_COLOR_MAP: dict[str, str] = {
    AGREE_POSITIVE_LABEL: "#76C7B7",
    AGREE_NEGATIVE_LABEL: "#4FA997",
    DISAGREE_LINEAR_POSITIVE_LABEL: "#DFA5BE",
    DISAGREE_LINEAR_NEGATIVE_LABEL: "#C86B98",
}

OverlaySelection = Literal["beats_constant", "chosen"]


def save_table(df: pd.DataFrame, out_file: Path, output_paths: list[str]) -> None:
    df.to_csv(out_file, index=False); output_paths.append(out_file.as_posix())


def infer_sign(value: float) -> str:
    if pd.isna(value): return SIGN_MISSING_LABEL
    if value > 0: return POSITIVE_SLOPE_LABEL
    if value < 0: return NEGATIVE_SLOPE_LABEL
    return ZERO_SLOPE_LABEL


def resolve_slope_columns(parameter_scale: str) -> tuple[str, str]:
    return resolve_parameter_column(LINEAR_P2_COL, parameter_scale), resolve_parameter_column(EXP_EFFECTIVE_SLOPE_COL, parameter_scale)


def build_ignore_aicc_table(baseline_df: pd.DataFrame) -> pd.DataFrame:
    ignore_df: pd.DataFrame = baseline_df.loc[baseline_df[LABEL_FUNC_TYPE_COL] == LABEL_IGNORE, [FAMILY_NAME_COL, AICC_COL]].copy()
    ignore_df[AICC_COL] = pd.to_numeric(ignore_df[AICC_COL], errors="coerce")
    return ignore_df.groupby(FAMILY_NAME_COL, as_index=False)[AICC_COL].min().rename(columns={AICC_COL: IGNORE_AICC_COL})


def build_chosen_model_table(chosen_df: pd.DataFrame) -> pd.DataFrame:
    out: pd.DataFrame = chosen_df[[FAMILY_NAME_COL, CHOSEN_FUNCTION_LABEL_COL]].copy()
    if out[FAMILY_NAME_COL].duplicated().any():
        duplicates: list[str] = out.loc[out[FAMILY_NAME_COL].duplicated(keep=False), FAMILY_NAME_COL].astype(str).unique().tolist()
        raise ValueError(f"Duplicate families in chosen-model table: {duplicates}")
    return out.rename(columns={CHOSEN_FUNCTION_LABEL_COL: OVERALL_CHOSEN_MODEL_COL})


def calculate_exp_effective_slope(row: pd.Series, exp_p1_col: str) -> float:
    min_chr, max_chr = to_float(row[MIN_CHROM_COL]), to_float(row[MAX_CHROM_COL])
    exp_p1, exp_p2 = to_float(row[exp_p1_col]), to_float(row[EXP_P2_COL])
    return compute_effective_slope(to_float(compute_exp_rate(min_chr, exp_p1, exp_p2)), to_float(compute_exp_rate(max_chr, exp_p1, exp_p2)), min_chr, max_chr)


def build_plotting_table(
    core_df: pd.DataFrame,
    features_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    chosen_df: pd.DataFrame,
    parameter_scale: str,
) -> pd.DataFrame:
    corrected_cols: dict[str, str] = {
        base: resolve_parameter_column(base, CORRECTED_PARAMETER_SCALE)
        for base in (EXP_P1_COL, LINEAR_P1_COL, LINEAR_P2_COL, EXP_EFFECTIVE_SLOPE_COL)
    }
    base_core_cols: list[str] = [FAMILY_NAME_COL, M0_AICC_COL, M1_LINEAR_AICC_COL, M1_EXP_AICC_COL, EXP_P1_COL, EXP_P2_COL, LINEAR_P1_COL, LINEAR_P2_COL]
    required_cols: list[str] = base_core_cols + ([corrected_cols[EXP_P1_COL], corrected_cols[LINEAR_P1_COL], corrected_cols[LINEAR_P2_COL]] if parameter_scale == CORRECTED_PARAMETER_SCALE else [])
    missing_cols: list[str] = [col for col in required_cols if col not in core_df.columns]
    if missing_cols: raise KeyError(f"Missing required core-table columns for parameter scale '{parameter_scale}': {missing_cols}")

    optional_cols: list[str] = [col for col in corrected_cols.values() if col != corrected_cols[EXP_EFFECTIVE_SLOPE_COL] and col in core_df.columns]
    if TREE_SCALING_FACTOR_COL in core_df.columns: optional_cols.append(TREE_SCALING_FACTOR_COL)
    plotting_df: pd.DataFrame = core_df[list(dict.fromkeys(base_core_cols + optional_cols))].copy().rename(columns={M0_AICC_COL: CONSTANT_AICC_COL, M1_LINEAR_AICC_COL: LINEAR_AICC_COL, M1_EXP_AICC_COL: EXP_AICC_COL})

    features_small: pd.DataFrame = features_df[[FAMILY_NAME_COL, MIN_CHROM_COL, MAX_CHROM_COL]].drop_duplicates(FAMILY_NAME_COL)
    plotting_df = plotting_df.merge(build_ignore_aicc_table(baseline_df), on=FAMILY_NAME_COL, how="left", validate="one_to_one")
    plotting_df = plotting_df.merge(features_small, on=FAMILY_NAME_COL, how="left", validate="one_to_one")
    plotting_df = plotting_df.merge(build_chosen_model_table(chosen_df), on=FAMILY_NAME_COL, how="left", validate="one_to_one")

    missing_chosen: pd.Series = plotting_df[OVERALL_CHOSEN_MODEL_COL].isna()
    if missing_chosen.any(): raise ValueError(f"Families missing from chosen-model table: {plotting_df.loc[missing_chosen, FAMILY_NAME_COL].astype(str).tolist()}")

    numeric_cols: list[str] = [CONSTANT_AICC_COL, LINEAR_AICC_COL, EXP_AICC_COL, IGNORE_AICC_COL, EXP_P1_COL, EXP_P2_COL, LINEAR_P1_COL, LINEAR_P2_COL, MIN_CHROM_COL, MAX_CHROM_COL]
    numeric_cols += [col for col in optional_cols if col in plotting_df.columns]
    for col in dict.fromkeys(numeric_cols): plotting_df[col] = pd.to_numeric(plotting_df[col], errors="coerce")

    plotting_df[EXP_EFFECTIVE_SLOPE_COL] = plotting_df.apply(lambda row: calculate_exp_effective_slope(row, EXP_P1_COL), axis=1)
    corrected_exp_p1: str = corrected_cols[EXP_P1_COL]
    corrected_exp_slope: str = corrected_cols[EXP_EFFECTIVE_SLOPE_COL]
    if corrected_exp_p1 in plotting_df.columns: plotting_df[corrected_exp_slope] = plotting_df.apply(lambda row: calculate_exp_effective_slope(row, corrected_exp_p1), axis=1)

    preferred: list[str] = [FAMILY_NAME_COL]
    if TREE_SCALING_FACTOR_COL in plotting_df.columns: preferred.append(TREE_SCALING_FACTOR_COL)
    preferred += [OVERALL_CHOSEN_MODEL_COL, CONSTANT_AICC_COL, LINEAR_AICC_COL, EXP_AICC_COL, IGNORE_AICC_COL, EXP_P1_COL]
    if corrected_exp_p1 in plotting_df.columns: preferred.append(corrected_exp_p1)
    preferred += [EXP_P2_COL, EXP_EFFECTIVE_SLOPE_COL]
    if corrected_exp_slope in plotting_df.columns: preferred.append(corrected_exp_slope)
    preferred.append(LINEAR_P1_COL)
    if corrected_cols[LINEAR_P1_COL] in plotting_df.columns: preferred.append(corrected_cols[LINEAR_P1_COL])
    preferred.append(LINEAR_P2_COL)
    if corrected_cols[LINEAR_P2_COL] in plotting_df.columns: preferred.append(corrected_cols[LINEAR_P2_COL])
    preferred += [MIN_CHROM_COL, MAX_CHROM_COL]
    return plotting_df[list(dict.fromkeys(preferred))].sort_values(FAMILY_NAME_COL).reset_index(drop=True)


def build_sign_comparison_table(plotting_df: pd.DataFrame, linear_slope_col: str) -> pd.DataFrame:
    mask: pd.Series = (plotting_df[LINEAR_AICC_COL] < plotting_df[CONSTANT_AICC_COL]) & (plotting_df[EXP_AICC_COL] < plotting_df[CONSTANT_AICC_COL])
    sign_df: pd.DataFrame = plotting_df.loc[mask, [FAMILY_NAME_COL, linear_slope_col, EXP_P2_COL]].copy()
    sign_df["linear_sign"] = sign_df[linear_slope_col].map(lambda value: infer_sign(to_float(value)))
    sign_df["exp_sign"] = sign_df[EXP_P2_COL].map(lambda value: infer_sign(to_float(value)))
    sign_df = sign_df.loc[sign_df["linear_sign"].isin(VALID_DIRECTION_LABELS) & sign_df["exp_sign"].isin(VALID_DIRECTION_LABELS)].copy()
    sign_df[SIGN_AGREEMENT_STATUS_COL] = np.where(sign_df["linear_sign"] == sign_df["exp_sign"], SIGN_AGREEMENT_LABEL, SIGN_DISAGREEMENT_LABEL)

    conditions: list[pd.Series] = [
        (sign_df["linear_sign"] == POSITIVE_SLOPE_LABEL) & (sign_df["exp_sign"] == POSITIVE_SLOPE_LABEL),
        (sign_df["linear_sign"] == NEGATIVE_SLOPE_LABEL) & (sign_df["exp_sign"] == NEGATIVE_SLOPE_LABEL),
        (sign_df["linear_sign"] == POSITIVE_SLOPE_LABEL) & (sign_df["exp_sign"] == NEGATIVE_SLOPE_LABEL),
        (sign_df["linear_sign"] == NEGATIVE_SLOPE_LABEL) & (sign_df["exp_sign"] == POSITIVE_SLOPE_LABEL),
    ]
    sign_df[SIGN_DIRECTION_COL] = np.select(conditions, SIGN_DIRECTION_ORDER, default="")
    return sign_df


def summarize_counts(values: pd.Series, order: tuple[str, ...], category_col: str) -> pd.DataFrame:
    counts: pd.Series = get_ordered_counts(values, order=order)
    total: int = int(counts.sum())
    return pd.DataFrame({category_col: counts.index, "count": counts.values, "percentage": [round(100 * count / total, 2) if total else 0.0 for count in counts.values]})


def plot_sign_agreement_pie(sign_df: pd.DataFrame, transition_label: str, tables_dir: Path, plots_dir: Path, output_paths: list[str]) -> None:
    summary: pd.DataFrame = summarize_counts(sign_df[SIGN_AGREEMENT_STATUS_COL], SIGN_AGREEMENT_ORDER, SIGN_AGREEMENT_STATUS_COL)
    save_table(summary, tables_dir / f"{transition_label}_sign_agreement_counts.csv", output_paths)
    plot_counts: pd.Series = summary.set_index(SIGN_AGREEMENT_STATUS_COL)["count"]; plot_counts = plot_counts[plot_counts > 0]
    if plot_counts.empty: return
    out_file: Path = plots_dir / f"{transition_label}_sign_agreement_counts_pie.png"
    plot_pie_counts(counts=plot_counts, title=f"{transition_label}: LINEAR/EXP slope-sign agreement", out_file=out_file, color_map=SIGN_AGREEMENT_COLOR_MAP, show_counts=True, total_n=int(summary["count"].sum()))
    output_paths.append(out_file.as_posix())


def build_direction_summary(sign_df: pd.DataFrame) -> pd.DataFrame:
    summary: pd.DataFrame = summarize_counts(sign_df[SIGN_DIRECTION_COL], SIGN_DIRECTION_ORDER, SIGN_DIRECTION_COL)
    summary["agreement_status"] = np.where(summary[SIGN_DIRECTION_COL].isin((AGREE_POSITIVE_LABEL, AGREE_NEGATIVE_LABEL)), SIGN_AGREEMENT_LABEL, SIGN_DISAGREEMENT_LABEL)
    return summary


def agreement_totals(direction_summary: pd.DataFrame) -> tuple[int, int, int, float, float]:
    total: int = int(direction_summary["count"].sum())
    agreement: int = int(direction_summary.loc[direction_summary["agreement_status"] == SIGN_AGREEMENT_LABEL, "count"].sum())
    disagreement: int = total - agreement
    return agreement, disagreement, total, (100 * agreement / total if total else 0.0), (100 * disagreement / total if total else 0.0)


def plot_transition_sign_direction_bar(direction_summary: pd.DataFrame, transition_label: str, plots_dir: Path, output_paths: list[str]) -> None:
    agreement, disagreement, total, agreement_pct, disagreement_pct = agreement_totals(direction_summary)
    if total == 0: return

    fig, ax = plt.subplots(figsize=(10, 3.4)); left: float = 0.0
    for category in SIGN_DIRECTION_ORDER:
        row: pd.Series = direction_summary.loc[direction_summary[SIGN_DIRECTION_COL] == category].iloc[0]
        count, percentage = int(row["count"]), float(row["percentage"])
        if percentage <= 0: continue
        ax.barh([0], [percentage], left=left, height=0.55, color=SIGN_DIRECTION_COLOR_MAP[category], label=category)
        if percentage >= 5: ax.text(left + percentage / 2, 0, f"{count}\n({percentage:.1f}%)", ha="center", va="center", fontsize=9)
        left += percentage

    ax.set(xlim=(0, 100), ylim=(-0.6, 0.6), yticks=[], xlabel="Families (%)")
    ax.set_title(f"{transition_label}: LINEAR/EXP directional behavior\nAgreement: {agreement}/{total} ({agreement_pct:.1f}%)   |   Disagreement: {disagreement}/{total} ({disagreement_pct:.1f}%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False)
    out_file: Path = plots_dir / f"{transition_label}_sign_direction_stacked_bar.png"
    save_close(fig, out_file); output_paths.append(out_file.as_posix())


def save_and_plot_sign_summaries(sign_df: pd.DataFrame, transition_label: str, tables_dir: Path, plots_dir: Path, output_paths: list[str]) -> None:
    plot_sign_agreement_pie(sign_df, transition_label, tables_dir, plots_dir, output_paths)
    direction_summary: pd.DataFrame = build_direction_summary(sign_df)
    save_table(direction_summary, tables_dir / f"{transition_label}_sign_direction_counts.csv", output_paths)
    plot_transition_sign_direction_bar(direction_summary, transition_label, plots_dir, output_paths)


def plot_combined_sign_direction_summary(analysis_root: Path, output_paths: list[str]) -> None:
    table_paths: dict[str, Path] = {transition: analysis_root / transition / SLOPE_ANALYSIS_SUBDIR / TABLES_SUBDIR / f"{transition}_sign_direction_counts.csv" for transition in TRANSITION_ORDER}
    if not all(path.exists() for path in table_paths.values()): return

    summaries: dict[str, pd.DataFrame] = {transition: pd.read_csv(path) for transition, path in table_paths.items()}
    combined_rows: list[dict[str, object]] = []
    for transition, summary in summaries.items():
        agreement, disagreement, total, agreement_pct, disagreement_pct = agreement_totals(summary)
        row: dict[str, object] = {"transition": transition, "total_n": total, "agreement_count": agreement, "agreement_percentage": agreement_pct, "disagreement_count": disagreement, "disagreement_percentage": disagreement_pct}
        for category in SIGN_DIRECTION_ORDER:
            category_row: pd.Series = summary.loc[summary[SIGN_DIRECTION_COL] == category].iloc[0]
            row[f"{category}_count"] = int(category_row["count"]); row[f"{category}_percentage"] = float(category_row["percentage"])
        combined_rows.append(row)
    combined_df: pd.DataFrame = pd.DataFrame(combined_rows)

    summary_root: Path = analysis_root / SLOPE_SUMMARY_DIR_NAME; tables_dir, plots_dir = summary_root / TABLES_SUBDIR, summary_root / PLOTS_SUBDIR
    ensure_dir(tables_dir); ensure_dir(plots_dir)
    summary_file: Path = tables_dir / "linear_exp_sign_direction_summary.csv"; combined_df.to_csv(summary_file, index=False)

    y: np.ndarray = np.arange(len(TRANSITION_ORDER)); left: np.ndarray = np.zeros(len(TRANSITION_ORDER)); fig, ax = plt.subplots(figsize=(12, 5.5))
    for category in SIGN_DIRECTION_ORDER:
        percentages: np.ndarray = np.array([float(combined_df.loc[combined_df["transition"] == transition, f"{category}_percentage"].iloc[0]) for transition in TRANSITION_ORDER])
        counts: np.ndarray = np.array([int(combined_df.loc[combined_df["transition"] == transition, f"{category}_count"].iloc[0]) for transition in TRANSITION_ORDER])
        ax.barh(y, percentages, left=left, height=0.62, color=SIGN_DIRECTION_COLOR_MAP[category], label=category)
        for i, (percentage, count) in enumerate(zip(percentages, counts)):
            if percentage >= 5: ax.text(left[i] + percentage / 2, i, str(count), ha="center", va="center", fontsize=9, fontweight="bold")
        left += percentages

    ax.axvline(100, color="black", linewidth=0.8, alpha=0.35)
    for i, transition in enumerate(TRANSITION_ORDER):
        row: pd.Series = combined_df.loc[combined_df["transition"] == transition].iloc[0]
        ax.text(102, i, f"Agree: {int(row['agreement_count'])}/{int(row['total_n'])} ({float(row['agreement_percentage']):.1f}%)\nDisagree: {int(row['disagreement_count'])}/{int(row['total_n'])} ({float(row['disagreement_percentage']):.1f}%)", ha="left", va="center", fontsize=9)

    ax.set_xlim(0, 132); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_yticks(y, [TRANSITION_DISPLAY_NAMES[t] for t in TRANSITION_ORDER]); ax.invert_yaxis()
    ax.set_xlabel("Families (%)"); ax.set_title("LINEAR and EXPONENTIAL models show consistent directional behavior")
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.14), ncol=2, frameon=False)
    out_file: Path = plots_dir / "linear_exp_sign_direction_summary_stacked_bar.png"; save_close(fig, out_file)
    output_paths.extend([summary_file.as_posix(), out_file.as_posix()])



def build_chosen_direction_summary(plotting_df: pd.DataFrame, linear_slope_col: str, exp_slope_col: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model, slope_col in ((LABEL_LINEAR, linear_slope_col), (LABEL_EXP, exp_slope_col)):
        signs: pd.Series = plotting_df.loc[plotting_df[OVERALL_CHOSEN_MODEL_COL] == model, slope_col].map(lambda value: infer_sign(to_float(value)))
        counts: pd.Series = get_ordered_counts(signs[signs.isin(VALID_DIRECTION_LABELS)], order=VALID_DIRECTION_LABELS)
        total: int = int(counts.sum())
        for direction in VALID_DIRECTION_LABELS:
            count: int = int(counts[direction])
            rows.append({"model": model, "direction": direction, "count": count, "percentage": round(100 * count / total, 2) if total else 0.0, "total_n": total})
    return pd.DataFrame(rows)


def plot_transition_chosen_direction_bar(summary: pd.DataFrame, transition_label: str, plots_dir: Path, output_paths: list[str]) -> None:
    models: tuple[str, str] = (LABEL_LINEAR, LABEL_EXP)
    y: np.ndarray = np.arange(len(models)); left: np.ndarray = np.zeros(len(models)); fig, ax = plt.subplots(figsize=(8, 4.2))

    for direction in VALID_DIRECTION_LABELS:
        values: np.ndarray = np.array([float(summary.loc[(summary["model"] == model) & (summary["direction"] == direction), "percentage"].iloc[0]) for model in models])
        counts: np.ndarray = np.array([int(summary.loc[(summary["model"] == model) & (summary["direction"] == direction), "count"].iloc[0]) for model in models])
        ax.barh(y, values, left=left, height=0.58, color=SLOPE_SIGN_COLOR_MAP[direction], label=direction)
        for i, (value, count) in enumerate(zip(values, counts)):
            if value >= 7: ax.text(left[i] + value / 2, i, f"{count}\n({value:.1f}%)", ha="center", va="center", fontsize=9)
        left += values

    labels: list[str] = []
    for model in models:
        total: int = int(summary.loc[summary["model"] == model, "total_n"].iloc[0])
        labels.append(f"{'Linear' if model == LABEL_LINEAR else 'Exponential'} (n={total})")
    ax.set(xlim=(0, 100), yticks=y, yticklabels=labels, xlabel="Families (%)"); ax.invert_yaxis()
    ax.set_title(f"{TRANSITION_DISPLAY_NAMES.get(transition_label, transition_label.title())}: direction of overall-chosen dependent models")
    ax.legend(title="Direction", loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    out_file: Path = plots_dir / f"{transition_label}_overall_chosen_direction_stacked_bar.png"; save_close(fig, out_file); output_paths.append(out_file.as_posix())



def plot_chosen_model_direction_pie(
    summary: pd.DataFrame,
    transition_label: str,
    model_label: str,
    plots_dir: Path,
    output_paths: list[str],
) -> None:
    """Plot positive/negative slope direction for one overall-chosen model."""
    model_summary: pd.DataFrame = summary[summary["model"] == model_label].copy()
    counts: pd.Series = model_summary.set_index("direction")["count"].reindex(VALID_DIRECTION_LABELS, fill_value=0)
    counts = counts[counts > 0]
    if counts.empty: return

    model_name: str = "LINEAR" if model_label == LABEL_LINEAR else "EXP"
    out_file: Path = plots_dir / f"{transition_label}_{model_label}_chosen_slope_sign_pie.png"
    plot_pie_counts(
        counts=counts,
        title=f"{transition_label}: slope direction of {model_name}-chosen families",
        out_file=out_file,
        color_map=SLOPE_SIGN_COLOR_MAP,
        show_counts=True,
        total_n=int(model_summary["total_n"].iloc[0]),
    )
    output_paths.append(out_file.as_posix())

def save_and_plot_chosen_direction_summary(
    plotting_df: pd.DataFrame,
    transition_label: str,
    linear_slope_col: str,
    exp_slope_col: str,
    tables_dir: Path,
    plots_dir: Path,
    output_paths: list[str],
) -> None:
    summary: pd.DataFrame = build_chosen_direction_summary(plotting_df, linear_slope_col, exp_slope_col)
    save_table(summary, tables_dir / f"{transition_label}_overall_chosen_direction_counts.csv", output_paths)
    plot_transition_chosen_direction_bar(summary, transition_label, plots_dir, output_paths)
    plot_chosen_model_direction_pie(summary, transition_label, LABEL_EXP, plots_dir, output_paths)


def plot_combined_chosen_direction_summary(analysis_root: Path, output_paths: list[str]) -> None:
    table_paths: dict[str, Path] = {transition: analysis_root / transition / SLOPE_ANALYSIS_SUBDIR / TABLES_SUBDIR / f"{transition}_overall_chosen_direction_counts.csv" for transition in TRANSITION_ORDER}
    if not all(path.exists() for path in table_paths.values()): return

    frames: list[pd.DataFrame] = []
    for transition, path in table_paths.items():
        table: pd.DataFrame = pd.read_csv(path); table.insert(0, "transition", transition); frames.append(table)
    combined: pd.DataFrame = pd.concat(frames, ignore_index=True)

    summary_root: Path = analysis_root / SLOPE_SUMMARY_DIR_NAME; tables_dir, plots_dir = summary_root / TABLES_SUBDIR, summary_root / PLOTS_SUBDIR
    ensure_dir(tables_dir); ensure_dir(plots_dir)
    summary_file: Path = tables_dir / "overall_chosen_direction_summary.csv"; combined.to_csv(summary_file, index=False)

    row_specs: list[tuple[str, str]] = [(transition, model) for transition in TRANSITION_ORDER for model in (LABEL_LINEAR, LABEL_EXP)]
    y: np.ndarray = np.arange(len(row_specs)); left: np.ndarray = np.zeros(len(row_specs)); fig, ax = plt.subplots(figsize=(9.5, 6.2))
    for direction in VALID_DIRECTION_LABELS:
        values: np.ndarray = np.array([float(combined.loc[(combined["transition"] == transition) & (combined["model"] == model) & (combined["direction"] == direction), "percentage"].iloc[0]) for transition, model in row_specs])
        counts: np.ndarray = np.array([int(combined.loc[(combined["transition"] == transition) & (combined["model"] == model) & (combined["direction"] == direction), "count"].iloc[0]) for transition, model in row_specs])
        ax.barh(y, values, left=left, height=0.58, color=SLOPE_SIGN_COLOR_MAP[direction], label=direction)
        for i, (value, count) in enumerate(zip(values, counts)):
            if value >= 7: ax.text(left[i] + value / 2, i, f"{count}\n({value:.0f}%)", ha="center", va="center", fontsize=8)
        left += values

    labels: list[str] = []
    for transition, model in row_specs:
        total: int = int(combined.loc[(combined["transition"] == transition) & (combined["model"] == model), "total_n"].iloc[0])
        labels.append(f"{TRANSITION_DISPLAY_NAMES[transition]} — {'Linear' if model == LABEL_LINEAR else 'Exponential'} (n={total})")
    ax.set(xlim=(0, 100), yticks=y, yticklabels=labels, xlabel="Families (%)"); ax.invert_yaxis()
    ax.set_title("Direction of overall-chosen chromosome-number-dependent models")
    ax.legend(title="Direction", loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False)
    out_file: Path = plots_dir / "overall_chosen_direction_summary_stacked_bar.png"; save_close(fig, out_file)
    output_paths.extend([summary_file.as_posix(), out_file.as_posix()])

def plot_effective_slope_scatter(plotting_df: pd.DataFrame, transition_label: str, linear_slope_col: str, exp_slope_col: str, parameter_scale: str, plots_dir: Path, output_paths: list[str]) -> None:
    plot_df: pd.DataFrame = plotting_df.copy()
    plot_df[LINEAR_EFFECTIVE_SLOPE_COL] = pd.to_numeric(plot_df[linear_slope_col], errors="coerce")
    plot_df["_exp_slope"] = pd.to_numeric(plot_df[exp_slope_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[LINEAR_EFFECTIVE_SLOPE_COL, "_exp_slope"])
    if plot_df.empty: return

    fig, ax = plt.subplots(figsize=(8, 7))
    for model_label in MODEL_LABELS_WITH_EXP:
        model_df: pd.DataFrame = plot_df[plot_df[OVERALL_CHOSEN_MODEL_COL] == model_label]
        if not model_df.empty: ax.scatter(model_df[LINEAR_EFFECTIVE_SLOPE_COL], model_df["_exp_slope"], label=model_label, alpha=0.75, color=MODEL_COLOR_MAP.get(model_label))

    min_axis: float = min(plot_df[LINEAR_EFFECTIVE_SLOPE_COL].min(), plot_df["_exp_slope"].min()); max_axis: float = max(plot_df[LINEAR_EFFECTIVE_SLOPE_COL].max(), plot_df["_exp_slope"].max())
    padding: float = 0.05 * (max_axis - min_axis) if max_axis > min_axis else 1.0; min_axis -= padding; max_axis += padding
    ax.axvline(0, color="black", linewidth=1, linestyle="--"); ax.axhline(0, color="black", linewidth=1, linestyle="--"); ax.plot([min_axis, max_axis], [min_axis, max_axis], color="black", linewidth=1, linestyle=":")
    ax.set(xlim=(min_axis, max_axis), ylim=(min_axis, max_axis), xlabel=("Tree-scale-corrected LINEAR effective slope" if parameter_scale == CORRECTED_PARAMETER_SCALE else "Raw LINEAR effective slope"), ylabel=("Tree-scale-corrected EXP effective slope" if parameter_scale == CORRECTED_PARAMETER_SCALE else "Raw EXP effective slope"))
    ax.set_title(f"{transition_label}: LINEAR vs EXP effective slopes ({parameter_scale})"); ax.legend(title="Overall chosen model")
    out_file: Path = plots_dir / f"{transition_label}_effective_slope_scatter.png"; save_close(fig, out_file); output_paths.append(out_file.as_posix())


def plot_histogram(values: pd.Series, title: str, xlabel: str, color: str, out_file: Path, output_paths: list[str]) -> None:
    clean: pd.Series = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty: return
    median: float = float(clean.median()); fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(clean, bins=30, color=color, alpha=0.8); ax.axvline(0, color="black", linestyle="--", linewidth=1, label="Zero"); ax.axvline(median, color="black", linewidth=1.8, label=f"Median = {median:.3g}")
    ax.set(title=f"{title} (n={len(clean)})", xlabel=xlabel, ylabel="Number of families"); ax.legend(); save_close(fig, out_file); output_paths.append(out_file.as_posix())


def plot_slope_histograms(plotting_df: pd.DataFrame, transition_label: str, linear_slope_col: str, exp_slope_col: str, parameter_scale: str, plots_dir: Path, output_paths: list[str]) -> None:
    scale_label: str = "Tree-scale-corrected" if parameter_scale == CORRECTED_PARAMETER_SCALE else "Raw"
    exp_df: pd.DataFrame = plotting_df[plotting_df[OVERALL_CHOSEN_MODEL_COL] == LABEL_EXP]; linear_df: pd.DataFrame = plotting_df[plotting_df[OVERALL_CHOSEN_MODEL_COL] == LABEL_LINEAR]
    plot_histogram(exp_df[exp_slope_col], f"{transition_label}: EXP effective slopes (EXP-chosen families, {parameter_scale})", f"{scale_label} EXP effective slope", MODEL_COLOR_MAP[LABEL_EXP], plots_dir / f"{transition_label}_exp_chosen_effective_slope_histogram.png", output_paths)
    plot_histogram(linear_df[linear_slope_col], f"{transition_label}: LINEAR slopes (LINEAR-chosen families, {parameter_scale})", f"{scale_label} LINEAR slope", MODEL_COLOR_MAP[LABEL_LINEAR], plots_dir / f"{transition_label}_linear_chosen_slope_histogram.png", output_paths)


def common_histogram_bins(*series: pd.Series, bins: int = 30) -> np.ndarray:
    parts: list[pd.Series] = [pd.to_numeric(values, errors="coerce").dropna() for values in series]
    parts = [values for values in parts if not values.empty]
    if not parts: raise ValueError("Cannot build histogram bins from empty slope data.")
    return np.histogram_bin_edges(pd.concat(parts, ignore_index=True).to_numpy(dtype=float), bins=bins)


def plot_overlay_histogram(plotting_df: pd.DataFrame, transition_label: str, linear_slope_col: str, exp_slope_col: str, parameter_scale: str, selection: OverlaySelection, plots_dir: Path, output_paths: list[str]) -> None:
    if selection == "beats_constant":
        linear_mask: pd.Series = plotting_df[LINEAR_AICC_COL] < plotting_df[CONSTANT_AICC_COL]; exp_mask: pd.Series = plotting_df[EXP_AICC_COL] < plotting_df[CONSTANT_AICC_COL]
        linear_label, exp_label = "LINEAR beats constant", "EXP beats constant"
        note: str = f"Only LINEAR beats constant: n={int((linear_mask & ~exp_mask).sum())}\nOnly EXP beats constant: n={int((exp_mask & ~linear_mask).sum())}\nBoth beat constant: n={int((linear_mask & exp_mask).sum())}"
        title: str = f"{transition_label}: slope distributions among models that beat constant ({parameter_scale})"
        filename: str = f"{transition_label}_linear_exp_beats_constant_slope_overlay_histogram.png"
    else:
        linear_mask = plotting_df[OVERALL_CHOSEN_MODEL_COL] == LABEL_LINEAR; exp_mask = plotting_df[OVERALL_CHOSEN_MODEL_COL] == LABEL_EXP
        linear_label, exp_label = "LINEAR chosen overall", "EXP chosen overall"
        note = f"LINEAR chosen overall: n={int(linear_mask.sum())}\nEXP chosen overall: n={int(exp_mask.sum())}"
        title = f"{transition_label}: slope distributions of overall chosen models ({parameter_scale})"
        filename = f"{transition_label}_linear_exp_overall_chosen_slope_overlay_histogram.png"

    linear_slopes: pd.Series = pd.to_numeric(plotting_df.loc[linear_mask, linear_slope_col], errors="coerce").dropna(); exp_slopes: pd.Series = pd.to_numeric(plotting_df.loc[exp_mask, exp_slope_col], errors="coerce").dropna()
    if linear_slopes.empty and exp_slopes.empty: return
    bins: np.ndarray = common_histogram_bins(linear_slopes, exp_slopes); fig, ax = plt.subplots(figsize=(8, 6))
    if not linear_slopes.empty:
        median: float = float(linear_slopes.median()); ax.hist(linear_slopes, bins=bins, color=MODEL_COLOR_MAP[LABEL_LINEAR], alpha=0.45, label=f"{linear_label} (n={len(linear_slopes)})"); ax.axvline(median, color=MODEL_COLOR_MAP[LABEL_LINEAR], linewidth=1.8, label=f"LINEAR median = {median:.3g}")
    if not exp_slopes.empty:
        median = float(exp_slopes.median()); ax.hist(exp_slopes, bins=bins, color=MODEL_COLOR_MAP[LABEL_EXP], alpha=0.45, label=f"{exp_label} (n={len(exp_slopes)})"); ax.axvline(median, color=MODEL_COLOR_MAP[LABEL_EXP], linewidth=1.8, label=f"EXP median = {median:.3g}")
    ax.axvline(0, color="black", linestyle="--", linewidth=1, label="Zero"); ax.text(0.98, 0.98, note, transform=ax.transAxes, ha="right", va="top", fontsize=9, bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.9))
    ax.set(title=title, xlabel=("Tree-scale-corrected effective slope" if parameter_scale == CORRECTED_PARAMETER_SCALE else "Raw effective slope"), ylabel="Number of families"); ax.legend(loc="upper left")
    out_file: Path = plots_dir / filename; save_close(fig, out_file); output_paths.append(out_file.as_posix())


def run_slope_analysis(transition_label: str, core_table_path: Path, features_summary_table_path: Path, baseline_model_summary_table_path: Path, chosen_model_file_path: Path, parameter_scale: str, output_dir: Path | None = None) -> list[str]:
    output_dir = output_dir or ANALYSIS_DIR / EXPONENTIAL_VS_LINEAR_SUBDIR / transition_label / SLOPE_ANALYSIS_SUBDIR
    tables_dir, plots_dir = output_dir / TABLES_SUBDIR, output_dir / PLOTS_SUBDIR
    for directory in (output_dir, tables_dir, plots_dir): ensure_dir(directory)

    core_df, features_df = pd.read_csv(core_table_path), pd.read_csv(features_summary_table_path)
    baseline_df, chosen_df = pd.read_csv(baseline_model_summary_table_path), pd.read_csv(chosen_model_file_path)
    linear_slope_col, exp_slope_col = resolve_slope_columns(parameter_scale); output_paths: list[str] = []

    plotting_df: pd.DataFrame = build_plotting_table(core_df, features_df, baseline_df, chosen_df, parameter_scale)
    save_table(plotting_df, tables_dir / f"{transition_label}_exp_lin_slope_plotting_table.csv", output_paths)

    sign_df: pd.DataFrame = build_sign_comparison_table(plotting_df, linear_slope_col)
    save_and_plot_sign_summaries(sign_df, transition_label, tables_dir, plots_dir, output_paths)
    save_and_plot_chosen_direction_summary(plotting_df, transition_label, linear_slope_col, exp_slope_col, tables_dir, plots_dir, output_paths)
    plot_effective_slope_scatter(plotting_df, transition_label, linear_slope_col, exp_slope_col, parameter_scale, plots_dir, output_paths)
    plot_slope_histograms(plotting_df, transition_label, linear_slope_col, exp_slope_col, parameter_scale, plots_dir, output_paths)
    plot_overlay_histogram(plotting_df, transition_label, linear_slope_col, exp_slope_col, parameter_scale, "beats_constant", plots_dir, output_paths)
    plot_overlay_histogram(plotting_df, transition_label, linear_slope_col, exp_slope_col, parameter_scale, "chosen", plots_dir, output_paths)

    if output_dir.name == SLOPE_ANALYSIS_SUBDIR and output_dir.parent.name == transition_label:
        analysis_root: Path = output_dir.parent.parent
        plot_combined_sign_direction_summary(analysis_root, output_paths)
        plot_combined_chosen_direction_summary(analysis_root, output_paths)
    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze EXP versus LINEAR slopes.")
    parser.add_argument("--transition-label", "--transition_label", dest="transition_label", required=True)
    parser.add_argument("--core-table-path", "--core_table_path", dest="core_table_path", type=Path, required=True)
    parser.add_argument("--features-summary-table-path", "--features_summary_table_path", dest="features_summary_table_path", type=Path, required=True)
    parser.add_argument("--baseline-model-summary-table-path", "--baseline_model_summary_table_path", dest="baseline_model_summary_table_path", type=Path, required=True)
    parser.add_argument("--chosen-model-file-path", "--chosen_model_file_path", dest="chosen_model_file_path", type=Path, required=True)
    parser.add_argument("--parameter-scale", required=True, choices=VALID_PARAMETER_SCALES, help="Use raw or tree-scale-corrected slope magnitudes.")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args: argparse.Namespace = parse_args(); linear_slope_col, exp_slope_col = resolve_slope_columns(args.parameter_scale)
    output_paths: list[str] = run_slope_analysis(args.transition_label, args.core_table_path, args.features_summary_table_path, args.baseline_model_summary_table_path, args.chosen_model_file_path, args.parameter_scale, args.output_dir)

    log_run(
        step="exp_linear_slope_analysis",
        script=Path(__file__),
        params={"transition_label": args.transition_label, "parameter_scale": args.parameter_scale, "linear_slope_column": linear_slope_col, "exp_effective_slope_column": exp_slope_col, "core_table_path": args.core_table_path, "features_summary_table_path": args.features_summary_table_path, "baseline_model_summary_table_path": args.baseline_model_summary_table_path, "chosen_model_file_path": args.chosen_model_file_path, "output_dir": args.output_dir},
        outputs=output_paths,
        description="Analyzed LINEAR and EXPONENTIAL slopes under the integrated CONSTANT-LINEAR-EXPONENTIAL-IGNORE comparison.",
        notes=(f"Slope scale: {args.parameter_scale}. Corrected analysis uses corrected LINEAR p2 and recomputed EXP effective slope from corrected EXP p1; EXP p2 remains unscaled. Sign summaries include only families where both LINEAR and EXP beat CONSTANT and both directions are valid and non-zero. Outputs include the agreement pie, transition-specific LINEAR/EXP sign-combination bar, positive/negative bars for overall-chosen LINEAR and EXP families, an EXP-chosen slope-sign pie, and combined summaries once all three transitions exist."),
    )

    print("[✓] EXP-LINEAR slope analysis completed.")
    print(f"Parameter scale: {args.parameter_scale}")
    print(f"LINEAR slope column: {linear_slope_col}")
    print(f"EXP effective slope column: {exp_slope_col}")
    for output_path in output_paths: print(output_path)


if __name__ == "__main__":
    main()
