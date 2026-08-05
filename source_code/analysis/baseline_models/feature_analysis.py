#!/usr/bin/env python3
"""
Generic H0-vs-H1 feature analysis for ChromEvol.

Only families whose global winner is H0 or H1 are analyzed; other winners
(e.g. ignore) are excluded. Metadata features are compared between H0- and
H1-winning families; inferred features are compared pairwise between H0/H1
runs.

Threshold scans:
  - at/above: retain value >= threshold; compare %H1, strong-H1 support and data retained, and mark an empirical candidate cutoff.
  - at/below: retain value <= threshold; mark saturation of %H1.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind, ttest_rel

from source_code import constants as core
from source_code.analysis import analysis_constants as ac
from source_code.logger import log_run


StatsRow = dict[str, Any]
RunResult = dict[str, Any]


@dataclass(frozen=True)
class Comparison:
    transition: str
    h0: str
    h1: str
    input_dir: Path
    output_dir: Path

    @property
    def label(self) -> str:
        return f"{self.h0}_vs_{self.h1}"

    def feature_file(self, model: str) -> Path:
        return self.input_dir / f"{self.transition}_{model}_{ac.FEATURES_SUMMARY_SUFFIX}"

    @property
    def chosen_file(self) -> Path:
        return self.input_dir / f"{self.transition}_{ac.CHOSEN_MODEL_SUFFIX}"

    @property
    def summary_file(self) -> Path:
        return self.input_dir / f"{self.transition}_{ac.MODELS_SUMMARY_SUFFIX}"


def require_columns(df: pd.DataFrame, cols: list[str], path: Path) -> None:
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")


def load_inputs(cmp: Comparison) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    h0_file, h1_file = cmp.feature_file(cmp.h0), cmp.feature_file(cmp.h1)
    for path in [h0_file, h1_file, cmp.chosen_file, cmp.summary_file]:
        if not path.exists():
            raise FileNotFoundError(path)

    h0, h1 = pd.read_csv(h0_file), pd.read_csv(h1_file)
    chosen, summary = pd.read_csv(cmp.chosen_file), pd.read_csv(cmp.summary_file)

    feature_cols = [core.FAMILY_NAME_COL, ac.FUNCTION_LABEL_COL, *ac.FEATURE_ANALYSIS_COLS]
    require_columns(h0, feature_cols, h0_file)
    require_columns(h1, feature_cols, h1_file)
    require_columns(chosen, [core.FAMILY_NAME_COL, ac.CHOSEN_FUNCTION_LABEL_COL], cmp.chosen_file)
    require_columns(summary, [core.FAMILY_NAME_COL, core.LABEL_FUNC_TYPE_COL, core.AICC_COL], cmp.summary_file)

    for df, label, path in [(h0, cmp.h0, h0_file), (h1, cmp.h1, h1_file)]:
        if df[core.FAMILY_NAME_COL].duplicated().any():
            raise ValueError(f"Duplicate families in {path}")
        labels = set(df[ac.FUNCTION_LABEL_COL].dropna().astype(str).unique())
        if labels != {label}:
            raise ValueError(f"{path} contains function labels {sorted(labels)}")

    if chosen[core.FAMILY_NAME_COL].duplicated().any():
        raise ValueError(f"Duplicate families in {cmp.chosen_file}")
    if summary.duplicated([core.FAMILY_NAME_COL, core.LABEL_FUNC_TYPE_COL]).any():
        raise ValueError(f"Duplicate family/model rows in {cmp.summary_file}")

    return h0, h1, chosen, summary


def pairwise_metrics(summary: pd.DataFrame, cmp: Comparison) -> pd.DataFrame:
    cols = [core.FAMILY_NAME_COL, core.LABEL_FUNC_TYPE_COL, core.AICC_COL]
    df = summary[summary[core.LABEL_FUNC_TYPE_COL].isin([cmp.h0, cmp.h1])][cols].copy()
    df[core.AICC_COL] = pd.to_numeric(df[core.AICC_COL], errors="coerce")
    wide = df.dropna(subset=[core.AICC_COL]).pivot(
        index=core.FAMILY_NAME_COL, columns=core.LABEL_FUNC_TYPE_COL, values=core.AICC_COL
    )

    if cmp.h0 not in wide.columns or cmp.h1 not in wide.columns:
        raise ValueError(f"Model summary must contain both '{cmp.h0}' and '{cmp.h1}'")

    wide = wide.dropna(subset=[cmp.h0, cmp.h1]).reset_index()
    wide.columns.name = None
    wide[ac.H0_AICC_COL], wide[ac.H1_AICC_COL] = wide[cmp.h0], wide[cmp.h1]
    wide[ac.SIGNED_DELTA_AICC_H0_MINUS_H1_COL] = wide[ac.H0_AICC_COL] - wide[ac.H1_AICC_COL]
    wide[ac.PAIRWISE_DELTA_AICC_COL] = wide[ac.SIGNED_DELTA_AICC_H0_MINUS_H1_COL].abs()

    best = wide[[ac.H0_AICC_COL, ac.H1_AICC_COL]].min(axis=1)
    rel_h0, rel_h1 = np.exp(-0.5 * (wide[ac.H0_AICC_COL] - best)), np.exp(-0.5 * (wide[ac.H1_AICC_COL] - best))
    wide[ac.H0_PAIRWISE_AKAIKE_WEIGHT_COL], wide[ac.H1_PAIRWISE_AKAIKE_WEIGHT_COL] = rel_h0 / (rel_h0 + rel_h1), rel_h1 / (rel_h0 + rel_h1)
    wide[ac.PAIRWISE_BEST_AKAIKE_WEIGHT_COL] = wide[[ac.H0_PAIRWISE_AKAIKE_WEIGHT_COL, ac.H1_PAIRWISE_AKAIKE_WEIGHT_COL]].max(axis=1)

    return wide[[core.FAMILY_NAME_COL, ac.H0_AICC_COL, ac.H1_AICC_COL, ac.SIGNED_DELTA_AICC_H0_MINUS_H1_COL, ac.PAIRWISE_DELTA_AICC_COL, ac.H0_PAIRWISE_AKAIKE_WEIGHT_COL, ac.H1_PAIRWISE_AKAIKE_WEIGHT_COL, ac.PAIRWISE_BEST_AKAIKE_WEIGHT_COL]]


def prepare_analysis_tables(
    h0: pd.DataFrame, h1: pd.DataFrame, chosen: pd.DataFrame, pairwise: pd.DataFrame, cmp: Comparison
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    population = chosen[[core.FAMILY_NAME_COL, ac.CHOSEN_FUNCTION_LABEL_COL]].merge(
        pairwise, on=core.FAMILY_NAME_COL, how="left", validate="one_to_one"
    )
    h0_families, h1_families = set(h0[core.FAMILY_NAME_COL]), set(h1[core.FAMILY_NAME_COL])

    population["included_in_analysis"] = (
        population[ac.CHOSEN_FUNCTION_LABEL_COL].isin([cmp.h0, cmp.h1])
        & population[[ac.H0_AICC_COL, ac.H1_AICC_COL]].notna().all(axis=1)
        & population[core.FAMILY_NAME_COL].isin(h0_families)
        & population[core.FAMILY_NAME_COL].isin(h1_families)
    )

    def exclusion_reason(row: pd.Series) -> str:
        if row["included_in_analysis"]:
            return ""
        reasons: list[str] = []
        if row[ac.CHOSEN_FUNCTION_LABEL_COL] not in {cmp.h0, cmp.h1}:
            reasons.append(f"other_model_chosen:{row[ac.CHOSEN_FUNCTION_LABEL_COL]}")
        if pd.isna(row[ac.H0_AICC_COL]) or pd.isna(row[ac.H1_AICC_COL]):
            reasons.append("missing_h0_or_h1_aicc")
        if row[core.FAMILY_NAME_COL] not in h0_families:
            reasons.append("missing_h0_feature_row")
        if row[core.FAMILY_NAME_COL] not in h1_families:
            reasons.append("missing_h1_feature_row")
        return ";".join(reasons)

    population["exclusion_reason"] = population.apply(exclusion_reason, axis=1)

    eligible_cols = [
        core.FAMILY_NAME_COL, ac.CHOSEN_FUNCTION_LABEL_COL, ac.H0_AICC_COL, ac.H1_AICC_COL,
        ac.SIGNED_DELTA_AICC_H0_MINUS_H1_COL, ac.PAIRWISE_DELTA_AICC_COL, ac.H0_PAIRWISE_AKAIKE_WEIGHT_COL, ac.H1_PAIRWISE_AKAIKE_WEIGHT_COL, ac.PAIRWISE_BEST_AKAIKE_WEIGHT_COL,
    ]
    eligible = population.loc[population["included_in_analysis"], eligible_cols]

    chosen_features = pd.concat([h0, h1], ignore_index=True).merge(
        eligible, on=core.FAMILY_NAME_COL, how="inner", validate="many_to_one"
    )
    chosen_features = chosen_features[
        chosen_features[ac.FUNCTION_LABEL_COL] == chosen_features[ac.CHOSEN_FUNCTION_LABEL_COL]
    ].copy()

    eligible_families = set(eligible[core.FAMILY_NAME_COL])
    paired_h0 = h0[h0[core.FAMILY_NAME_COL].isin(eligible_families)][
        [core.FAMILY_NAME_COL, *ac.FEATURE_ANALYSIS_INFERRED_COLS]
    ].rename(columns={feature: f"{feature}__h0" for feature in ac.FEATURE_ANALYSIS_INFERRED_COLS})
    paired_h1 = h1[h1[core.FAMILY_NAME_COL].isin(eligible_families)][
        [core.FAMILY_NAME_COL, *ac.FEATURE_ANALYSIS_INFERRED_COLS]
    ].rename(columns={feature: f"{feature}__h1" for feature in ac.FEATURE_ANALYSIS_INFERRED_COLS})
    paired = paired_h0.merge(paired_h1, on=core.FAMILY_NAME_COL, validate="one_to_one")

    return population, chosen_features, paired


def groups(
    feature: str, chosen_features: pd.DataFrame, paired: pd.DataFrame, cmp: Comparison
) -> tuple[pd.Series, pd.Series]:
    if feature in ac.FEATURE_ANALYSIS_INFERRED_COLS:
        df = paired[[f"{feature}__h0", f"{feature}__h1"]].apply(pd.to_numeric, errors="coerce").dropna()
        return df.iloc[:, 0], df.iloc[:, 1]

    df = chosen_features[[feature, ac.CHOSEN_FUNCTION_LABEL_COL]].copy()
    df[feature] = pd.to_numeric(df[feature], errors="coerce")
    df = df.dropna()
    h0 = df.loc[df[ac.CHOSEN_FUNCTION_LABEL_COL] == cmp.h0, feature]
    h1 = df.loc[df[ac.CHOSEN_FUNCTION_LABEL_COL] == cmp.h1, feature]
    return h0, h1


def effect_size(h1: pd.Series, h0: pd.Series, paired: bool) -> float:
    if len(h0) < 2 or len(h1) < 2:
        return np.nan
    if paired:
        diff, sd = h1 - h0, (h1 - h0).std(ddof=1)
        return np.nan if pd.isna(sd) or sd == 0 else float(diff.mean() / sd)

    pooled = ((len(h1) - 1) * h1.var(ddof=1) + (len(h0) - 1) * h0.var(ddof=1)) / (len(h1) + len(h0) - 2)
    return np.nan if pd.isna(pooled) or pooled <= 0 else float((h1.mean() - h0.mean()) / np.sqrt(pooled))


def feature_stats(
    feature: str, chosen_features: pd.DataFrame, paired_df: pd.DataFrame, cmp: Comparison
) -> StatsRow:
    h0, h1 = groups(feature, chosen_features, paired_df, cmp)
    is_paired = feature in ac.FEATURE_ANALYSIS_INFERRED_COLS
    mean_diff = (
        (h1 - h0).mean() if is_paired and len(h0)
        else h1.mean() - h0.mean() if len(h0) and len(h1)
        else np.nan
    )

    row: StatsRow = {
        "feature": feature, "comparison_type": "paired_inferred" if is_paired else "unpaired_metadata",
        "h0_model": cmp.h0, "h1_model": cmp.h1, "n_h0": len(h0), "n_h1": len(h1),
        "n_pairs": len(h0) if is_paired else np.nan,
        "mean_h0": h0.mean() if len(h0) else np.nan, "mean_h1": h1.mean() if len(h1) else np.nan,
        "mean_diff_h1_minus_h0": mean_diff, "effect_size_h1_minus_h0": effect_size(h1, h0, is_paired),
        "paired_t_stat": np.nan, "paired_p_value": np.nan, "welch_t_stat": np.nan, "welch_p_value": np.nan,
        "mannwhitney_u": np.nan, "mannwhitney_p_value": np.nan,
    }

    if not len(h0) or not len(h1):
        return row
    if is_paired:
        row["paired_t_stat"], row["paired_p_value"] = ttest_rel(h1, h0, nan_policy="omit")
    else:
        row["welch_t_stat"], row["welch_p_value"] = ttest_ind(h1, h0, equal_var=False, nan_policy="omit")
        row["mannwhitney_u"], row["mannwhitney_p_value"] = mannwhitneyu(h1, h0, alternative="two-sided")
    return row


def stats_text(row: StatsRow) -> list[str]:
    if row["comparison_type"] == "paired_inferred":
        return [
            f"n pairs = {row['n_pairs']}", f"mean H0 = {row['mean_h0']:.2f}", f"mean H1 = {row['mean_h1']:.2f}",
            f"mean H1-H0 = {row['mean_diff_h1_minus_h0']:.2f}", f"paired p = {row['paired_p_value']:.3g}",
            f"dz = {row['effect_size_h1_minus_h0']:.2f}",
        ]
    return [
        f"n H0 = {row['n_h0']}", f"n H1 = {row['n_h1']}", f"mean H0 = {row['mean_h0']:.2f}",
        f"mean H1 = {row['mean_h1']:.2f}", f"Welch p = {row['welch_p_value']:.3g}",
        f"MWU p = {row['mannwhitney_p_value']:.3g}", f"d = {row['effect_size_h1_minus_h0']:.2f}",
    ]


def add_text_box(ax: plt.Axes, lines: list[str]) -> None:
    ax.text(
        0.98, 0.98, "\n".join(lines), transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )


def model_colors(cmp: Comparison) -> dict[str, str]:
    return {
        cmp.h0: ac.MODEL_COLOR_MAP.get(cmp.h0, "#4C72B0"),
        cmp.h1: ac.MODEL_COLOR_MAP.get(cmp.h1, "#DD8452"),
    }


def plot_box_and_hist(
    feature: str, h0: pd.Series, h1: pd.Series, row: StatsRow, cmp: Comparison, out_dir: Path
) -> list[Path]:
    colors, outputs = model_colors(cmp), []

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.boxplot([h0, h1], tick_labels=[cmp.h0, cmp.h1])
    ax.set(title=f"{feature}: H0 vs H1", ylabel=feature)
    add_text_box(ax, stats_text(row))
    path = out_dir / f"{feature}_boxplot.png"
    fig.tight_layout(); fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig); outputs.append(path)

    pooled = pd.concat([h0, h1], ignore_index=True)
    bins = 20 if pooled.nunique() > 20 else min(10, max(3, pooled.nunique()))
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(h0, bins=bins, alpha=0.5, edgecolor="black", label=cmp.h0, color=colors[cmp.h0])
    ax.hist(h1, bins=bins, alpha=0.5, edgecolor="black", label=cmp.h1, color=colors[cmp.h1])
    ax.set(title=f"{feature}: H0 vs H1", xlabel=feature, ylabel="Number of families")
    ax.legend(); add_text_box(ax, stats_text(row))
    path = out_dir / f"{feature}_hist.png"
    fig.tight_layout(); fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig); outputs.append(path)
    return outputs


def plot_signed_delta(feature: str, chosen_features: pd.DataFrame, cmp: Comparison, out_dir: Path) -> Optional[Path]:
    df = chosen_features[[feature, ac.SIGNED_DELTA_AICC_H0_MINUS_H1_COL, ac.CHOSEN_FUNCTION_LABEL_COL]].copy()
    df[feature], df[ac.SIGNED_DELTA_AICC_H0_MINUS_H1_COL] = pd.to_numeric(df[feature], errors="coerce"), pd.to_numeric(df[ac.SIGNED_DELTA_AICC_H0_MINUS_H1_COL], errors="coerce")
    df = df.dropna()
    if df.empty:
        return None

    colors = model_colors(cmp)
    fig, ax = plt.subplots(figsize=(9, 6))
    for label in [cmp.h0, cmp.h1]:
        part = df[df[ac.CHOSEN_FUNCTION_LABEL_COL] == label]
        ax.scatter(part[feature], part[ac.SIGNED_DELTA_AICC_H0_MINUS_H1_COL], alpha=0.75, label=label, color=colors[label])

    ax.axhline(0, linestyle="--", color="black")
    ax.set(title=f"{feature} vs signed pairwise ΔAICc", xlabel=feature, ylabel=f"AICc({cmp.h0}) - AICc({cmp.h1})")
    ax.legend()
    path = out_dir / f"{feature}_signed_pairwise_delta_scatter.png"
    fig.tight_layout(); fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)
    return path


def threshold_grid(values: pd.Series, feature: str, direction: str) -> list[float]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return []

    if feature == ac.NUM_OF_EVENTS_COL:
        end = int(np.floor(values.max())) if direction == ac.THRESHOLD_AT_OR_ABOVE else int(np.ceil(values.max()))
        return [float(value) for value in range(1, max(1, end) + 1)]

    unique = np.sort(values.unique())
    if len(unique) <= 20:
        return [float(value) for value in unique]

    quantiles = values.quantile(np.linspace(0, 1, 21))
    return [float(value) for value in np.unique(quantiles)]


def threshold_scan(
    feature: str, chosen_features: pd.DataFrame, cmp: Comparison, direction: str, min_retained: int
) -> pd.DataFrame:
    df = chosen_features[[feature, ac.CHOSEN_FUNCTION_LABEL_COL, ac.PAIRWISE_DELTA_AICC_COL, ac.PAIRWISE_BEST_AKAIKE_WEIGHT_COL]].copy()
    for col in [feature, ac.PAIRWISE_DELTA_AICC_COL, ac.PAIRWISE_BEST_AKAIKE_WEIGHT_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[feature, ac.CHOSEN_FUNCTION_LABEL_COL])

    rows: list[dict[str, Any]] = []
    for threshold in threshold_grid(df[feature], feature, direction):
        kept = df[df[feature] >= threshold] if direction == ac.THRESHOLD_AT_OR_ABOVE else df[df[feature] <= threshold]
        if kept.empty:
            continue

        n_h0 = int((kept[ac.CHOSEN_FUNCTION_LABEL_COL] == cmp.h0).sum())
        n_h1 = int((kept[ac.CHOSEN_FUNCTION_LABEL_COL] == cmp.h1).sum())
        strong = kept[kept[ac.PAIRWISE_DELTA_AICC_COL] >= ac.STRONG_DELTA_AICC_THRESHOLD]
        n_h0_strong = int((strong[ac.CHOSEN_FUNCTION_LABEL_COL] == cmp.h0).sum())
        n_h1_strong = int((strong[ac.CHOSEN_FUNCTION_LABEL_COL] == cmp.h1).sum())

        rows.append({
            "threshold_direction": direction, "threshold": threshold,
            "threshold_rule": f"{feature} >= {threshold:g}" if direction == ac.THRESHOLD_AT_OR_ABOVE else f"{feature} <= {threshold:g}",
            "n_retained": len(kept), "pct_original_retained": 100 * len(kept) / len(df),
            "n_h0": n_h0, "n_h1": n_h1, "pct_h0": 100 * n_h0 / len(kept), "pct_h1": 100 * n_h1 / len(kept),
            "n_h0_strong_delta": n_h0_strong, "n_h1_strong_delta": n_h1_strong,
            "pct_h1_strong_delta": 100 * n_h1_strong / len(kept),
            "mean_pairwise_delta_aicc": kept[ac.PAIRWISE_DELTA_AICC_COL].mean(),
            "mean_pairwise_best_akaike_weight": kept[ac.PAIRWISE_BEST_AKAIKE_WEIGHT_COL].mean(),
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result["candidate_filter_threshold"] = False
        if direction == ac.THRESHOLD_AT_OR_ABOVE:
            candidate = candidate_filter_point(result)
            if candidate is not None:
                result.loc[result["threshold"] == candidate["threshold"], "candidate_filter_threshold"] = True
    return result


def candidate_filter_point(scan: pd.DataFrame) -> Optional[pd.Series]:
    """
    Empirical candidate cutoff for the >= scan.

    Starting from the least-filtered threshold, require:
      - a meaningful increase in %H1,
      - a meaningful increase in strongly supported H1 families,
      - little additional improvement from stricter thresholds,
      - and at least a minimum fraction of families retained.

    This is a descriptive candidate cutoff, not a simulation-validated power threshold.
    """
    if scan.empty:
        return None

    df = scan.sort_values("threshold").reset_index(drop=True)
    baseline_h1, baseline_strong = float(df.iloc[0]["pct_h1"]), float(df.iloc[0]["pct_h1_strong_delta"])

    for index, row in df.iterrows():
        if row["pct_original_retained"] < ac.CANDIDATE_MIN_RETAINED_PCT:
            continue

        tail = df.iloc[index:]
        tail = tail[tail["pct_original_retained"] >= ac.CANDIDATE_MIN_RETAINED_PCT]
        if tail.empty:
            continue

        h1_gain = float(row["pct_h1"]) - baseline_h1
        strong_gain = float(row["pct_h1_strong_delta"]) - baseline_strong
        future_h1_gain = float(tail["pct_h1"].max()) - float(row["pct_h1"])
        future_strong_gain = float(tail["pct_h1_strong_delta"].max()) - float(row["pct_h1_strong_delta"])

        if (
            h1_gain >= ac.CANDIDATE_MIN_H1_GAIN_PP
            and strong_gain >= ac.CANDIDATE_MIN_STRONG_H1_GAIN_PP
            and future_h1_gain <= ac.CANDIDATE_MAX_FUTURE_GAIN_PP
            and future_strong_gain <= ac.CANDIDATE_MAX_FUTURE_GAIN_PP
        ):
            return row
    return None


def saturation_point(scan: pd.DataFrame) -> tuple[Optional[pd.Series], float]:
    if scan.empty:
        return None, np.nan

    scan = scan.sort_values("threshold").reset_index(drop=True)
    final_pct = float(scan.iloc[-1]["pct_h1"])
    for index in range(len(scan)):
        tail = scan.iloc[index:]
        if len(tail) < ac.SATURATION_MIN_POINTS:
            break
        if (tail["pct_h1"] - final_pct).abs().max() <= ac.SATURATION_TOLERANCE_PP:
            return scan.iloc[index], final_pct
    return None, final_pct


def plot_threshold(
    scan: pd.DataFrame, feature: str, cmp: Comparison, direction: str, out_dir: Path
) -> Optional[Path]:
    if scan.empty:
        return None

    scan = scan.sort_values("threshold").reset_index(drop=True)
    color = model_colors(cmp)[cmp.h1]
    fig, ax = plt.subplots(figsize=(9, 6))

    if direction == ac.THRESHOLD_AT_OR_ABOVE:
        ax.plot(scan["threshold"], scan["pct_h1"], marker="o", color=color, label=f"% {cmp.h1}")
        ax.plot(scan["threshold"], scan["pct_h1_strong_delta"], marker="s", linestyle="--", label=f"% strong {cmp.h1}")
        ax.plot(scan["threshold"], scan["pct_original_retained"], linestyle=":", linewidth=2, label="% families retained")

        candidate = candidate_filter_point(scan)
        if candidate is not None:
            x = float(candidate["threshold"])
            ax.axvline(x, linestyle="-.", color="black", linewidth=1.3)
            ax.annotate(
                f"candidate cutoff ≈ {x:g}\n"
                f"H1 = {candidate['pct_h1']:.1f}%\n"
                f"strong H1 = {candidate['pct_h1_strong_delta']:.1f}%\n"
                f"retained = {candidate['pct_original_retained']:.1f}%",
                xy=(x, float(candidate["pct_h1"])), xytext=(12, 15), textcoords="offset points",
                arrowprops=dict(arrowstyle="->"), bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
            )
        else:
            ax.text(
                0.02, 0.05, "No candidate cutoff met all criteria", transform=ax.transAxes,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
            )

        ax.set_xlabel(f"Minimum {feature} required (value ≥ threshold)")
        ax.set_title(f"{feature}: information gain vs data retained")
        suffix = "threshold"

    else:
        ax.plot(scan["threshold"], scan["pct_h1"], marker="o", color=color, label=f"% {cmp.h1}")
        point, final_pct = saturation_point(scan)
        ax.axhline(final_pct, linestyle="--", color="black", label=f"final H1 = {final_pct:.1f}%")
        ax.axhspan(
            max(0, final_pct - ac.SATURATION_TOLERANCE_PP), min(100, final_pct + ac.SATURATION_TOLERANCE_PP),
            alpha=0.08, color=color, label=f"±{ac.SATURATION_TOLERANCE_PP:g} pp band",
        )
        if point is not None:
            x, y = float(point["threshold"]), float(point["pct_h1"])
            ax.axvline(x, linestyle=":", color="black")
            ax.scatter([x], [y], s=75, color=color, edgecolor="black", zorder=5)
            ax.annotate(
                f"saturation starts\nthreshold ≈ {x:g}\nH1 = {y:.1f}%", xy=(x, y), xytext=(10, 12),
                textcoords="offset points", arrowprops=dict(arrowstyle="->"),
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
            )
        ax.set_xlabel(f"Maximum {feature} (value ≤ threshold)")
        ax.set_title(f"{feature}: % {cmp.h1} at or below threshold")
        suffix = "at_or_below_threshold"

    ax.set_ylabel("Percent of eligible families")
    ax.set_ylim(0, 100)
    ax.legend()
    path = out_dir / f"{feature}_{suffix}_summary.png"
    fig.tight_layout(); fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)
    return path


def analyze_feature(
    feature: str, chosen_features: pd.DataFrame, paired: pd.DataFrame, cmp: Comparison, min_retained: int
) -> tuple[StatsRow, list[Path]]:
    out_dir = cmp.output_dir / feature
    out_dir.mkdir(parents=True, exist_ok=True)

    row = feature_stats(feature, chosen_features, paired, cmp)
    h0, h1 = groups(feature, chosen_features, paired, cmp)
    outputs = plot_box_and_hist(feature, h0, h1, row, cmp, out_dir)

    if feature not in ac.FEATURE_ANALYSIS_INFERRED_COLS:
        scatter = plot_signed_delta(feature, chosen_features, cmp, out_dir)
        if scatter:
            outputs.append(scatter)

    for direction, suffix in [(ac.THRESHOLD_AT_OR_ABOVE, "threshold"), (ac.THRESHOLD_AT_OR_BELOW, "at_or_below_threshold")]:
        scan = threshold_scan(feature, chosen_features, cmp, direction, min_retained)
        csv_path = out_dir / f"{feature}_{suffix}_scan.csv"
        scan.to_csv(csv_path, index=False); outputs.append(csv_path)
        plot_path = plot_threshold(scan, feature, cmp, direction, out_dir)
        if plot_path:
            outputs.append(plot_path)

    return row, outputs


def run(cmp: Comparison, min_retained: int) -> RunResult:
    cmp.output_dir.mkdir(parents=True, exist_ok=True)
    h0, h1, chosen, summary = load_inputs(cmp)
    pairwise = pairwise_metrics(summary, cmp)
    population, chosen_features, paired = prepare_analysis_tables(h0, h1, chosen, pairwise, cmp)

    population_file = cmp.output_dir / f"{cmp.transition}_{cmp.label}_analysis_population.csv"
    population.to_csv(population_file, index=False)
    outputs: list[Path] = [population_file]
    stats_rows: list[StatsRow] = []

    for feature in ac.FEATURE_ANALYSIS_COLS:
        row, feature_outputs = analyze_feature(feature, chosen_features, paired, cmp, min_retained)
        stats_rows.append(row); outputs.extend(feature_outputs)

    stats_file = cmp.output_dir / f"{cmp.transition}_{cmp.label}_feature_stats.csv"
    pd.DataFrame(stats_rows).to_csv(stats_file, index=False); outputs.append(stats_file)

    counts = chosen_features[ac.CHOSEN_FUNCTION_LABEL_COL].value_counts()
    return {
        "n_total": len(population), "n_included": int(population["included_in_analysis"].sum()),
        "n_excluded": int((~population["included_in_analysis"]).sum()),
        "n_h0": int(counts.get(cmp.h0, 0)), "n_h1": int(counts.get(cmp.h1, 0)),
        "outputs": [str(path) for path in outputs],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generic H0-vs-H1 ChromEvol feature analysis.")
    parser.add_argument("--transition", required=True, choices=list(core.LABEL_TRANSITIONS_ORDERED))
    parser.add_argument("--h0-function", required=True)
    parser.add_argument("--h1-function", required=True)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--min-threshold-families", type=int, default=ac.DEFAULT_MIN_THRESHOLD_FAMILIES)
    parser.add_argument("--skip-log", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else input_dir / ac.FEATURE_ANALYSIS_SUBDIR / f"{args.h0_function}_vs_{args.h1_function}"
    )
    cmp = Comparison(args.transition, args.h0_function, args.h1_function, input_dir, output_dir)
    result = run(cmp, args.min_threshold_families)

    print(f"[✓] Included families: {result['n_included']} / {result['n_total']}")
    print(f"[✓] Excluded families: {result['n_excluded']}")
    print(f"[✓] Global winners among included families: {cmp.h0}={result['n_h0']}, {cmp.h1}={result['n_h1']}")
    print(f"[✓] Outputs: {cmp.output_dir}")

    if not args.skip_log:
        log_run(
            step="analysis", script=Path(__file__),
            params={
                "transition": cmp.transition, "comparison": cmp.label, "h0_model": cmp.h0, "h1_model": cmp.h1,
                "input_dir": str(cmp.input_dir), "output_dir": str(cmp.output_dir),
                "n_total": result["n_total"], "n_included": result["n_included"], "n_excluded": result["n_excluded"],
                "n_h0": result["n_h0"], "n_h1": result["n_h1"],
                "min_threshold_families": args.min_threshold_families,
                "saturation_tolerance_pp": ac.SATURATION_TOLERANCE_PP, "saturation_min_points": ac.SATURATION_MIN_POINTS,
                "candidate_min_h1_gain_pp": ac.CANDIDATE_MIN_H1_GAIN_PP, "candidate_min_strong_h1_gain_pp": ac.CANDIDATE_MIN_STRONG_H1_GAIN_PP,
                "candidate_max_future_gain_pp": ac.CANDIDATE_MAX_FUTURE_GAIN_PP, "candidate_min_retained_pct": ac.CANDIDATE_MIN_RETAINED_PCT,
            },
            outputs=result["outputs"],
            description=f"Feature analysis for {cmp.transition}: {cmp.h0} vs {cmp.h1}.",
            notes=(
                "Non-H0/H1 global winners excluded. Metadata features use unpaired winner-group comparisons; "
                "inferred features use paired H0/H1 runs. >= scans mark 50% H1; <= scans mark saturation."
            ),
            log_relative_path=Path("feature_analysis") / cmp.transition / f"{cmp.label}.log",
        )


if __name__ == "__main__":
    main()