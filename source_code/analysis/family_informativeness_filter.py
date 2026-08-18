#!/usr/bin/env python3
"""
Search practical family-informativeness filters for ChromEvol analyses.

For each transition:
    S = family size
    R = chromosome-number range
    E = expected number of events inferred under a selected event model

Rules:
    S, R, E, S&R, S&E, R&E, S&R&E

Combined rules use AND elimination: a family is removed only when it fails
every feature included in that rule.

Main outputs include the complete threshold search, top candidate options,
family-level failure flags, a fixed family-size benchmark, single-feature
trade-off plots, pairwise heatmaps, S&R heatmaps at each E threshold, and
Pareto-frontier summaries.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from source_code import constants as core
from source_code.analysis import analysis_constants as ac
from source_code.analysis.baseline_models.feature_analysis import Comparison, pairwise_metrics, require_columns
from source_code.logger import log_run


FAMILY_SIZE_COL = "family_size"
CHROMOSOME_RANGE_COL = "diff_chrom"
RULE_FEATURES: dict[str, tuple[str, ...]] = {"S": ("S",), "R": ("R",), "E": ("E",), "S&R": ("S", "R"), "S&E": ("S", "E"), "R&E": ("R", "E"), "S&R&E": ("S", "R", "E")}
PATTERN_ORDER = ["none", "S", "R", "E", "S+R", "S+E", "R+E", "S+R+E"]
THRESHOLD_COLS = {"S": "size_threshold", "R": "range_threshold", "E": "events_threshold"}


def numeric_grid(values: Any, label: str) -> list[float]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"Grid entry '{label}' must be a non-empty list")
    result = sorted({float(value) for value in values})
    if not np.isfinite(result).all():
        raise ValueError(f"Grid entry '{label}' contains a non-finite value")
    return result


def load_grid(path: Path, event_model: str, transitions: list[str]) -> tuple[list[float], list[float], dict[str, list[float]], dict[str, float | int]]:
    config = json.loads(path.read_text(encoding="utf-8"))
    size_grid = numeric_grid(config.get("family_size"), "family_size")
    range_grid = numeric_grid(config.get("chromosome_range"), "chromosome_range")
    expected_events = config.get("expected_events")
    if not isinstance(expected_events, dict):
        raise ValueError("Grid JSON must contain an 'expected_events' object")
    event_section = expected_events.get(event_model, expected_events)
    if not isinstance(event_section, dict):
        raise ValueError(f"No expected-events grid found for model '{event_model}'")
    missing = [transition for transition in transitions if transition not in event_section]
    if missing:
        raise ValueError(f"Expected-events grid for model '{event_model}' is missing transitions: {missing}")
    event_grids = {transition: numeric_grid(event_section[transition], f"expected_events.{event_model}.{transition}") for transition in transitions}
    selection_config = config.get("selection", {})
    if not isinstance(selection_config, dict):
        raise ValueError("'selection' must be a JSON object")
    selection = {
        "baseline_min_family_size": float(selection_config.get("baseline_min_family_size", min(size_grid))),
        "min_retained_pct": float(selection_config.get("min_retained_pct", ac.CANDIDATE_MIN_RETAINED_PCT)),
        "min_h1_gain_pp": float(selection_config.get("min_h1_gain_pp", ac.CANDIDATE_MIN_H1_GAIN_PP)),
        "min_strong_h1_gain_pp": float(selection_config.get("min_strong_h1_gain_pp", ac.CANDIDATE_MIN_STRONG_H1_GAIN_PP)),
        "number_of_options": int(selection_config.get("number_of_options", 5)),
    }
    return size_grid, range_grid, event_grids, selection


def transition_input_dir(analysis_root: Path, transition: str, transition_input_subdir: Path) -> Path:
    base = analysis_root / transition
    return base if str(transition_input_subdir) in {"", "."} else base / transition_input_subdir


def load_transition_table(analysis_root: Path, transition_input_subdir: Path, transition: str, h0_function: str, h1_function: str, event_model: str) -> tuple[pd.DataFrame, list[Path]]:
    input_dir = transition_input_dir(analysis_root, transition, transition_input_subdir)
    comparison = Comparison(transition=transition, h0=h0_function, h1=h1_function, input_dir=input_dir, output_dir=input_dir)
    event_file, chosen_file, summary_file = comparison.feature_file(event_model), comparison.chosen_file, comparison.summary_file
    for path in [event_file, chosen_file, summary_file]:
        if not path.exists():
            raise FileNotFoundError(path)

    features, chosen, summary = pd.read_csv(event_file), pd.read_csv(chosen_file), pd.read_csv(summary_file)
    require_columns(features, [core.FAMILY_NAME_COL, ac.FUNCTION_LABEL_COL, FAMILY_SIZE_COL, CHROMOSOME_RANGE_COL, ac.NUM_OF_EVENTS_COL], event_file)
    require_columns(chosen, [core.FAMILY_NAME_COL, ac.CHOSEN_FUNCTION_LABEL_COL], chosen_file)
    require_columns(summary, [core.FAMILY_NAME_COL, core.LABEL_FUNC_TYPE_COL, core.AICC_COL], summary_file)

    if features[core.FAMILY_NAME_COL].duplicated().any():
        raise ValueError(f"Duplicate families in {event_file}")
    if chosen[core.FAMILY_NAME_COL].duplicated().any():
        raise ValueError(f"Duplicate families in {chosen_file}")

    labels = set(features[ac.FUNCTION_LABEL_COL].dropna().astype(str).unique())
    if labels != {event_model}:
        raise ValueError(f"{event_file} contains function labels {sorted(labels)}; expected only '{event_model}'")

    pairwise = pairwise_metrics(summary, comparison)
    table = features[[core.FAMILY_NAME_COL, FAMILY_SIZE_COL, CHROMOSOME_RANGE_COL, ac.NUM_OF_EVENTS_COL]].rename(columns={CHROMOSOME_RANGE_COL: "chromosome_range", ac.NUM_OF_EVENTS_COL: "expected_events"})
    table = table.merge(chosen[[core.FAMILY_NAME_COL, ac.CHOSEN_FUNCTION_LABEL_COL]], on=core.FAMILY_NAME_COL, how="inner", validate="one_to_one").merge(pairwise, on=core.FAMILY_NAME_COL, how="left", validate="one_to_one")
    for column in [FAMILY_SIZE_COL, "chromosome_range", "expected_events"]:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    return table, [event_file, chosen_file, summary_file]


def calculate_metrics(kept: pd.DataFrame, baseline_n: int, h0_function: str, h1_function: str, ignore_function: str | None, strong_delta_threshold: float) -> dict[str, float | int]:
    n_retained = len(kept)
    chosen = kept[ac.CHOSEN_FUNCTION_LABEL_COL]
    n_h0, n_h1 = int((chosen == h0_function).sum()), int((chosen == h1_function).sum())
    n_ignore = int((chosen == ignore_function).sum()) if ignore_function else 0
    n_other, n_h0_h1 = int(n_retained - n_h0 - n_h1 - n_ignore), n_h0 + n_h1
    n_strong_h1 = int(((chosen == h1_function) & (kept[ac.PAIRWISE_DELTA_AICC_COL] >= strong_delta_threshold)).fillna(False).sum())
    return {
        "n_eliminated": int(baseline_n - n_retained), "n_retained": n_retained, "pct_retained": 100 * n_retained / baseline_n if baseline_n else np.nan,
        "n_h0": n_h0, "n_h1": n_h1, "n_ignore": n_ignore, "n_other": n_other, "n_h0_h1": n_h0_h1,
        "pct_h0_selected": 100 * n_h0 / n_retained if n_retained else np.nan, "pct_h1_selected": 100 * n_h1 / n_retained if n_retained else np.nan,
        "pct_ignore_selected": 100 * n_ignore / n_retained if n_retained else np.nan, "pct_h1": 100 * n_h1 / n_h0_h1 if n_h0_h1 else np.nan,
        "pct_strong_h1": 100 * n_strong_h1 / n_h0_h1 if n_h0_h1 else np.nan, "pct_ignore": 100 * n_ignore / n_retained if n_retained else np.nan,
    }


def failure_mask(df: pd.DataFrame, rule_features: tuple[str, ...], size_threshold: float | None, range_threshold: float | None, events_threshold: float | None) -> pd.Series:
    masks = []
    if "S" in rule_features:
        masks.append(df[FAMILY_SIZE_COL].isna() | (df[FAMILY_SIZE_COL] < float(size_threshold)))
    if "R" in rule_features:
        masks.append(df["chromosome_range"].isna() | (df["chromosome_range"] < float(range_threshold)))
    if "E" in rule_features:
        masks.append(df["expected_events"].isna() | (df["expected_events"] < float(events_threshold)))
    if not masks:
        return pd.Series(False, index=df.index)
    result = masks[0].copy()
    for mask in masks[1:]:
        result &= mask
    return result


def selection_pool(group: pd.DataFrame, selection: dict[str, float | int]) -> tuple[pd.DataFrame, str, list[str], list[bool]]:
    default_sort, default_ascending = ["n_retained", "pct_strong_h1", "pct_h1", "size_threshold", "range_threshold", "events_threshold"], [False, False, False, True, True, True]
    candidates = group[group["meets_criteria"]]
    if not candidates.empty:
        return candidates, "criteria_met", default_sort, default_ascending
    positive = group[(group["pct_retained"] >= selection["min_retained_pct"]) & (group["n_eliminated"] > 0) & (group["h1_gain_pp"] >= 0) & (group["strong_h1_gain_pp"] >= 0) & ((group["h1_gain_pp"] > 0) | (group["strong_h1_gain_pp"] > 0))]
    if not positive.empty:
        return positive, "fallback_smallest_improvement", default_sort, default_ascending
    fallback = group[(group["pct_retained"] >= selection["min_retained_pct"]) & (group["n_eliminated"] > 0)]
    if fallback.empty:
        fallback = group
    return fallback, "fallback_best_available", ["pct_strong_h1", "pct_h1", "n_retained", "size_threshold", "range_threshold", "events_threshold"], [False, False, False, True, True, True]


def search_transition(table: pd.DataFrame, transition: str, size_grid: list[float], range_grid: list[float], events_grid: list[float], selection: dict[str, float | int], h0_function: str, h1_function: str, ignore_function: str | None, strong_delta_threshold: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline = table[table[FAMILY_SIZE_COL].notna() & (table[FAMILY_SIZE_COL] >= selection["baseline_min_family_size"])].copy()
    if baseline.empty:
        raise ValueError(f"No {transition} families meet baseline family-size threshold {selection['baseline_min_family_size']:g}")
    baseline_metrics = calculate_metrics(baseline, len(baseline), h0_function, h1_function, ignore_function, strong_delta_threshold)

    rows = [{"transition": transition, "filter_rule": "BASELINE", "size_threshold": selection["baseline_min_family_size"], "range_threshold": np.nan, "events_threshold": np.nan, **baseline_metrics, "h1_gain_pp": 0.0, "strong_h1_gain_pp": 0.0, "ignore_change_pp": 0.0, "meets_criteria": True, "is_rule_selected": False, "is_rule_option": False, "option_rank": np.nan, "selection_basis": "baseline"}]
    grids = {"S": size_grid, "R": range_grid, "E": events_grid}

    for rule, rule_features in RULE_FEATURES.items():
        for values in itertools.product(*(grids[feature] for feature in rule_features)):
            thresholds = dict(zip(rule_features, values))
            size_threshold, range_threshold, events_threshold = thresholds.get("S"), thresholds.get("R"), thresholds.get("E")
            kept = baseline.loc[~failure_mask(baseline, rule_features, size_threshold, range_threshold, events_threshold)]
            metrics = calculate_metrics(kept, len(baseline), h0_function, h1_function, ignore_function, strong_delta_threshold)
            h1_gain, strong_gain = metrics["pct_h1"] - baseline_metrics["pct_h1"], metrics["pct_strong_h1"] - baseline_metrics["pct_strong_h1"]
            ignore_change = metrics["pct_ignore"] - baseline_metrics["pct_ignore"]
            meets = metrics["pct_retained"] >= selection["min_retained_pct"] and h1_gain >= selection["min_h1_gain_pp"] and strong_gain >= selection["min_strong_h1_gain_pp"]
            rows.append({"transition": transition, "filter_rule": rule, "size_threshold": size_threshold, "range_threshold": range_threshold, "events_threshold": events_threshold, **metrics, "h1_gain_pp": h1_gain, "strong_h1_gain_pp": strong_gain, "ignore_change_pp": ignore_change, "meets_criteria": bool(meets), "is_rule_selected": False, "is_rule_option": False, "option_rank": np.nan, "selection_basis": ""})

    search = pd.DataFrame(rows)
    option_rows = []
    for rule in RULE_FEATURES:
        group = search[search["filter_rule"] == rule]
        pool, basis, sort_columns, ascending = selection_pool(group, selection)
        top = pool.sort_values(sort_columns, ascending=ascending, na_position="last").head(int(selection["number_of_options"]))
        search.loc[top.index, "is_rule_option"] = True
        search.loc[top.index, "option_rank"] = np.arange(1, len(top) + 1)
        search.loc[top.index, "selection_basis"] = basis
        search.loc[top.index[0], "is_rule_selected"] = True
        option_rows.append(search.loc[top.index].copy())

    options = pd.concat(option_rows, ignore_index=True).sort_values(["filter_rule", "option_rank"])
    three_selected = search[(search["filter_rule"] == "S&R&E") & search["is_rule_selected"]]
    if len(three_selected) != 1:
        raise RuntimeError(f"Expected one selected S&R&E row for {transition}, found {len(three_selected)}")

    selected = three_selected.iloc[0]
    family = table.copy()
    family["baseline_eligible"] = family[FAMILY_SIZE_COL].notna() & (family[FAMILY_SIZE_COL] >= selection["baseline_min_family_size"])
    family["size_threshold"], family["range_threshold"], family["events_threshold"] = selected["size_threshold"], selected["range_threshold"], selected["events_threshold"]
    family["three_rule_selection_basis"] = selected["selection_basis"]
    family["fail_size"] = family[FAMILY_SIZE_COL].isna() | (family[FAMILY_SIZE_COL] < selected["size_threshold"])
    family["fail_range"] = family["chromosome_range"].isna() | (family["chromosome_range"] < selected["range_threshold"])
    family["fail_events"] = family["expected_events"].isna() | (family["expected_events"] < selected["events_threshold"])
    family["n_failed_features"] = family[["fail_size", "fail_range", "fail_events"]].sum(axis=1)
    family["failure_pattern"] = family.apply(lambda row: "+".join(label for label, column in [("S", "fail_size"), ("R", "fail_range"), ("E", "fail_events")] if bool(row[column])) or "none", axis=1)
    family["eliminate_three_failed"] = family["baseline_eligible"] & family["fail_size"] & family["fail_range"] & family["fail_events"]
    family = family[[core.FAMILY_NAME_COL, ac.CHOSEN_FUNCTION_LABEL_COL, FAMILY_SIZE_COL, "chromosome_range", "expected_events", "baseline_eligible", "size_threshold", "range_threshold", "events_threshold", "three_rule_selection_basis", "fail_size", "fail_range", "fail_events", "n_failed_features", "failure_pattern", "eliminate_three_failed"]].sort_values(["eliminate_three_failed", "n_failed_features", core.FAMILY_NAME_COL], ascending=[False, False, True])
    return search, family, options


def save_figure(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_selected_tradeoff(search: pd.DataFrame, transition: str, output: Path) -> None:
    baseline, selected = search[search["filter_rule"] == "BASELINE"].iloc[0], search[search["is_rule_selected"]].copy()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(selected["pct_retained"], selected["pct_strong_h1"], s=70)
    for _, row in selected.iterrows():
        ax.annotate(row["filter_rule"], (row["pct_retained"], row["pct_strong_h1"]), xytext=(5, 5), textcoords="offset points")
    ax.scatter([baseline["pct_retained"]], [baseline["pct_strong_h1"]], marker="x", s=100)
    ax.annotate("baseline", (baseline["pct_retained"], baseline["pct_strong_h1"]), xytext=(5, 5), textcoords="offset points")
    ax.set(title=f"{transition}: selected-rule trade-off", xlabel="Families retained (%)", ylabel="Strong H1 among H0/H1 families (%)")
    ax.grid(alpha=0.25)
    save_figure(fig, output)


def plot_failure_patterns(family: pd.DataFrame, transition: str, output: Path) -> None:
    counts = family.loc[family["baseline_eligible"], "failure_pattern"].value_counts().reindex(PATTERN_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(counts.index, counts.values)
    ax.set(title=f"{transition}: feature-failure overlap at selected S&R&E thresholds", xlabel="Exact failed-feature combination", ylabel="Number of families")
    ax.tick_params(axis="x", rotation=35)
    save_figure(fig, output)


def plot_single_feature_tradeoff(search: pd.DataFrame, transition: str, output: Path, benchmark_threshold: float) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    for rule in ["S", "R", "E"]:
        group = search[search["filter_rule"] == rule].copy().sort_values(THRESHOLD_COLS[rule])
        ax.plot(group["pct_retained"], group["strong_h1_gain_pp"], marker="o", label=rule)
        for _, row in group.iterrows():
            threshold = row[THRESHOLD_COLS[rule]]
            label = f"{rule}={threshold:g}" + (" *" if rule == "S" and np.isclose(threshold, benchmark_threshold) else "")
            ax.annotate(label, (row["pct_retained"], row["strong_h1_gain_pp"]), xytext=(3, 3), textcoords="offset points", fontsize=7)
    ax.axhline(0, linewidth=1)
    ax.set(title=f"{transition}: single-feature informativeness trade-off", xlabel="Families retained (%)", ylabel="Strong-H1 gain vs baseline (percentage points)")
    ax.grid(alpha=0.25)
    ax.legend(title="Filter")
    save_figure(fig, output)


def plot_heatmap(search: pd.DataFrame, transition: str, rule: str, x_feature: str, y_feature: str, output: Path, fixed_feature: str | None = None, fixed_value: float | None = None) -> None:
    data = search[search["filter_rule"] == rule].copy()
    if fixed_feature is not None:
        data = data[np.isclose(data[THRESHOLD_COLS[fixed_feature]], fixed_value)]
    x_col, y_col = THRESHOLD_COLS[x_feature], THRESHOLD_COLS[y_feature]
    gain = data.pivot(index=y_col, columns=x_col, values="strong_h1_gain_pp").sort_index().sort_index(axis=1)
    retained = data.pivot(index=y_col, columns=x_col, values="pct_retained").reindex(index=gain.index, columns=gain.columns)
    if gain.empty:
        return

    fig, ax = plt.subplots(figsize=(max(9, len(gain.columns) * 0.8), max(7, len(gain.index) * 0.65)))
    image = ax.imshow(gain.values, aspect="auto", origin="lower")
    ax.set_xticks(np.arange(len(gain.columns)), labels=[f"{value:g}" for value in gain.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(gain.index)), labels=[f"{value:g}" for value in gain.index])
    ax.set(xlabel=f"{x_feature} threshold", ylabel=f"{y_feature} threshold")
    title = f"{transition}: {rule} strong-H1 gain"
    if fixed_feature is not None:
        title += f" ({fixed_feature}={fixed_value:g})"
    ax.set_title(title)
    for i in range(len(gain.index)):
        for j in range(len(gain.columns)):
            if np.isfinite(gain.iat[i, j]):
                ax.text(j, i, f"{gain.iat[i, j]:+.1f}\n{retained.iat[i, j]:.0f}%", ha="center", va="center", fontsize=6)
    fig.colorbar(image, ax=ax, label="Strong-H1 gain (pp)")
    save_figure(fig, output)


def pareto_frontier(search: pd.DataFrame) -> pd.DataFrame:
    data = search[(search["filter_rule"] != "BASELINE") & (search["n_eliminated"] > 0) & search["pct_retained"].notna() & search["strong_h1_gain_pp"].notna()].copy()
    data = data.sort_values(["pct_retained", "strong_h1_gain_pp", "h1_gain_pp"], ascending=[False, False, False]).drop_duplicates("pct_retained", keep="first")
    frontier_rows, best_gain = [], -np.inf
    for _, row in data.iterrows():
        if row["strong_h1_gain_pp"] > best_gain:
            frontier_rows.append(row)
            best_gain = row["strong_h1_gain_pp"]
    return pd.DataFrame(frontier_rows).sort_values("pct_retained")


def plot_pareto_tradeoff(search: pd.DataFrame, frontier: pd.DataFrame, transition: str, output: Path) -> None:
    data = search[(search["filter_rule"] != "BASELINE") & (search["n_eliminated"] > 0)].copy()
    fig, ax = plt.subplots(figsize=(10, 7))
    for rule, group in data.groupby("filter_rule"):
        ax.scatter(group["pct_retained"], group["strong_h1_gain_pp"], s=18, alpha=0.45, label=rule)
    if not frontier.empty:
        ax.plot(frontier["pct_retained"], frontier["strong_h1_gain_pp"], marker="o", linewidth=2, label="Pareto frontier")
        for _, row in frontier.iterrows():
            ax.annotate(row["filter_rule"], (row["pct_retained"], row["strong_h1_gain_pp"]), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.axhline(0, linewidth=1)
    ax.set(title=f"{transition}: filter trade-off and Pareto frontier", xlabel="Families retained (%)", ylabel="Strong-H1 gain vs baseline (percentage points)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    save_figure(fig, output)


def threshold_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def build_visualizations(search: pd.DataFrame, transition: str, output_dir: Path, benchmark_threshold: float) -> list[Path]:
    outputs = []
    single_plot = output_dir / f"{transition}_single_feature_tradeoff.png"
    plot_single_feature_tradeoff(search, transition, single_plot, benchmark_threshold)
    outputs.append(single_plot)

    for rule, x_feature, y_feature, label in [("S&R", "S", "R", "SR"), ("S&E", "S", "E", "SE"), ("R&E", "R", "E", "RE")]:
        path = output_dir / f"{transition}_{label}_heatmap.png"
        plot_heatmap(search, transition, rule, x_feature, y_feature, path)
        outputs.append(path)

    three_dir = output_dir / "three_feature_heatmaps" / transition
    for event_threshold in sorted(search.loc[search["filter_rule"] == "S&R&E", "events_threshold"].dropna().unique()):
        path = three_dir / f"E_{threshold_tag(float(event_threshold))}.png"
        plot_heatmap(search, transition, "S&R&E", "S", "R", path, fixed_feature="E", fixed_value=float(event_threshold))
        outputs.append(path)

    frontier = pareto_frontier(search)
    frontier_file = output_dir / f"{transition}_pareto_frontier.csv"
    frontier_cols = ["filter_rule", "size_threshold", "range_threshold", "events_threshold", "n_retained", "pct_retained", "h1_gain_pp", "strong_h1_gain_pp", "meets_criteria"]
    frontier[frontier_cols].to_csv(frontier_file, index=False)
    pareto_plot = output_dir / f"{transition}_pareto_tradeoff.png"
    plot_pareto_tradeoff(search, frontier, transition, pareto_plot)
    outputs.extend([frontier_file, pareto_plot])
    return outputs



def plot_family_size_thresholds(search_tables: dict[str, pd.DataFrame], transitions: list[str], benchmark_threshold: float, output: Path) -> None:
    fig, axes = plt.subplots(len(transitions), 1, figsize=(10, 4 * len(transitions)), sharex=True)
    axes = np.atleast_1d(axes)

    for ax, transition in zip(axes, transitions):
        data = search_tables[transition].loc[search_tables[transition]["filter_rule"] == "S"].sort_values("size_threshold")
        ax.plot(data["size_threshold"], data["h1_gain_pp"], marker="o", label="H1 gain")
        ax.plot(data["size_threshold"], data["strong_h1_gain_pp"], marker="s", label="Strong-H1 gain")
        ax.axvline(benchmark_threshold, linestyle=":", linewidth=1.5, label=f"S={benchmark_threshold:g}")
        ax.set_ylabel("Gain vs baseline (pp)")
        ax.set_title(transition)
        ax.grid(alpha=0.25)

        retained_ax = ax.twinx()
        retained_ax.plot(data["size_threshold"], data["pct_retained"], marker="^", linestyle="--", label="Families retained")
        retained_ax.set_ylabel("Families retained (%)")
        retained_ax.set_ylim(0, 105)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = retained_ax.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="best")

    axes[-1].set_xlabel("Minimum family size")
    fig.suptitle("Family-size cutoff: informativeness gain versus dataset retention", y=1.01)
    save_figure(fig, output)


def evaluate_fixed_size_cutoff(table: pd.DataFrame, transition: str, cutoff: float, selection: dict[str, float | int], h0_function: str, h1_function: str, ignore_function: str | None, strong_delta_threshold: float) -> tuple[dict[str, Any], list[str], pd.DataFrame]:
    baseline = table[table[FAMILY_SIZE_COL].notna() & (table[FAMILY_SIZE_COL] >= selection["baseline_min_family_size"])].copy()
    kept = baseline[baseline[FAMILY_SIZE_COL] >= cutoff].copy()
    baseline_metrics = calculate_metrics(baseline, len(baseline), h0_function, h1_function, ignore_function, strong_delta_threshold)
    metrics = calculate_metrics(kept, len(baseline), h0_function, h1_function, ignore_function, strong_delta_threshold)
    eliminated_names = sorted(baseline.loc[baseline[FAMILY_SIZE_COL] < cutoff, core.FAMILY_NAME_COL].astype(str).tolist())
    row = {"transition": transition, "size_threshold": cutoff, **metrics, "h1_gain_pp": metrics["pct_h1"] - baseline_metrics["pct_h1"], "strong_h1_gain_pp": metrics["pct_strong_h1"] - baseline_metrics["pct_strong_h1"], "ignore_change_pp": metrics["pct_ignore"] - baseline_metrics["pct_ignore"]}

    observed_models = set(baseline[ac.CHOSEN_FUNCTION_LABEL_COL].dropna().astype(str))
    ordered_models = [model for model in [h0_function, h1_function] if model in observed_models]
    ordered_models += sorted(observed_models - set(ordered_models) - ({ignore_function} if ignore_function else set()))
    if ignore_function and ignore_function in observed_models:
        ordered_models.append(ignore_function)

    model_rows = []
    for stage, data in [("before_cutoff", baseline), ("after_cutoff", kept)]:
        counts = data[ac.CHOSEN_FUNCTION_LABEL_COL].astype(str).value_counts()
        model_rows.extend({"transition": transition, "size_threshold": cutoff, "stage": stage, "model": model, "n": int(counts.get(model, 0)), "pct_selected": 100 * int(counts.get(model, 0)) / len(data) if len(data) else np.nan} for model in ordered_models)
    return row, eliminated_names, pd.DataFrame(model_rows)


def rank_single_features(single_feature_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranked_parts = []
    for transition, group in single_feature_rows.groupby("transition", sort=False):
        ranked = group.copy()
        ranked["criteria_met"] = ranked["selection_basis"].eq("criteria_met")
        ranked["pct_eliminated"] = 100 - ranked["pct_retained"]
        ranked["h1_gain_per_10pct_removed"] = np.where(ranked["pct_eliminated"] > 0, 10 * ranked["h1_gain_pp"] / ranked["pct_eliminated"], np.nan)
        ranked["strong_h1_gain_per_10pct_removed"] = np.where(ranked["pct_eliminated"] > 0, 10 * ranked["strong_h1_gain_pp"] / ranked["pct_eliminated"], np.nan)
        ranked = ranked.sort_values(["criteria_met", "pct_retained", "strong_h1_gain_pp", "h1_gain_pp"], ascending=[False, False, False, False]).reset_index(drop=True)
        ranked["transition_rank"] = np.arange(1, len(ranked) + 1)
        ranked_parts.append(ranked)

    ranked_all = pd.concat(ranked_parts, ignore_index=True)
    summary = ranked_all.groupby("filter_rule", as_index=False).agg(
        n_transitions=("transition", "nunique"),
        n_transitions_criteria_met=("criteria_met", "sum"),
        n_transitions_best=("transition_rank", lambda x: int((x == 1).sum())),
        mean_transition_rank=("transition_rank", "mean"),
        mean_pct_retained=("pct_retained", "mean"),
        mean_h1_gain_pp=("h1_gain_pp", "mean"),
        mean_strong_h1_gain_pp=("strong_h1_gain_pp", "mean"),
        mean_h1_gain_per_10pct_removed=("h1_gain_per_10pct_removed", "mean"),
        mean_strong_h1_gain_per_10pct_removed=("strong_h1_gain_per_10pct_removed", "mean"),
    )
    summary = summary.sort_values(["n_transitions_criteria_met", "n_transitions_best", "mean_transition_rank", "mean_pct_retained", "mean_strong_h1_gain_pp", "mean_h1_gain_pp"], ascending=[False, False, True, False, False, False]).reset_index(drop=True)
    summary["overall_rank"] = np.arange(1, len(summary) + 1)
    return ranked_all, summary


def build_cross_transition_output(family_tables: dict[str, pd.DataFrame], transitions: list[str], output_csv: Path, output_plot: Path) -> pd.DataFrame:
    merged = None
    for transition in transitions:
        part = family_tables[transition][[core.FAMILY_NAME_COL, FAMILY_SIZE_COL, "chromosome_range", "expected_events", "eliminate_three_failed"]].rename(columns={"expected_events": f"{transition}_expected_events", "eliminate_three_failed": f"{transition}_three_failed"})
        merged = part if merged is None else merged.merge(part.drop(columns=[FAMILY_SIZE_COL, "chromosome_range"]), on=core.FAMILY_NAME_COL, how="outer", validate="one_to_one")
    if merged is None:
        raise RuntimeError("No family tables were provided")

    flag_columns = [f"{transition}_three_failed" for transition in transitions]
    for column in flag_columns:
        merged[column] = merged[column].fillna(False).astype(bool)
    merged["n_transitions_three_failed"] = merged[flag_columns].sum(axis=1)
    all_transitions = merged[merged["n_transitions_three_failed"] == len(transitions)].copy()
    all_transitions.to_csv(output_csv, index=False)

    counts = merged["n_transitions_three_failed"].value_counts().reindex(range(len(transitions) + 1), fill_value=0).sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(counts.index.astype(str), counts.values)
    ax.set(title="Three-feature elimination agreement across transitions", xlabel="Number of transitions eliminating the family", ylabel="Number of families")
    save_figure(fig, output_plot)
    return all_transitions


def run(args: argparse.Namespace) -> dict[str, Any]:
    analysis_root, grid_json = args.analysis_root.resolve(), args.grid_json.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else analysis_root.parent / "family_informativeness_filter" / (args.comparison_label or f"{args.h0_function}_vs_{args.h1_function}")
    output_dir.mkdir(parents=True, exist_ok=True)

    transitions = list(args.transitions)
    size_grid, range_grid, event_grids, selection = load_grid(grid_json, args.event_model, transitions)
    size_grid = sorted(set(size_grid + [float(args.size_benchmark_threshold)]))

    family_tables, search_tables, single_feature_rows, size_benchmark_rows, size_benchmark_model_rows, outputs = {}, {}, [], [], [], []

    for transition in transitions:
        table, _ = load_transition_table(analysis_root, args.transition_input_subdir, transition, args.h0_function, args.h1_function, args.event_model)
        search, family, options = search_transition(table, transition, size_grid, range_grid, event_grids[transition], selection, args.h0_function, args.h1_function, args.ignore_function, args.strong_delta_threshold)

        search_file, options_file = output_dir / f"{transition}_filter_search.csv", output_dir / f"{transition}_candidate_options.csv"
        family_file, tradeoff_plot = output_dir / f"{transition}_family_filter.csv", output_dir / f"{transition}_rule_tradeoff.png"
        overlap_plot = output_dir / f"{transition}_feature_overlap.png"
        search.to_csv(search_file, index=False)
        options.to_csv(options_file, index=False)
        family.to_csv(family_file, index=False)
        plot_selected_tradeoff(search, transition, tradeoff_plot)
        plot_failure_patterns(family, transition, overlap_plot)
        outputs.extend([search_file, options_file, family_file, tradeoff_plot, overlap_plot])
        outputs.extend(build_visualizations(search, transition, output_dir, args.size_benchmark_threshold))

        family_tables[transition] = family
        search_tables[transition] = search
        single_feature_rows.append(search.loc[search["is_rule_selected"] & search["filter_rule"].isin(["S", "R", "E"])].copy())

        benchmark, benchmark_names, benchmark_models = evaluate_fixed_size_cutoff(table, transition, args.size_benchmark_threshold, selection, args.h0_function, args.h1_function, args.ignore_function, args.strong_delta_threshold)
        size_benchmark_rows.append(benchmark)
        size_benchmark_model_rows.append(benchmark_models)

        print(f"\n[✓] {transition}: tested {len(search) - 1} filter combinations")
        print(search.loc[search["is_rule_selected"], ["filter_rule", "size_threshold", "range_threshold", "events_threshold", "n_retained", "pct_retained", "pct_h1", "pct_strong_h1", "h1_gain_pp", "strong_h1_gain_pp", "selection_basis"]].to_string(index=False))

        eliminated_families = sorted(family.loc[family["eliminate_three_failed"], core.FAMILY_NAME_COL].astype(str).tolist())
        print(f"\n[✓] Families eliminated by all three features for {transition}: {len(eliminated_families)}")
        print(f"    {', '.join(eliminated_families)}" if eliminated_families else "    None")

        print(f"\n[✓] Fixed family-size benchmark S >= {args.size_benchmark_threshold:g} for {transition}")
        print(f"    retained: {benchmark['n_retained']} ({benchmark['pct_retained']:.2f}%) | H1: {benchmark['pct_h1']:.2f}% ({benchmark['h1_gain_pp']:+.2f} pp) | strong H1: {benchmark['pct_strong_h1']:.2f}% ({benchmark['strong_h1_gain_pp']:+.2f} pp)")
        for stage, label in [("before_cutoff", "model selection before cutoff"), ("after_cutoff", f"model selection after S >= {args.size_benchmark_threshold:g}")]:
            stage_rows = benchmark_models[benchmark_models["stage"] == stage]
            summary = " | ".join(f"{row.model} {row.pct_selected:.2f}%" for row in stage_rows.itertuples())
            print(f"    {label}: {summary}")
        print(f"    eliminated families ({len(benchmark_names)}): {', '.join(benchmark_names) if benchmark_names else 'None'}")

    family_size_plot = output_dir / "family_size_threshold_tradeoff.png"
    plot_family_size_thresholds(search_tables, transitions, args.size_benchmark_threshold, family_size_plot)
    outputs.append(family_size_plot)

    single_feature_comparison, single_feature_summary = rank_single_features(pd.concat(single_feature_rows, ignore_index=True))
    single_feature_file, single_feature_summary_file = output_dir / "single_feature_comparison.csv", output_dir / "single_feature_summary.csv"
    size_benchmark_file = output_dir / f"family_size_{args.size_benchmark_threshold:g}_benchmark.csv"
    size_benchmark_models_file = output_dir / f"family_size_{args.size_benchmark_threshold:g}_model_selection.csv"
    single_feature_comparison.to_csv(single_feature_file, index=False)
    single_feature_summary.to_csv(single_feature_summary_file, index=False)
    pd.DataFrame(size_benchmark_rows).to_csv(size_benchmark_file, index=False)
    pd.concat(size_benchmark_model_rows, ignore_index=True).to_csv(size_benchmark_models_file, index=False)
    outputs.extend([single_feature_file, single_feature_summary_file, size_benchmark_file, size_benchmark_models_file])

    print("\n[✓] Single-feature comparison")
    print(single_feature_comparison[["transition", "filter_rule", "size_threshold", "range_threshold", "events_threshold", "pct_retained", "h1_gain_pp", "strong_h1_gain_pp", "transition_rank"]].to_string(index=False))
    print("\n[✓] Single-feature summary across transitions")
    print(single_feature_summary[["overall_rank", "filter_rule", "n_transitions_criteria_met", "n_transitions_best", "mean_transition_rank", "mean_pct_retained", "mean_h1_gain_pp", "mean_strong_h1_gain_pp"]].to_string(index=False))
    leading_feature = str(single_feature_summary.iloc[0]["filter_rule"])
    print(f"\n[✓] Leading single feature: {leading_feature}")

    all_transitions_file, cross_plot = output_dir / "families_three_failed_all_transitions.csv", output_dir / "three_failed_cross_transition_summary.png"
    all_transitions = build_cross_transition_output(family_tables, transitions, all_transitions_file, cross_plot)
    outputs.extend([all_transitions_file, cross_plot])

    all_three_names = sorted(all_transitions[core.FAMILY_NAME_COL].astype(str).tolist())
    at_least_two_sets = [set(family_tables[t].loc[family_tables[t]["baseline_eligible"] & (family_tables[t]["n_failed_features"] >= 2), core.FAMILY_NAME_COL].astype(str)) for t in transitions]
    all_at_least_two_names = sorted(set.intersection(*at_least_two_sets)) if at_least_two_sets else []

    print(f"\n[✓] Families eliminated by all three features in all transitions: {len(all_three_names)}")
    print(f"    {', '.join(all_three_names)}" if all_three_names else "    None")
    print(f"\n[✓] Families eliminated by at least two features in all transitions: {len(all_at_least_two_names)}")
    print(f"    {', '.join(all_at_least_two_names)}" if all_at_least_two_names else "    None")
    print(f"\n[✓] Outputs: {output_dir}")

    if not args.skip_log:
        log_run(
            step="analysis", script=Path(__file__),
            params={"comparison_label": args.comparison_label, "analysis_root": str(analysis_root), "transition_input_subdir": str(args.transition_input_subdir), "grid_json": str(grid_json), "h0_function": args.h0_function, "h1_function": args.h1_function, "ignore_function": args.ignore_function, "event_model": args.event_model, "transitions": transitions, "strong_delta_threshold": args.strong_delta_threshold, "size_benchmark_threshold": args.size_benchmark_threshold, **selection},
            outputs=[str(path) for path in outputs],
            description=f"Family informativeness filter search for {args.h0_function} vs {args.h1_function}.",
            notes="Tests one-, two-, and three-feature AND rules, returns multiple candidate options, compares S/R/E, evaluates a fixed size benchmark, and writes single-feature, heatmap, and Pareto trade-off visualizations.",
            log_relative_path=Path("family_informativeness_filter") / (args.comparison_label or f"{args.h0_function}_vs_{args.h1_function}") / "run.log",
        )

    return {"output_dir": output_dir, "outputs": outputs, "leading_single_feature": leading_feature, "n_all_transition_three_failed": len(all_transitions)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search family-size, chromosome-range, and expected-event filtering rules for ChromEvol model comparisons.")
    parser.add_argument("--analysis-root", required=True, type=Path, help="Root containing one subdirectory per transition.")
    parser.add_argument("--transition-input-subdir", type=Path, default=Path("."), help="Optional subdirectory inside each transition directory containing the input CSVs.")
    parser.add_argument("--grid-json", required=True, type=Path, help="JSON containing S, R, E threshold grids and selection criteria.")
    parser.add_argument("--h0-function", required=True)
    parser.add_argument("--h1-function", required=True)
    parser.add_argument("--event-model", required=True, help="Model whose inferred expected-event values are used as feature E.")
    parser.add_argument("--ignore-function", default="ignore", help="Chosen-function label representing ignore. Use an empty string to disable.")
    parser.add_argument("--comparison-label", help="Label used in the default output and log directory names.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--transitions", nargs="+", choices=list(core.LABEL_TRANSITIONS_ORDERED), default=["gain", "loss", "dupl"])
    parser.add_argument("--strong-delta-threshold", type=float, default=ac.STRONG_DELTA_AICC_THRESHOLD, help="Absolute pairwise ΔAICc required for strong H1 support.")
    parser.add_argument("--size-benchmark-threshold", type=float, default=100.0, help="Fixed family-size cutoff evaluated as a simple benchmark. Default: 100.")
    parser.add_argument("--skip-log", action="store_true")
    args = parser.parse_args()
    if args.ignore_function == "":
        args.ignore_function = None
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()