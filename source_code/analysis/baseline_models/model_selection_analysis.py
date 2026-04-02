import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from source_code.analysis.analysis_constants import *
from source_code.analysis.baseline_models.plot_utils import (
    ensure_dir,
    write_text,
    plot_bar_counts,
    plot_hist_overlay,
)


MODEL_ORDER = list(BASELINE_FUNC_LABELS_ORDERED)
EXPECTED_MODEL_LABELS = {LABEL_CONSTANT, LABEL_LINEAR, LABEL_IGNORE}


# Loading/validation
def load_summary_table(transition: str) -> pd.DataFrame:
    file = BASELINE_ANALYSIS_DIR / transition / f"{transition}_models_summary_table.csv"

    if not file.exists():
        raise FileNotFoundError(f"Missing models summary table: {file}")

    return pd.read_csv(file)


def validate_columns(df: pd.DataFrame, required_cols: List[str]) -> None:
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")


# AICc helpers
def add_akaike_weights(group: pd.DataFrame) -> pd.DataFrame:
    group = group.copy()
    group[AICC_COL] = pd.to_numeric(group[AICC_COL], errors="coerce")
    group = group.dropna(subset=[AICC_COL])

    if group.empty:
        group["delta_aicc"] = np.nan
        group["akaike_weight"] = np.nan
        return group

    best_aicc = group[AICC_COL].min()
    group["delta_aicc"] = group[AICC_COL] - best_aicc

    rel_lik = np.exp(-0.5 * group["delta_aicc"])
    denom = rel_lik.sum()
    group["akaike_weight"] = rel_lik / denom if denom > 0 else np.nan

    return group


def delta_support_label(delta: float) -> str:
    if pd.isna(delta):
        return "missing"
    if delta <= 2:
        return "<=2"
    if delta <= 4:
        return "2-4"
    if delta <= 10:
        return "4-10"
    return ">10"


def weight_support_label(weight: float) -> str:
    if pd.isna(weight):
        return "missing"
    if weight < 0.6:
        return "<0.6"
    if weight < 0.8:
        return "0.6-0.8"
    if weight < 0.95:
        return "0.8-0.95"
    return ">0.95"


# Family-level table
def build_family_selection_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for family_name, group in summary_df.groupby(FAMILY_NAME_COL):
        group = add_akaike_weights(group)

        if group.empty or len(group) < 2:
            continue

        group = group.sort_values([AICC_COL, LABEL_FUNC_TYPE_COL]).reset_index(drop=True)

        best_row = group.iloc[0]
        second_best_aicc = group.iloc[1][AICC_COL] if len(group) > 1 else np.nan
        delta_best_vs_second = second_best_aicc - best_row[AICC_COL] if pd.notna(second_best_aicc) else np.nan

        best_aicc = best_row[AICC_COL]
        n_best = int((group[AICC_COL] == best_aicc).sum())
        present_labels = set(group[LABEL_FUNC_TYPE_COL].dropna())
        missing_labels = sorted(EXPECTED_MODEL_LABELS - present_labels)

        row = {
            FAMILY_NAME_COL: family_name,
            CHOSEN_FUNCTION_LABEL_COL: best_row[LABEL_FUNC_TYPE_COL],
            CHOSEN_CONFIG_COL: best_row[CONFIG_COL],
            "best_aicc": best_aicc,
            "second_best_aicc": second_best_aicc,
            "delta_best_vs_second": delta_best_vs_second,
            "best_akaike_weight": best_row["akaike_weight"],
            "n_models_compared": len(group),
            "is_tied_best": n_best > 1,
            "missing_model_labels": ",".join(missing_labels),
        }

        if NUM_OF_EVENTS_COL in best_row.index:
            row[NUM_OF_EVENTS_COL] = best_row[NUM_OF_EVENTS_COL]

        if ROOT_CHROM_NUM_COL in best_row.index:
            row[ROOT_CHROM_NUM_COL] = best_row[ROOT_CHROM_NUM_COL]

        if BASE_CHROM_NUM_COL in best_row.index:
            row[BASE_CHROM_NUM_COL] = best_row[BASE_CHROM_NUM_COL]

        rows.append(row)

    return pd.DataFrame(rows)


# Summary text
def build_summary_text(selection_df: pd.DataFrame, transition: str) -> str:
    chosen_counts = selection_df[CHOSEN_FUNCTION_LABEL_COL].value_counts()
    delta_counts = selection_df["delta_support"].value_counts()
    weight_counts = selection_df["weight_support"].value_counts()

    n_tied = int(selection_df["is_tied_best"].sum()) if "is_tied_best" in selection_df.columns else 0
    incomplete_count = 0
    if "missing_model_labels" in selection_df.columns:
        incomplete_count = int((selection_df["missing_model_labels"] != "").sum())

    lines = [
        f"Transition: {transition}",
        "=" * 50,
        "",
        f"Families included: {len(selection_df)}",
        f"Tied best AICc families: {n_tied}",
        f"Families with missing candidate models: {incomplete_count}",
        "",
        "Chosen model counts:",
        chosen_counts.to_string(),
        "",
        "Delta AICc support counts:",
        delta_counts.to_string(),
        "",
        "Akaike weight support counts:",
        weight_counts.to_string(),
        "",
        "Delta AICc summary:",
        selection_df["delta_best_vs_second"].describe().to_string(),
        "",
        "Best Akaike weight summary:",
        selection_df["best_akaike_weight"].describe().to_string(),
    ]

    return "\n".join(lines)


# Plots
def plot_chosen_model_counts(selection_df: pd.DataFrame, transition: str, out_dir: Path) -> None:
    counts = selection_df[CHOSEN_FUNCTION_LABEL_COL].value_counts()
    counts = pd.Series({label: int(counts.get(label, 0)) for label in MODEL_ORDER})

    plot_bar_counts(
        counts=counts,
        title=f"{transition}: chosen model counts",
        xlabel="Chosen model",
        ylabel="Number of families",
        out_file=out_dir / f"{transition}_chosen_model_counts.png",
    )


def plot_chosen_model_counts_no_ignore(selection_df: pd.DataFrame, transition: str, out_dir: Path) -> None:
    sub = selection_df.loc[
        selection_df[CHOSEN_FUNCTION_LABEL_COL].isin([LABEL_CONSTANT, LABEL_LINEAR])
    ].copy()

    counts = sub[CHOSEN_FUNCTION_LABEL_COL].value_counts()
    counts = pd.Series(
        {
            LABEL_CONSTANT: int(counts.get(LABEL_CONSTANT, 0)),
            LABEL_LINEAR: int(counts.get(LABEL_LINEAR, 0)),
        }
    )

    plot_bar_counts(
        counts=counts,
        title=f"{transition}: chosen model counts (constant vs linear)",
        xlabel="Chosen model",
        ylabel="Number of families",
        out_file=out_dir / f"{transition}_chosen_model_counts_no_ignore.png",
    )


def plot_delta_aicc_hist(selection_df: pd.DataFrame, transition: str, out_dir: Path) -> None:
    plot_hist_overlay(
        series_map={"delta AICc": selection_df["delta_best_vs_second"]},
        title=f"{transition}: delta AICc between best and second-best",
        xlabel="Delta AICc",
        ylabel="Number of families",
        out_file=out_dir / f"{transition}_delta_aicc_hist.png",
        bins=30,
    )


def plot_akaike_weight_hist(selection_df: pd.DataFrame, transition: str, out_dir: Path) -> None:
    plot_hist_overlay(
        series_map={"best Akaike weight": selection_df["best_akaike_weight"]},
        title=f"{transition}: best-model Akaike weight",
        xlabel="Akaike weight",
        ylabel="Number of families",
        out_file=out_dir / f"{transition}_best_akaike_weight_hist.png",
        bins=30,
    )


def plot_support_counts(selection_df: pd.DataFrame, transition: str, out_dir: Path) -> None:
    delta_order = ["<=2", "2-4", "4-10", ">10", "missing"]

    delta_plot_df = selection_df[[CHOSEN_FUNCTION_LABEL_COL, "delta_support"]].copy()
    delta_plot_df["delta_support"] = delta_plot_df["delta_support"].fillna("missing")

    delta_counts_by_model = (
        delta_plot_df.groupby(["delta_support", CHOSEN_FUNCTION_LABEL_COL])
        .size()
        .unstack(fill_value=0)
    )

    delta_counts_by_model = delta_counts_by_model.reindex(delta_order, fill_value=0)

    for model_label in BASELINE_FUNC_LABELS_ORDERED:
        if model_label not in delta_counts_by_model.columns:
            delta_counts_by_model[model_label] = 0

    delta_counts_by_model = delta_counts_by_model[list(BASELINE_FUNC_LABELS_ORDERED)]

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    x_labels = delta_counts_by_model.index.tolist()
    bottoms = [0] * len(x_labels)

    for model_label in delta_counts_by_model.columns:
        vals = delta_counts_by_model[model_label].tolist()

        bars = ax.bar(
            x_labels,
            vals,
            bottom=bottoms,
            label=model_label,
            color=MODEL_COLOR_MAP.get(model_label),
        )

        for i, (bar, val) in enumerate(zip(bars, vals)):
            total = int(delta_counts_by_model.iloc[i].sum())
            if val == 0 or total == 0:
                continue

            pct = 100 * val / total
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bottoms[i] + val / 2,
                f"{val}\n{pct:.1f}%",
                ha="center",
                va="center",
                fontsize=9,
            )

        bottoms = [b + v for b, v in zip(bottoms, vals)]

    for i, total in enumerate(delta_counts_by_model.sum(axis=1).tolist()):
        if total > 0:
            ax.text(i, total, f"n={int(total)}", ha="center", va="bottom")

    ax.set_title(f"{transition}: delta AICc support categories by chosen model")
    ax.set_xlabel("Support category")
    ax.set_ylabel("Number of families")
    ax.legend()

    save_close(fig, out_dir / f"{transition}_delta_support_counts.png")

    weight_order = ["<0.6", "0.6-0.8", "0.8-0.95", ">0.95", "missing"]
    weight_counts = selection_df["weight_support"].value_counts()
    weight_counts = pd.Series({label: int(weight_counts.get(label, 0)) for label in weight_order})

    plot_bar_counts(
        counts=weight_counts,
        title=f"{transition}: Akaike weight support categories",
        xlabel="Support category",
        ylabel="Number of families",
        out_file=out_dir / f"{transition}_akaike_weight_support_counts.png",
        color_map=None,
    )


# Run
def run_transition(transition: str) -> None:
    summary_df = load_summary_table(transition)

    validate_columns(
        summary_df,
        [
            FAMILY_NAME_COL,
            CONFIG_COL,
            LABEL_FUNC_TYPE_COL,
            AICC_COL,
            NUM_OF_EVENTS_COL,
            ROOT_CHROM_NUM_COL,
        ],
    )

    out_dir = BASELINE_ANALYSIS_DIR / transition / "model_selection"
    ensure_dir(out_dir)

    selection_df = build_family_selection_table(summary_df)

    if selection_df.empty:
        raise ValueError(f"No valid family-level model selection rows were built for transition '{transition}'.")

    selection_df["delta_support"] = selection_df["delta_best_vs_second"].apply(delta_support_label)
    selection_df["weight_support"] = selection_df["best_akaike_weight"].apply(weight_support_label)

    selection_df.to_csv(out_dir / f"{transition}_family_model_selection_table.csv", index=False)

    plot_chosen_model_counts(selection_df, transition, out_dir)
    plot_chosen_model_counts_no_ignore(selection_df, transition, out_dir)
    plot_delta_aicc_hist(selection_df, transition, out_dir)
    plot_akaike_weight_hist(selection_df, transition, out_dir)
    plot_support_counts(selection_df, transition, out_dir)

    summary = build_summary_text(selection_df, transition)
    write_text(summary, out_dir / f"{transition}_model_selection_summary.txt")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transition", required=True, type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    run_transition(args.transition)


if __name__ == "__main__":
    main()