import argparse

import numpy as np
import pandas as pd

from source_code.analysis.plot_utils import (
    ensure_dir,
    get_ordered_counts,
    plot_bar_counts,
    plot_pie_counts,
)
from source_code.analysis.exp_lin_analysis.exp_lin_analysis_constants import *
from source_code.logger import log_run


# Model-selection model order

MODEL_LABELS_WITH_EXP: tuple[str, ...] = (
    LABEL_CONSTANT,
    LABEL_LINEAR,
    LABEL_EXP,
    LABEL_IGNORE,
)


# Build EXP rows

def build_exp_summary_df_from_core(core_df: pd.DataFrame) -> pd.DataFrame:
    # Build EXP rows in the same long format as baseline models_summary_table

    required_cols = [FAMILY_NAME_COL, M1_EXP_AICC_COL]
    missing_cols = [col for col in required_cols if col not in core_df.columns]

    if missing_cols:
        raise ValueError(f"Core comparison table is missing columns: {missing_cols}")

    exp_df = core_df[required_cols].copy()

    exp_df = exp_df.rename(
        columns={
            M1_EXP_AICC_COL: AICC_COL,
        }
    )

    exp_df[LABEL_FUNC_TYPE_COL] = LABEL_EXP
    exp_df[AICC_COL] = pd.to_numeric(exp_df[AICC_COL], errors="coerce")

    return exp_df[[FAMILY_NAME_COL, LABEL_FUNC_TYPE_COL, AICC_COL]]


# Build combined model summary

def build_model_summary_with_exp(baseline_model_summary_df: pd.DataFrame, core_df: pd.DataFrame, transition_label: str) -> pd.DataFrame:
    # Concatenate baseline model summary with EXP rows

    baseline_df = baseline_model_summary_df.copy()

    if LABEL_TESTED_TRANSITION_COL in baseline_df.columns:
        baseline_df = baseline_df[
            baseline_df[LABEL_TESTED_TRANSITION_COL] == transition_label
        ].copy()

    required_cols = [FAMILY_NAME_COL, LABEL_FUNC_TYPE_COL, AICC_COL]
    missing_cols = [col for col in required_cols if col not in baseline_df.columns]

    if missing_cols:
        raise ValueError(f"Baseline model summary table is missing columns: {missing_cols}")

    baseline_df = baseline_df[
        baseline_df[LABEL_FUNC_TYPE_COL].isin(
            [LABEL_CONSTANT, LABEL_LINEAR, LABEL_IGNORE]
        )
    ].copy()

    baseline_df = baseline_df[required_cols].copy()
    baseline_df[AICC_COL] = pd.to_numeric(baseline_df[AICC_COL], errors="coerce")

    exp_df = build_exp_summary_df_from_core(core_df=core_df)

    model_summary_df = pd.concat(
        [baseline_df, exp_df],
        ignore_index=True,
    )

    model_summary_df = model_summary_df.dropna(
        subset=[FAMILY_NAME_COL, LABEL_FUNC_TYPE_COL, AICC_COL]
    )

    model_summary_df = model_summary_df.sort_values(
        by=[FAMILY_NAME_COL, LABEL_FUNC_TYPE_COL],
        ascending=[True, True],
    ).reset_index(drop=True)

    return model_summary_df


# Akaike weights

def add_akaike_weights(model_summary_df: pd.DataFrame) -> pd.DataFrame:
    # Add delta AICc and Akaike weights per family

    output_df = model_summary_df.copy()
    output_df[AICC_COL] = pd.to_numeric(output_df[AICC_COL], errors="coerce")

    output_df[DELTA_AICC_COL] = (
        output_df[AICC_COL]
        - output_df.groupby(FAMILY_NAME_COL)[AICC_COL].transform("min")
    )

    output_df["relative_likelihood"] = np.exp(-0.5 * output_df[DELTA_AICC_COL])

    output_df[AKAIKE_WEIGHT_COL] = (
        output_df["relative_likelihood"]
        / output_df.groupby(FAMILY_NAME_COL)["relative_likelihood"].transform("sum")
    )

    output_df = output_df.drop(columns=["relative_likelihood"])

    return output_df


# Chosen model table

def build_chosen_model_table_with_exp(
    model_summary_with_weights_df: pd.DataFrame,
) -> pd.DataFrame:
    # Build chosen model table after adding EXP

    idx = model_summary_with_weights_df.groupby(FAMILY_NAME_COL)[AICC_COL].idxmin()

    chosen_df = model_summary_with_weights_df.loc[idx].copy()

    chosen_df = chosen_df.rename(
        columns={
            LABEL_FUNC_TYPE_COL: CHOSEN_FUNCTION_LABEL_COL,
            AKAIKE_WEIGHT_COL: BEST_AKAIKE_WEIGHT_COL,
        }
    )

    chosen_df = chosen_df[
        [
            FAMILY_NAME_COL,
            CHOSEN_FUNCTION_LABEL_COL,
            AICC_COL,
            DELTA_AICC_COL,
            BEST_AKAIKE_WEIGHT_COL,
        ]
    ].copy()

    chosen_df = chosen_df.sort_values(by=FAMILY_NAME_COL).reset_index(drop=True)

    return chosen_df


# Dependence detection status

def add_dependence_detection_status(core_df: pd.DataFrame) -> pd.DataFrame:
    # Add EXP/LINEAR-vs-M0 dependence-detection category

    output_df = core_df.copy()

    exp_chosen = output_df[M0_VS_M1_EXP_COL] == LABEL_EXP
    linear_chosen = output_df[M0_VS_M1_LINEAR_COL] == LABEL_LINEAR

    output_df[DEPENDENCE_DETECTION_STATUS_COL] = CONSTANT_BEATS_BOTH_STATUS

    output_df.loc[exp_chosen & ~linear_chosen, DEPENDENCE_DETECTION_STATUS_COL] = EXP_ONLY_STATUS
    output_df.loc[exp_chosen & linear_chosen, DEPENDENCE_DETECTION_STATUS_COL] = EXP_AND_LINEAR_STATUS
    output_df.loc[~exp_chosen & linear_chosen, DEPENDENCE_DETECTION_STATUS_COL] = LINEAR_ONLY_STATUS
    return output_df


def get_detection_status_color_map() -> dict[str, str]:
    return {
        EXP_ONLY_STATUS: "#2A9D8F",
        EXP_AND_LINEAR_STATUS: "#8E7CC3",
        LINEAR_ONLY_STATUS: "#E76F51",
        CONSTANT_BEATS_BOTH_STATUS: "#9E9E9E",
    }


# Summary tables

def build_count_table(counts: pd.Series, label_col: str) -> pd.DataFrame:
    total = int(counts.sum())
    return pd.DataFrame(
        {
            label_col: counts.index,
            "count": counts.values,
            "percentage": [
                round(100 * value / total, 2) if total > 0 else 0.0
                for value in counts.values
            ],
        }
    )


def build_aggregated_weights(model_summary_with_weights_df: pd.DataFrame) -> pd.Series:
    agg_weights = model_summary_with_weights_df.groupby(LABEL_FUNC_TYPE_COL)[AKAIKE_WEIGHT_COL].sum()
    return pd.Series(
        {
            model_label: float(agg_weights.get(model_label, 0.0))
            for model_label in MODEL_LABELS_WITH_EXP
        }
    )


def save_table(df: pd.DataFrame, out_file: Path, output_paths: List[str]) -> None:
    df.to_csv(out_file, index=False)
    output_paths.append(str(out_file))

def save_count_bar_and_pie(    counts: pd.Series,
    label_col: str,
    title: str,
    output_prefix: Path,
    color_map: dict[str, str],
    total_n: int,
    output_paths: List[str],
) -> None:
    # Save count table, bar plot, and pie plot

    counts_file = output_prefix.with_name(f"{output_prefix.name}_counts.csv")
    bar_file = output_prefix.with_name(f"{output_prefix.name}_bar.png")
    pie_file = output_prefix.with_name(f"{output_prefix.name}_pie.png")

    save_table(
        df=build_count_table(counts=counts, label_col=label_col),
        out_file=counts_file,
        output_paths=output_paths,
    )

    plot_bar_counts(
        counts=counts,
        title=title,
        xlabel=label_col,
        ylabel="Number of families",
        out_file=bar_file,
        rotation=25,
        color_map=color_map,
    )
    output_paths.append(str(bar_file))

    plot_pie_counts(
        counts=counts,
        title=title,
        out_file=pie_file,
        color_map=color_map,
        show_counts=True,
        total_n=total_n,
    )
    output_paths.append(str(pie_file))


# Main analysis

def run_model_selection_analysis(transition_label: str, core_table_path: Path, baseline_model_summary_table_path: Path) -> List[str]:
    output_dir = ANALYSIS_DIR / EXPONENTIAL_VS_LINEAR_SUBDIR / transition_label / MODEL_SELECTION_SUBDIR
    ensure_dir(output_dir)

    output_paths: List[str] = []

    core_df = pd.read_csv(core_table_path)
    baseline_model_summary_df = pd.read_csv(baseline_model_summary_table_path)

    # 1. Dependence detection categories from pairwise comparisons
    core_with_detection_df = add_dependence_detection_status(core_df)

    core_with_detection_file = output_dir / f"{transition_label}_core_table_with_dependence_detection.csv"
    save_table(core_with_detection_df, core_with_detection_file,output_paths)

    detection_order = (EXP_ONLY_STATUS, EXP_AND_LINEAR_STATUS, LINEAR_ONLY_STATUS, CONSTANT_BEATS_BOTH_STATUS)
    detection_counts = get_ordered_counts(core_with_detection_df[DEPENDENCE_DETECTION_STATUS_COL], detection_order)

    save_count_bar_and_pie(
        counts=detection_counts,
        label_col=DEPENDENCE_DETECTION_STATUS_COL,
        title=f"{transition_label}: EXP/LINEAR dependence detection",
        output_prefix=output_dir / f"{transition_label}_dependence_detection",
        color_map=get_detection_status_color_map(),
        total_n=len(core_with_detection_df),
        output_paths=output_paths,
    )

    # 2. Full model summary with EXP
    model_summary_with_exp_df = build_model_summary_with_exp(
        baseline_model_summary_df=baseline_model_summary_df,
        core_df=core_df,
        transition_label=transition_label,
    )

    model_summary_with_weights_df = add_akaike_weights(
        model_summary_df=model_summary_with_exp_df,
    )

    model_summary_with_weights_file = (
        output_dir / f"{transition_label}_model_summary_with_exp_and_weights.csv"
    )
    save_table(
        df=model_summary_with_weights_df,
        out_file=model_summary_with_weights_file,
        output_paths=output_paths,
    )

    # 3. Chosen model with EXP
    chosen_with_exp_df = build_chosen_model_table_with_exp(
        model_summary_with_weights_df=model_summary_with_weights_df,
    )

    chosen_with_exp_file = output_dir / f"{transition_label}_chosen_model_with_exp.csv"
    save_table(
        df=chosen_with_exp_df,
        out_file=chosen_with_exp_file,
        output_paths=output_paths,
    )

    chosen_counts = get_ordered_counts(
        chosen_with_exp_df[CHOSEN_FUNCTION_LABEL_COL],
        order=MODEL_LABELS_WITH_EXP,
    )

    save_count_bar_and_pie(
        counts=chosen_counts,
        label_col=CHOSEN_FUNCTION_LABEL_COL,
        title=f"{transition_label}: chosen model after adding EXP",
        output_prefix=output_dir / f"{transition_label}_chosen_model_with_exp",
        color_map=MODEL_COLOR_MAP,
        total_n=len(chosen_with_exp_df),
        output_paths=output_paths,
    )

    # 4. Aggregated Akaike weights with EXP
    agg_weights = build_aggregated_weights(
        model_summary_with_weights_df=model_summary_with_weights_df,
    )

    agg_weights_file = output_dir / f"{transition_label}_aggregated_akaike_weights_with_exp.csv"
    save_table(
        df=pd.DataFrame(
            {
                "model": agg_weights.index,
                "aggregated_akaike_weight": agg_weights.values,
            }
        ),
        out_file=agg_weights_file,
        output_paths=output_paths,
    )

    agg_weights_pie_file = output_dir / f"{transition_label}_aggregated_akaike_weights_with_exp_pie.png"

    plot_pie_counts(
        counts=agg_weights,
        title=f"{transition_label}: aggregated Akaike weights with EXP",
        out_file=agg_weights_pie_file,
        color_map=MODEL_COLOR_MAP,
        show_counts=False,
        total_n=len(chosen_with_exp_df),
    )
    output_paths.append(str(agg_weights_pie_file))

    return output_paths



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EXP-vs-LINEAR model-selection analysis.")
    parser.add_argument("--transition_label", type=str, required=True, choices=[LABEL_GAIN, LABEL_LOSS, LABEL_DUPL])
    parser.add_argument("--core_table_path", type=Path, required=True)
    parser.add_argument("--baseline_model_summary_table_path", type=Path, required=True)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    output_paths = run_model_selection_analysis(args.transition_label, args.core_table_path, args.baseline_model_summary_table_path)

    print("[✓] EXP-LINEAR model-selection analysis completed.")
    for output_path in output_paths:
        print(output_path)

    log_run(
        step="analysis",
        script=Path(__file__),
        params={
            "analysis_type": EXPONENTIAL_VS_LINEAR_SUBDIR,
            "transition_label": args.transition_label,
            "core_table_path": str(args.core_table_path),
            "baseline_model_summary_table_path": str(args.baseline_model_summary_table_path),
        },
        outputs=output_paths,
        description="Created model-selection tables and plots for EXP-vs-LINEAR branch.",
        notes=(
            "The script builds a full long-format model summary by concatenating the "
            "baseline models_summary_table with EXP rows from the core comparison table. "
            "It then computes delta AICc, Akaike weights, chosen models, count plots, "
            "and aggregated Akaike-weight plots including exponential."
        ),
        log_relative_path=(
            Path(BASELINE_MODELS_LABEL)
            / EXPONENTIAL_VS_LINEAR_SUBDIR
            / args.transition_label
            / MODEL_SELECTION_SUBDIR
            / f"{args.transition_label}_exp_lin_model_selection_analysis.log"
        ),
    )


if __name__ == "__main__":
    main()