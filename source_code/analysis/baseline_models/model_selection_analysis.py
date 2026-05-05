import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from source_code.analysis.analysis_constants import (
    BASELINE_ANALYSIS_DIR,
    BASELINE_FUNC_LABELS_NO_IGNORE,
    BASELINE_FUNC_LABELS_ORDERED,
    BASELINE_MODELS_LABEL,
    BEST_AKAIKE_WEIGHT_COL,
    CHOSEN_FUNCTION_LABEL_COL,
    CHOSEN_MODEL_SUFFIX,
    DELTA_BEST_VS_SECOND_COL,
    DELTA_SUPPORT_CLASS_COL,
    DELTA_SUPPORT_CLASS_COLOR_MAP,
    DELTA_SUPPORT_COL,
    LABEL_TRANSITIONS_ORDERED,
    MODEL_COLOR_MAP,
    MODEL_SELECTION_SUBDIR,
    WEIGHT_SUPPORT_CLASS_COL,
    WEIGHT_SUPPORT_CLASS_COLOR_MAP,
    WEIGHT_SUPPORT_COL,
)
from source_code.analysis.baseline_models.plot_utils import ensure_dir, plot_hist_by_class, plot_pie_counts
from source_code.logger import log_run


# -------- loading --------
def load_chosen_table(transition_label: str) -> pd.DataFrame:
    chosen_file: Path = BASELINE_ANALYSIS_DIR / transition_label / f"{transition_label}_{CHOSEN_MODEL_SUFFIX}"

    if not chosen_file.exists():
        raise FileNotFoundError(f"Missing chosen model table: {chosen_file}")

    return pd.read_csv(chosen_file)


# -------- summary helpers --------
def build_count_series(df: pd.DataFrame, labels_order: tuple[str, ...]) -> pd.Series:
    counts: pd.Series = df[CHOSEN_FUNCTION_LABEL_COL].value_counts()
    return pd.Series({label: int(counts.get(label, 0)) for label in labels_order})


def build_count_table(counts: pd.Series) -> pd.DataFrame:
    total: int = int(counts.sum())
    percentages: pd.Series = (100 * counts / total).round(2) if total > 0 else pd.Series(0.0, index=counts.index)

    return pd.DataFrame(
        {
            "model": counts.index,
            "count": counts.values,
            "percentage": percentages.values,
        }
    )


def build_support_count_table(df: pd.DataFrame, support_col: str) -> pd.DataFrame:
    support_df: pd.DataFrame = df[support_col].value_counts(dropna=False).rename("count").reset_index()
    support_df.columns = [support_col, "count"]
    return support_df


def build_aggregated_weights(df: pd.DataFrame) -> pd.Series:
    agg_weights: pd.Series = df.groupby(CHOSEN_FUNCTION_LABEL_COL)[BEST_AKAIKE_WEIGHT_COL].sum()
    return pd.Series({label: float(agg_weights.get(label, 0.0)) for label in BASELINE_FUNC_LABELS_NO_IGNORE})


def build_hist_box_lines(counts: pd.Series, header: str, threshold_line: str, label_texts: Dict[int, str]) -> List[str]:
    lines: List[str] = [header, threshold_line, "", "Counts:"]

    for class_value in [1, 2, 3, 4]:
        lines.append(f"{class_value} ({label_texts[class_value]}): {int(counts.get(class_value, 0))}")

    if int(counts.get(0, 0)) > 0:
        lines.append(f"0 (missing): {int(counts.get(0, 0))}")

    return lines


def save_table(df: pd.DataFrame, out_file: Path, output_paths: List[str]) -> None:
    df.to_csv(out_file, index=False)
    output_paths.append(str(out_file))


# -------- plotting helpers --------
def plot_model_selection_pies(
    transition_label: str,
    out_dir: Path,
    counts_all: pd.Series,
    counts_no_ignore: pd.Series,
    agg_weights: pd.Series,
    n_total: int,
    n_no_ignore: int,
    output_paths: List[str],
) -> None:
    pie_all_file = out_dir / f"{transition_label}_chosen_model_pie.png"
    pie_no_ignore_file = out_dir / f"{transition_label}_chosen_model_pie_no_ignore.png"
    weight_pie_file = out_dir / f"{transition_label}_aggregated_best_akaike_weight_pie_no_ignore.png"

    # ---- counts (all) ----
    plot_pie_counts(
        counts=counts_all,
        title=f"{transition_label}: chosen model distribution",
        out_file=pie_all_file,
        color_map=MODEL_COLOR_MAP,
        show_counts=True,
        total_n=n_total,
    )

    # ---- counts (no ignore) ----
    plot_pie_counts(
        counts=counts_no_ignore,
        title=f"{transition_label}: chosen model distribution (constant vs linear)",
        out_file=pie_no_ignore_file,
        color_map=MODEL_COLOR_MAP,
        show_counts=True,
        total_n=n_no_ignore,
    )

    # ---- akaike weights ----
    plot_pie_counts(
        counts=agg_weights,
        title=f"{transition_label}: aggregated best Akaike weights (constant vs linear)",
        out_file=weight_pie_file,
        color_map=MODEL_COLOR_MAP,
        show_counts=False,
        total_n=n_no_ignore,  # IMPORTANT: number of families
    )

    output_paths.extend([
        str(pie_all_file),
        str(pie_no_ignore_file),
        str(weight_pie_file),
    ])

def plot_threshold_histograms(
    transition_label: str,
    out_dir: Path,
    chosen_no_ignore_df: pd.DataFrame,
    output_paths: List[str],
) -> None:
    delta_hist_file: Path = out_dir / f"{transition_label}_delta_best_vs_second_hist_no_ignore.png"
    weight_hist_file: Path = out_dir / f"{transition_label}_best_akaike_weight_hist_no_ignore.png"

    delta_counts: pd.Series = chosen_no_ignore_df[DELTA_SUPPORT_CLASS_COL].value_counts()

    delta_summary_rows: List[tuple[int, str, int]] = [
        (1, "<=2", int(delta_counts.get(1, 0))),
        (2, "2-4", int(delta_counts.get(2, 0))),
        (3, "4-10", int(delta_counts.get(3, 0))),
        (4, ">10", int(delta_counts.get(4, 0))),
    ]

    weight_counts: pd.Series = chosen_no_ignore_df[WEIGHT_SUPPORT_CLASS_COL].value_counts()

    weight_summary_rows: List[tuple[int, str, int]] = [
        (1, "<0.6", int(weight_counts.get(1, 0))),
        (2, "0.6-0.8", int(weight_counts.get(2, 0))),
        (3, "0.8-0.95", int(weight_counts.get(3, 0))),
        (4, ">0.95", int(weight_counts.get(4, 0))),
    ]

    plot_hist_by_class(
        df=chosen_no_ignore_df,
        value_col=DELTA_BEST_VS_SECOND_COL,
        class_col=DELTA_SUPPORT_CLASS_COL,
        title=f"{transition_label}: delta AICc between best and second-best (constant vs linear)",
        xlabel="Delta AICc",
        ylabel="Number of families",
        out_file=delta_hist_file,
        class_order=[1, 2, 3, 4, 0],
        class_color_map=DELTA_SUPPORT_CLASS_COLOR_MAP,
        bin_size=0.5,
        summary_rows=delta_summary_rows,
    )

    plot_hist_by_class(
        df=chosen_no_ignore_df,
        value_col=BEST_AKAIKE_WEIGHT_COL,
        class_col=WEIGHT_SUPPORT_CLASS_COL,
        title=f"{transition_label}: best Akaike weight (constant vs linear)",
        xlabel="Best Akaike weight",
        ylabel="Number of families",
        out_file=weight_hist_file,
        class_order=[1, 2, 3, 4, 0],
        class_color_map=WEIGHT_SUPPORT_CLASS_COLOR_MAP,
        bin_size=0.025,
        summary_rows=weight_summary_rows,
    )

    output_paths.extend([str(delta_hist_file), str(weight_hist_file)])


# -------- main analysis --------
def run_transition(transition_label: str) -> Dict[str, object]:
    chosen_df: pd.DataFrame = load_chosen_table(transition_label)
    out_dir: Path = BASELINE_ANALYSIS_DIR / transition_label / MODEL_SELECTION_SUBDIR
    output_paths: List[str] = []

    ensure_dir(out_dir)

    chosen_no_ignore_df: pd.DataFrame = chosen_df[chosen_df[CHOSEN_FUNCTION_LABEL_COL].isin(BASELINE_FUNC_LABELS_NO_IGNORE)].copy()
    chosen_no_ignore_df = chosen_no_ignore_df.reset_index(drop=True)

    counts_all: pd.Series = build_count_series(chosen_df, BASELINE_FUNC_LABELS_ORDERED)
    counts_no_ignore: pd.Series = build_count_series(chosen_no_ignore_df, BASELINE_FUNC_LABELS_NO_IGNORE)
    agg_weights: pd.Series = build_aggregated_weights(chosen_no_ignore_df)

    counts_all_file: Path = out_dir / f"{transition_label}_chosen_model_counts.csv"
    counts_no_ignore_file: Path = out_dir / f"{transition_label}_chosen_model_counts_no_ignore.csv"
    delta_support_file: Path = out_dir / f"{transition_label}_delta_support_counts_no_ignore.csv"
    weight_support_file: Path = out_dir / f"{transition_label}_weight_support_counts_no_ignore.csv"

    save_table(build_count_table(counts_all), counts_all_file, output_paths)
    save_table(build_count_table(counts_no_ignore), counts_no_ignore_file, output_paths)
    save_table(build_support_count_table(chosen_no_ignore_df, DELTA_SUPPORT_COL), delta_support_file, output_paths)
    save_table(build_support_count_table(chosen_no_ignore_df, WEIGHT_SUPPORT_COL), weight_support_file, output_paths)

    plot_model_selection_pies(
        transition_label=transition_label,
        out_dir=out_dir,
        counts_all=counts_all,
        counts_no_ignore=counts_no_ignore,
        agg_weights=agg_weights,
        n_total=len(chosen_df),
        n_no_ignore=len(chosen_no_ignore_df),
        output_paths=output_paths,
    )

    plot_threshold_histograms(
        transition_label=transition_label,
        out_dir=out_dir,
        chosen_no_ignore_df=chosen_no_ignore_df,
        output_paths=output_paths,
    )

    return {
        "transition": transition_label,
        "n_families_total": len(chosen_df),
        "n_families_constant_vs_linear": len(chosen_no_ignore_df),
        "chosen_counts_all": counts_all.to_dict(),
        "chosen_counts_no_ignore": counts_no_ignore.to_dict(),
        "aggregated_best_akaike_weights_no_ignore": {k: round(v, 6) for k, v in agg_weights.to_dict().items()},
        "outputs": output_paths,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transition", required=True, type=str, choices=list(LABEL_TRANSITIONS_ORDERED))
    return parser.parse_args()


def main() -> None:
    args: argparse.Namespace = parse_args()
    result: Dict[str, object] = run_transition(args.transition)

    log_run(
        step="analysis",
        script=Path(__file__),
        params={
            "transition": result["transition"],
            "n_families_total": result["n_families_total"],
            "n_families_constant_vs_linear": result["n_families_constant_vs_linear"],
            "chosen_counts_all": result["chosen_counts_all"],
            "chosen_counts_no_ignore": result["chosen_counts_no_ignore"],
            "aggregated_best_akaike_weights_no_ignore": result["aggregated_best_akaike_weights_no_ignore"],
        },
        outputs=result["outputs"],
        description=f"Stage 1 model selection analysis for transition '{result['transition']}'",
        notes="Created count tables, support count tables, chosen-model pies, aggregated Akaike-weight pie, and threshold-colored histograms.",
        log_relative_path=Path(BASELINE_MODELS_LABEL) / f"{result['transition']}.log",
    )


if __name__ == "__main__":
    main()