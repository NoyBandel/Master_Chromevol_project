import argparse
from typing import Callable, Optional

import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, ttest_ind, ttest_rel

from source_code.analysis.analysis_constants import *
from source_code.analysis.plot_utils import (
    build_stats,
    ensure_dir,
    get_ordered_counts,
    plot_bar_counts,
    plot_box_by_group,
    plot_hist_overlay,
    plot_scatter_by_group,
    plot_single_hist,
    write_text,
)


MODEL_ORDER = list(BASELINE_FUNC_LABELS_ORDERED)
EPSILON = 1e-9


# Progress / logging helpers
def log(msg: str) -> None:
    print(f"[INFO] {msg}")


def log_warn(msg: str) -> None:
    print(f"[WARNING] {msg}")


def log_done(msg: str) -> None:
    print(f"[DONE] {msg}")


def write_progress_file(out_dir: Path, lines: List[str]) -> None:
    write_text("\n".join(lines), out_dir / "run_progress.txt")


# Loading/validation
def get_summary_table_file(transition: str) -> Path:
    return BASELINE_ANALYSIS_DIR / transition / f"{transition}_baseline_models_summary_table.csv"


def get_chosen_model_table_file(transition: str) -> Path:
    return BASELINE_ANALYSIS_DIR / transition / f"{transition}_chosen_model_table.csv"


def validate_columns(df: pd.DataFrame, required_cols: List[str]) -> None:
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")


def load_input_tables(transition: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_file = get_summary_table_file(transition)
    chosen_file = get_chosen_model_table_file(transition)

    if not summary_file.exists():
        raise FileNotFoundError(f"Missing summary table: {summary_file}")

    if not chosen_file.exists():
        raise FileNotFoundError(f"Missing chosen model table: {chosen_file}")

    summary_df = pd.read_csv(summary_file)
    chosen_df = pd.read_csv(chosen_file)
    return summary_df, chosen_df


def infer_constant_and_linear_event_columns(summary_df: pd.DataFrame) -> pd.DataFrame:
    required_cols = [FAMILY_NAME_COL, LABEL_FUNC_TYPE_COL, NUM_OF_EVENTS_COL]
    validate_columns(summary_df, required_cols)

    constant_df = (
        summary_df.loc[summary_df[LABEL_FUNC_TYPE_COL] == LABEL_CONSTANT, [FAMILY_NAME_COL, NUM_OF_EVENTS_COL]]
        .rename(columns={NUM_OF_EVENTS_COL: CONST_EVENTS_COL})
        .copy()
    )

    linear_df = (
        summary_df.loc[summary_df[LABEL_FUNC_TYPE_COL] == LABEL_LINEAR, [FAMILY_NAME_COL, NUM_OF_EVENTS_COL]]
        .rename(columns={NUM_OF_EVENTS_COL: LINEAR_EVENTS_COL})
        .copy()
    )

    return constant_df.merge(linear_df, on=FAMILY_NAME_COL, how="outer")


def build_analysis_df(summary_df: pd.DataFrame, chosen_df: pd.DataFrame) -> pd.DataFrame:
    event_compare_df = infer_constant_and_linear_event_columns(summary_df)

    df = chosen_df.merge(event_compare_df, on=FAMILY_NAME_COL, how="left")

    df[CONST_EVENTS_COL] = pd.to_numeric(df[CONST_EVENTS_COL], errors="coerce")
    df[LINEAR_EVENTS_COL] = pd.to_numeric(df[LINEAR_EVENTS_COL], errors="coerce")

    df[EVENTS_DIFF_COL] = df[LINEAR_EVENTS_COL] - df[CONST_EVENTS_COL]
    df[EVENTS_REL_DIFF_COL] = df[EVENTS_DIFF_COL] / (df[CONST_EVENTS_COL] + EPSILON)

    return df


# Summary helpers
def format_stats_dict(stats_dict: dict) -> str:
    return (
        f"n={stats_dict['n']}, "
        f"mean={stats_dict['mean']:.4f}, "
        f"median={stats_dict['median']:.4f}, "
        f"std={stats_dict['std']:.4f}, "
        f"min={stats_dict['min']:.4f}, "
        f"q25={stats_dict['q25']:.4f}, "
        f"q75={stats_dict['q75']:.4f}, "
        f"max={stats_dict['max']:.4f}"
    )


def corr_summary_line(df: pd.DataFrame, x_col: str, y_col: str, label: str) -> str:
    sub = df[[x_col, y_col]].copy()
    sub[x_col] = pd.to_numeric(sub[x_col], errors="coerce")
    sub[y_col] = pd.to_numeric(sub[y_col], errors="coerce")
    sub = sub.dropna()

    if len(sub) < 3:
        return f"{label}: not enough data"

    spearman_r = sub[x_col].corr(sub[y_col], method="spearman")
    pearson_r = sub[x_col].corr(sub[y_col], method="pearson")

    return f"{label}: n={len(sub)}, spearman={spearman_r:.3f}, pearson={pearson_r:.3f}"


def run_paired_ttest(df: pd.DataFrame) -> Optional[str]:
    sub_df = df[[CONST_EVENTS_COL, LINEAR_EVENTS_COL]].copy().dropna()
    if sub_df.empty:
        return None

    t_stat, p_val = ttest_rel(sub_df[LINEAR_EVENTS_COL], sub_df[CONST_EVENTS_COL])
    diff_stats = build_stats(sub_df[LINEAR_EVENTS_COL] - sub_df[CONST_EVENTS_COL])

    lines = [
        "Paired t-test: linear expected events vs constant expected events",
        f"n paired families = {len(sub_df)}",
        f"t-statistic = {t_stat:.6f}",
        f"p-value = {p_val:.6g}",
        f"diff (linear - constant): {format_stats_dict(diff_stats)}",
    ]
    return "\n".join(lines)


def run_diff_by_chosen_model_ttest(df: pd.DataFrame) -> Optional[str]:
    sub_df = df[[EVENTS_DIFF_COL, CHOSEN_FUNCTION_LABEL_COL]].copy().dropna()

    linear_winners = sub_df.loc[sub_df[CHOSEN_FUNCTION_LABEL_COL] == LABEL_LINEAR, EVENTS_DIFF_COL]
    constant_winners = sub_df.loc[sub_df[CHOSEN_FUNCTION_LABEL_COL] == LABEL_CONSTANT, EVENTS_DIFF_COL]

    if linear_winners.empty or constant_winners.empty:
        return None

    t_stat, p_val = ttest_ind(linear_winners, constant_winners, equal_var=False)

    lines = [
        "Welch t-test: diff(linear - constant) in linear winners vs constant winners",
        f"linear winners n = {len(linear_winners)}",
        f"constant winners n = {len(constant_winners)}",
        f"t-statistic = {t_stat:.6f}",
        f"p-value = {p_val:.6g}",
        f"linear winners diff stats: {format_stats_dict(build_stats(linear_winners))}",
        f"constant winners diff stats: {format_stats_dict(build_stats(constant_winners))}",
    ]
    return "\n".join(lines)


def run_feature_kruskal_test(df: pd.DataFrame, feature_col: str) -> Optional[str]:
    sub_df = df[[feature_col, CHOSEN_FUNCTION_LABEL_COL]].copy()
    sub_df[feature_col] = pd.to_numeric(sub_df[feature_col], errors="coerce")
    sub_df = sub_df.dropna()

    groups = []
    used_labels = []

    for label in MODEL_ORDER:
        vals = sub_df.loc[sub_df[CHOSEN_FUNCTION_LABEL_COL] == label, feature_col]
        if not vals.empty:
            groups.append(vals)
            used_labels.append(label)

    if len(groups) < 2:
        return None

    stat, p_val = kruskal(*groups)

    lines = [
        f"Kruskal-Wallis test for feature: {feature_col}",
        f"groups tested = {used_labels}",
        f"H-statistic = {stat:.6f}",
        f"p-value = {p_val:.6g}",
    ]

    for label in used_labels:
        vals = sub_df.loc[sub_df[CHOSEN_FUNCTION_LABEL_COL] == label, feature_col]
        lines.append(f"{label}: {format_stats_dict(build_stats(vals))}")

    return "\n".join(lines)


def run_pairwise_mannwhitney_tests(df: pd.DataFrame, feature_col: str) -> List[str]:
    sub_df = df[[feature_col, CHOSEN_FUNCTION_LABEL_COL]].copy()
    sub_df[feature_col] = pd.to_numeric(sub_df[feature_col], errors="coerce")
    sub_df = sub_df.dropna()

    pairs = [
        (LABEL_IGNORE, LABEL_CONSTANT),
        (LABEL_IGNORE, LABEL_LINEAR),
        (LABEL_CONSTANT, LABEL_LINEAR),
    ]

    out_lines = []

    for g1, g2 in pairs:
        vals1 = sub_df.loc[sub_df[CHOSEN_FUNCTION_LABEL_COL] == g1, feature_col]
        vals2 = sub_df.loc[sub_df[CHOSEN_FUNCTION_LABEL_COL] == g2, feature_col]

        if vals1.empty or vals2.empty:
            continue

        stat, p_val = mannwhitneyu(vals1, vals2, alternative="two-sided")
        out_lines.append(
            f"Mann-Whitney U for {feature_col}: {g1} vs {g2} | "
            f"n1={len(vals1)}, n2={len(vals2)}, U={stat:.6f}, p={p_val:.6g}"
        )

    return out_lines


def build_summary_text(df: pd.DataFrame, transition: str) -> str:
    counts = get_ordered_counts(df[CHOSEN_FUNCTION_LABEL_COL], MODEL_ORDER)

    feature_cols = [
        FAMILY_SIZE_COL,
        DIFF_COL,
        MIN_CHROM_COL,
        MAX_CHROM_COL,
        ROOT_CHROM_NUM_COL,
        CONST_EVENTS_COL,
        LINEAR_EVENTS_COL,
        EVENTS_DIFF_COL,
        EVENTS_REL_DIFF_COL,
    ]

    lines = [
        f"Transition: {transition}",
        "=" * 70,
        "",
        "Chosen model counts:",
        counts.to_string(),
        "",
    ]

    paired_ttest_text = run_paired_ttest(df)
    if paired_ttest_text is not None:
        lines.append(paired_ttest_text)
        lines.append("")

    diff_by_model_text = run_diff_by_chosen_model_ttest(df)
    if diff_by_model_text is not None:
        lines.append(diff_by_model_text)
        lines.append("")

    lines.extend([
        "Correlations with linear - constant expected events:",
        corr_summary_line(df, ROOT_CHROM_NUM_COL, EVENTS_DIFF_COL, "root chrom num vs events diff"),
    ])

    if FAMILY_SIZE_COL in df.columns:
        lines.append(corr_summary_line(df, FAMILY_SIZE_COL, EVENTS_DIFF_COL, "family size vs events diff"))

    if DIFF_COL in df.columns:
        lines.append(corr_summary_line(df, DIFF_COL, EVENTS_DIFF_COL, "chrom range vs events diff"))

    lines.append("")

    for feature_col in feature_cols:
        if feature_col not in df.columns:
            continue

        kw_text = run_feature_kruskal_test(df, feature_col)
        if kw_text is not None:
            lines.append(kw_text)
            lines.extend(run_pairwise_mannwhitney_tests(df, feature_col))
            lines.append("")

    return "\n".join(lines)


# Plots
def plot_chosen_model_counts(df: pd.DataFrame, transition: str, out_dir: Path) -> Path:
    out_file = out_dir / f"{transition}_chosen_model_counts.png"
    counts = get_ordered_counts(df[CHOSEN_FUNCTION_LABEL_COL], MODEL_ORDER)

    plot_bar_counts(
        counts=counts,
        title=f"{transition}: chosen model counts",
        xlabel="Chosen model",
        ylabel="Number of families",
        out_file=out_file,
        color_map=MODEL_COLOR_MAP,
    )
    return out_file


def plot_expected_events_box(df: pd.DataFrame, transition: str, out_dir: Path) -> Path:
    out_file = out_dir / f"{transition}_expected_events_boxplot.png"
    plot_box_by_group(
        df=df,
        value_col=NUM_OF_EVENTS_COL,
        group_col=CHOSEN_FUNCTION_LABEL_COL,
        title=f"{transition}: expected events by chosen model",
        xlabel="Chosen model",
        ylabel=NUM_OF_EVENTS_COL,
        out_file=out_file,
        group_order=MODEL_ORDER,
    )
    return out_file


def plot_expected_events_hist_constant_vs_linear(df: pd.DataFrame, transition: str, out_dir: Path) -> Path:
    out_file = out_dir / f"{transition}_expected_events_hist_constant_vs_linear_density.png"

    plot_hist_overlay(
        series_map={
            LABEL_CONSTANT: df[CONST_EVENTS_COL],
            LABEL_LINEAR: df[LINEAR_EVENTS_COL],
        },
        title=f"{transition}: expected events, constant vs linear",
        xlabel="Expected number of events",
        ylabel="Density",
        out_file=out_file,
        bins=20,
        density=True,
        color_map={
            LABEL_CONSTANT: MODEL_COLOR_MAP[LABEL_CONSTANT],
            LABEL_LINEAR: MODEL_COLOR_MAP[LABEL_LINEAR],
        },
    )
    return out_file


def plot_expected_events_diff_hist(df: pd.DataFrame, transition: str, out_dir: Path) -> Path:
    out_file = out_dir / f"{transition}_linear_minus_constant_events_hist.png"

    plot_single_hist(
        series=df[EVENTS_DIFF_COL],
        title=f"{transition}: linear - constant expected events",
        xlabel="Linear - constant expected events",
        ylabel="Number of families",
        out_file=out_file,
        bins=20,
        color=MODEL_COLOR_MAP[LABEL_LINEAR],
        vline_x=0.0,
    )
    return out_file


def plot_expected_events_diff_box_by_chosen_model(df: pd.DataFrame, transition: str, out_dir: Path) -> Path:
    out_file = out_dir / f"{transition}_linear_minus_constant_events_box_by_chosen_model.png"

    plot_box_by_group(
        df=df,
        value_col=EVENTS_DIFF_COL,
        group_col=CHOSEN_FUNCTION_LABEL_COL,
        title=f"{transition}: linear - constant expected events by chosen model",
        xlabel="Chosen model",
        ylabel="Linear - constant expected events",
        out_file=out_file,
        group_order=MODEL_ORDER,
    )
    return out_file


def plot_events_vs_root(df: pd.DataFrame, transition: str, out_dir: Path) -> Path:
    out_file = out_dir / f"{transition}_events_vs_root_chrom_num.png"
    plot_scatter_by_group(
        df=df,
        x_col=ROOT_CHROM_NUM_COL,
        y_col=NUM_OF_EVENTS_COL,
        group_col=CHOSEN_FUNCTION_LABEL_COL,
        title=f"{transition}: expected events vs root chrom num",
        xlabel=ROOT_CHROM_NUM_COL,
        ylabel=NUM_OF_EVENTS_COL,
        out_file=out_file,
        color_map=MODEL_COLOR_MAP,
    )
    return out_file


def plot_events_diff_vs_root(df: pd.DataFrame, transition: str, out_dir: Path) -> Path:
    out_file = out_dir / f"{transition}_events_diff_vs_root_chrom_num.png"
    plot_scatter_by_group(
        df=df,
        x_col=ROOT_CHROM_NUM_COL,
        y_col=EVENTS_DIFF_COL,
        group_col=CHOSEN_FUNCTION_LABEL_COL,
        title=f"{transition}: linear - constant expected events vs root chrom num",
        xlabel=ROOT_CHROM_NUM_COL,
        ylabel=EVENTS_DIFF_COL,
        out_file=out_file,
        color_map=MODEL_COLOR_MAP,
    )
    return out_file


def plot_feature_boxplots_by_chosen_model(df: pd.DataFrame, transition: str, out_dir: Path) -> List[Path]:
    feature_cols = [
        FAMILY_SIZE_COL,
        DIFF_COL,
        MIN_CHROM_COL,
        MAX_CHROM_COL,
        ROOT_CHROM_NUM_COL,
        CONST_EVENTS_COL,
        LINEAR_EVENTS_COL,
        EVENTS_DIFF_COL,
        EVENTS_REL_DIFF_COL,
    ]

    output_files = []

    for feature_col in feature_cols:
        if feature_col not in df.columns:
            continue

        out_file = out_dir / f"{transition}_{feature_col}_box_by_chosen_model.png"
        plot_box_by_group(
            df=df,
            value_col=feature_col,
            group_col=CHOSEN_FUNCTION_LABEL_COL,
            title=f"{transition}: {feature_col} by chosen model",
            xlabel="Chosen model",
            ylabel=feature_col,
            out_file=out_file,
            group_order=MODEL_ORDER,
        )
        output_files.append(out_file)

    return output_files


def plot_events_diff_vs_family_size(df: pd.DataFrame, transition: str, out_dir: Path) -> Optional[Path]:
    if FAMILY_SIZE_COL not in df.columns:
        return None

    out_file = out_dir / f"{transition}_events_diff_vs_family_size.png"
    plot_scatter_by_group(
        df=df,
        x_col=FAMILY_SIZE_COL,
        y_col=EVENTS_DIFF_COL,
        group_col=CHOSEN_FUNCTION_LABEL_COL,
        title=f"{transition}: linear - constant expected events vs family size",
        xlabel=FAMILY_SIZE_COL,
        ylabel=EVENTS_DIFF_COL,
        out_file=out_file,
        color_map=MODEL_COLOR_MAP,
    )
    return out_file


def plot_events_diff_vs_chrom_range(df: pd.DataFrame, transition: str, out_dir: Path) -> Optional[Path]:
    if DIFF_COL not in df.columns:
        return None

    out_file = out_dir / f"{transition}_events_diff_vs_chrom_range.png"
    plot_scatter_by_group(
        df=df,
        x_col=DIFF_COL,
        y_col=EVENTS_DIFF_COL,
        group_col=CHOSEN_FUNCTION_LABEL_COL,
        title=f"{transition}: linear - constant expected events vs chrom range",
        xlabel=DIFF_COL,
        ylabel=EVENTS_DIFF_COL,
        out_file=out_file,
        color_map=MODEL_COLOR_MAP,
    )
    return out_file


def run_step(
    step_name: str,
    step_func: Callable[[], Path | None],
    progress_lines: List[str],
) -> None:
    log(f"Starting: {step_name}")
    progress_lines.append(f"STARTED: {step_name}")

    try:
        out_file = step_func()

        if out_file is None:
            log_warn(f"Skipped: {step_name}")
            progress_lines.append(f"SKIPPED: {step_name}")
            return

        if out_file.exists():
            log_done(f"Created: {out_file.name}")
            progress_lines.append(f"DONE: {step_name} -> {out_file.name}")
        else:
            log_warn(f"Finished but output file was not found: {out_file}")
            progress_lines.append(f"WARNING: {step_name} finished but file missing -> {out_file}")

    except Exception as e:
        log_warn(f"Failed: {step_name} | {e}")
        progress_lines.append(f"FAILED: {step_name} -> {e}")
        raise


# Run
def run_transition(transition: str) -> None:
    transition_dir = BASELINE_ANALYSIS_DIR / transition
    out_dir = transition_dir / "descriptive"
    ensure_dir(out_dir)

    progress_lines: List[str] = [f"Transition: {transition}"]

    summary_file = get_summary_table_file(transition)
    chosen_file = get_chosen_model_table_file(transition)

    progress_lines.append(f"Summary input file: {summary_file}")
    progress_lines.append(f"Chosen input file: {chosen_file}")

    if not summary_file.exists():
        msg = f"Missing summary input file: {summary_file}"
        progress_lines.append(f"FAILED: {msg}")
        write_progress_file(out_dir, progress_lines)
        raise FileNotFoundError(msg)

    if not chosen_file.exists():
        msg = f"Missing chosen input file: {chosen_file}"
        progress_lines.append(f"FAILED: {msg}")
        write_progress_file(out_dir, progress_lines)
        raise FileNotFoundError(msg)

    log(f"Loading summary table: {summary_file}")
    log(f"Loading chosen table: {chosen_file}")

    summary_df, chosen_df = load_input_tables(transition)
    df = build_analysis_df(summary_df, chosen_df)

    progress_lines.append(f"Loaded rows: {len(df)}")
    progress_lines.append(f"Loaded columns: {len(df.columns)}")

    required_cols = [
        FAMILY_NAME_COL,
        CHOSEN_FUNCTION_LABEL_COL,
        ROOT_CHROM_NUM_COL,
        CONST_EVENTS_COL,
        LINEAR_EVENTS_COL,
        EVENTS_DIFF_COL,
    ]
    validate_columns(df, required_cols)
    progress_lines.append(f"Required columns OK: {required_cols}")

    write_progress_file(out_dir, progress_lines)

    run_step(
        "chosen model counts",
        lambda: plot_chosen_model_counts(df, transition, out_dir),
        progress_lines,
    )
    write_progress_file(out_dir, progress_lines)

    run_step(
        "expected events boxplot",
        lambda: plot_expected_events_box(df, transition, out_dir),
        progress_lines,
    )
    write_progress_file(out_dir, progress_lines)

    run_step(
        "expected events histogram constant vs linear density",
        lambda: plot_expected_events_hist_constant_vs_linear(df, transition, out_dir),
        progress_lines,
    )
    write_progress_file(out_dir, progress_lines)

    run_step(
        "linear minus constant events histogram",
        lambda: plot_expected_events_diff_hist(df, transition, out_dir),
        progress_lines,
    )
    write_progress_file(out_dir, progress_lines)

    run_step(
        "linear minus constant events boxplot by chosen model",
        lambda: plot_expected_events_diff_box_by_chosen_model(df, transition, out_dir),
        progress_lines,
    )
    write_progress_file(out_dir, progress_lines)

    run_step(
        "expected events vs root chrom num",
        lambda: plot_events_vs_root(df, transition, out_dir),
        progress_lines,
    )
    write_progress_file(out_dir, progress_lines)

    run_step(
        "events diff vs root chrom num",
        lambda: plot_events_diff_vs_root(df, transition, out_dir),
        progress_lines,
    )
    write_progress_file(out_dir, progress_lines)

    run_step(
        "events diff vs family size",
        lambda: plot_events_diff_vs_family_size(df, transition, out_dir),
        progress_lines,
    )
    write_progress_file(out_dir, progress_lines)

    run_step(
        "events diff vs chrom range",
        lambda: plot_events_diff_vs_chrom_range(df, transition, out_dir),
        progress_lines,
    )
    write_progress_file(out_dir, progress_lines)

    log("Starting: feature boxplots by chosen model")
    progress_lines.append("STARTED: feature boxplots by chosen model")
    feature_boxplot_files = plot_feature_boxplots_by_chosen_model(df, transition, out_dir)
    progress_lines.append(f"DONE: feature boxplots by chosen model -> {len(feature_boxplot_files)} files")
    log_done(f"Created {len(feature_boxplot_files)} feature boxplot files")
    write_progress_file(out_dir, progress_lines)

    log("Building summary text")
    summary = build_summary_text(df, transition)
    summary_out_file = out_dir / f"{transition}_descriptive_summary.txt"
    write_text(summary, summary_out_file)

    if summary_out_file.exists():
        progress_lines.append(f"DONE: summary -> {summary_out_file.name}")
        log_done(f"Created: {summary_out_file.name}")
    else:
        progress_lines.append(f"WARNING: summary file missing -> {summary_out_file}")
        log_warn(f"Summary file was not found after writing: {summary_out_file}")

    analysis_table_file = out_dir / f"{transition}_descriptive_analysis_table.csv"
    df.to_csv(analysis_table_file, index=False)

    if analysis_table_file.exists():
        progress_lines.append(f"DONE: analysis table -> {analysis_table_file.name}")
        log_done(f"Created: {analysis_table_file.name}")
    else:
        progress_lines.append(f"WARNING: analysis table missing -> {analysis_table_file}")
        log_warn(f"Analysis table was not found after writing: {analysis_table_file}")

    write_progress_file(out_dir, progress_lines)
    log_done(f"Finished transition: {transition}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transition", required=True, type=str, choices=list(LABEL_TRANSITIONS_ORDERED))
    return parser.parse_args()


def main():
    args = parse_args()
    run_transition(args.transition)


if __name__ == "__main__":
    main()