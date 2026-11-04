import argparse
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
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


IGNORE_MODE_INCLUDE = "include"
IGNORE_MODE_EXCLUDE = "exclude"
IGNORE_MODE_CHOICES = (IGNORE_MODE_INCLUDE, IGNORE_MODE_EXCLUDE)

MODEL_LABELS_WITH_EXP_AND_IGNORE: tuple[str, ...] = (
    LABEL_CONSTANT,
    LABEL_LINEAR,
    LABEL_EXP,
    LABEL_IGNORE,
)

MODEL_LABELS_WITH_EXP_NO_IGNORE: tuple[str, ...] = (
    LABEL_CONSTANT,
    LABEL_LINEAR,
    LABEL_EXP,
)

# Biological interpretation labels used in plots/tables.
NO_DETECTED_DEPENDENCE_STATUS = "No detected chromosome-number dependence"
LINEAR_ONLY_DEPENDENCE_STATUS = "Dependence supported by linear form only"
EXP_ONLY_DEPENDENCE_STATUS = "Dependence supported by exponential form only"
BOTH_FORMS_DEPENDENCE_STATUS = "Dependence supported by both functional forms"
TRANSITION_IGNORED_STATUS = "Transition ignored"


DEPENDENCE_STATUS_DISPLAY_LABELS = {
    NO_DETECTED_DEPENDENCE_STATUS: "No detected dependence",
    LINEAR_ONLY_DEPENDENCE_STATUS: "Linear dependence only",
    EXP_ONLY_DEPENDENCE_STATUS: "Exponential dependence only",
    BOTH_FORMS_DEPENDENCE_STATUS: "Both functional forms",
    TRANSITION_IGNORED_STATUS: "Transition ignored",
}

TRANSITION_DISPLAY_NAMES = {
    LABEL_GAIN: "Chromosome Gain",
    LABEL_LOSS: "Chromosome Loss",
    LABEL_DUPL: "Whole-Genome Duplication",
}


def get_transition_display_name(transition_label: str) -> str:
    return TRANSITION_DISPLAY_NAMES.get(
        transition_label,
        transition_label.replace("_", " ").title(),
    )


def get_model_selection_title(transition_label: str, ignore_mode: str) -> str:
    transition_name = get_transition_display_name(transition_label)
    model_scope = (
        "Constant, Linear, Exponential, and Ignore Models"
        if ignore_mode == IGNORE_MODE_INCLUDE
        else "Constant, Linear, and Exponential Models"
    )
    return f"{transition_name}: Best-Supported Rate Model\n{model_scope}"


def get_dependence_title(transition_label: str, ignore_mode: str) -> str:
    transition_name = get_transition_display_name(transition_label)
    scope = (
        "Including the Ignore Model"
        if ignore_mode == IGNORE_MODE_INCLUDE
        else "Among Constant, Linear, and Exponential Models"
    )
    return (
        f"{transition_name}: Evidence for Chromosome-Number Dependence\n"
        f"{scope}"
    )


def get_akaike_title(transition_label: str, ignore_mode: str) -> str:
    transition_name = get_transition_display_name(transition_label)
    scope = (
        "Including the Ignore Model"
        if ignore_mode == IGNORE_MODE_INCLUDE
        else "Constant, Linear, and Exponential Models"
    )
    return f"{transition_name}: Aggregated Akaike Model Support\n{scope}"


def plot_dependence_pie_with_legend(
    counts: pd.Series,
    title: str,
    out_file: Path,
    color_map: dict[str, str],
) -> None:
    """
    Plot dependence-detection proportions with short category names in a
    legend rather than around the pie.

    Only non-zero categories are displayed. Slice labels show percentage and
    family count; the biological interpretation is shown in the side legend.
    """
    nonzero_counts = counts[counts > 0]
    if nonzero_counts.empty:
        raise ValueError("Cannot plot dependence pie: all category counts are zero.")

    values = nonzero_counts.to_numpy(dtype=float)
    labels = [
        DEPENDENCE_STATUS_DISPLAY_LABELS.get(status, str(status))
        for status in nonzero_counts.index
    ]
    colors = [color_map.get(status) for status in nonzero_counts.index]

    total = int(values.sum())

    def autopct_with_count(pct: float) -> str:
        count = int(round(pct * total / 100.0))
        return f"{pct:.1f}%\n(n={count})"

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    wedges, _, _ = ax.pie(
        values,
        labels=None,
        colors=colors,
        autopct=autopct_with_count,
        startangle=90,
        counterclock=False,
        wedgeprops={"edgecolor": "white", "linewidth": 1.2},
        textprops={"fontsize": 10},
    )

    legend_labels = [
        f"{label} (n={int(count)})"
        for label, count in zip(labels, nonzero_counts.values)
    ]
    ax.legend(
        wedges,
        legend_labels,
        title="Biological interpretation",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=10,
        title_fontsize=10,
    )

    ax.set_title(title, fontsize=14, pad=16)
    ax.axis("equal")

    fig.tight_layout()
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Input validation and long-format model summaries
# -----------------------------------------------------------------------------

def validate_core_table(core_df: pd.DataFrame) -> None:
    required_cols = [
        FAMILY_NAME_COL,
        M0_AICC_COL,
        M1_LINEAR_AICC_COL,
        M1_EXP_AICC_COL,
        M0_VS_M1_LINEAR_COL,
        M0_VS_M1_EXP_COL,
    ]
    missing_cols = [col for col in required_cols if col not in core_df.columns]
    if missing_cols:
        raise ValueError(f"Core comparison table is missing columns: {missing_cols}")


def validate_family_sets(
    core_df: pd.DataFrame,
    baseline_model_summary_df: pd.DataFrame,
    transition_label: str,
) -> None:
    baseline_df = baseline_model_summary_df.copy()

    if LABEL_TESTED_TRANSITION_COL in baseline_df.columns:
        baseline_df = baseline_df[
            baseline_df[LABEL_TESTED_TRANSITION_COL] == transition_label
        ].copy()

    core_families = set(core_df[FAMILY_NAME_COL].dropna().astype(str))
    baseline_families = set(baseline_df[FAMILY_NAME_COL].dropna().astype(str))

    missing_from_core = sorted(baseline_families - core_families)
    extra_in_core = sorted(core_families - baseline_families)

    if missing_from_core or extra_in_core:
        raise ValueError(
            f"{transition_label}: family-set mismatch between core and "
            f"baseline model summary.\n"
            f"Missing from core: {missing_from_core}\n"
            f"Extra in core: {extra_in_core}"
        )


def build_model_summary_from_core(core_df: pd.DataFrame) -> pd.DataFrame:
    """Build Constant/Linear/EXP long-format rows directly from the core table."""
    validate_core_table(core_df)

    model_specs = (
        (LABEL_CONSTANT, M0_AICC_COL),
        (LABEL_LINEAR, M1_LINEAR_AICC_COL),
        (LABEL_EXP, M1_EXP_AICC_COL),
    )

    pieces: list[pd.DataFrame] = []
    for model_label, aicc_col in model_specs:
        piece = core_df[[FAMILY_NAME_COL, aicc_col]].copy()
        piece = piece.rename(columns={aicc_col: AICC_COL})
        piece[LABEL_FUNC_TYPE_COL] = model_label
        piece[AICC_COL] = pd.to_numeric(piece[AICC_COL], errors="coerce")
        pieces.append(piece[[FAMILY_NAME_COL, LABEL_FUNC_TYPE_COL, AICC_COL]])

    model_summary_df = pd.concat(pieces, ignore_index=True)
    model_summary_df = model_summary_df.dropna(
        subset=[FAMILY_NAME_COL, LABEL_FUNC_TYPE_COL, AICC_COL]
    )

    return model_summary_df.sort_values(
        by=[FAMILY_NAME_COL, LABEL_FUNC_TYPE_COL]
    ).reset_index(drop=True)


def build_model_summary_with_ignore(
    baseline_model_summary_df: pd.DataFrame,
    core_df: pd.DataFrame,
    transition_label: str,
) -> pd.DataFrame:
    """Build Constant/Linear/EXP/Ignore long-format model summary."""
    baseline_df = baseline_model_summary_df.copy()

    if LABEL_TESTED_TRANSITION_COL in baseline_df.columns:
        baseline_df = baseline_df[
            baseline_df[LABEL_TESTED_TRANSITION_COL] == transition_label
        ].copy()

    required_cols = [FAMILY_NAME_COL, LABEL_FUNC_TYPE_COL, AICC_COL]
    missing_cols = [col for col in required_cols if col not in baseline_df.columns]
    if missing_cols:
        raise ValueError(
            f"Baseline model summary table is missing columns: {missing_cols}"
        )

    # Use Constant/Linear/Ignore from the baseline summary, then add EXP from core.
    baseline_df = baseline_df[
        baseline_df[LABEL_FUNC_TYPE_COL].isin(
            [LABEL_CONSTANT, LABEL_LINEAR, LABEL_IGNORE]
        )
    ][required_cols].copy()
    baseline_df[AICC_COL] = pd.to_numeric(baseline_df[AICC_COL], errors="coerce")

    exp_df = core_df[[FAMILY_NAME_COL, M1_EXP_AICC_COL]].copy()
    exp_df = exp_df.rename(columns={M1_EXP_AICC_COL: AICC_COL})
    exp_df[LABEL_FUNC_TYPE_COL] = LABEL_EXP
    exp_df[AICC_COL] = pd.to_numeric(exp_df[AICC_COL], errors="coerce")
    exp_df = exp_df[[FAMILY_NAME_COL, LABEL_FUNC_TYPE_COL, AICC_COL]]

    model_summary_df = pd.concat([baseline_df, exp_df], ignore_index=True)
    model_summary_df = model_summary_df.dropna(
        subset=[FAMILY_NAME_COL, LABEL_FUNC_TYPE_COL, AICC_COL]
    )

    return model_summary_df.sort_values(
        by=[FAMILY_NAME_COL, LABEL_FUNC_TYPE_COL]
    ).reset_index(drop=True)


def build_model_summary(
    core_df: pd.DataFrame,
    transition_label: str,
    ignore_mode: str,
    baseline_model_summary_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if ignore_mode == IGNORE_MODE_EXCLUDE:
        return build_model_summary_from_core(core_df)

    if baseline_model_summary_df is None:
        raise ValueError(
            "--baseline_model_summary_table_path is required when "
            "--ignore-mode include."
        )

    validate_family_sets(
        core_df=core_df,
        baseline_model_summary_df=baseline_model_summary_df,
        transition_label=transition_label,
    )

    return build_model_summary_with_ignore(
        baseline_model_summary_df=baseline_model_summary_df,
        core_df=core_df,
        transition_label=transition_label,
    )


# -----------------------------------------------------------------------------
# Akaike weights and chosen model
# -----------------------------------------------------------------------------

def add_akaike_weights(model_summary_df: pd.DataFrame) -> pd.DataFrame:
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

    return output_df.drop(columns=["relative_likelihood"])


def build_chosen_model_table_with_exp(
    model_summary_with_weights_df: pd.DataFrame,
) -> pd.DataFrame:
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

    return chosen_df.sort_values(by=FAMILY_NAME_COL).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Biological dependence interpretation
# -----------------------------------------------------------------------------

def add_dependence_detection_status(
    core_df: pd.DataFrame,
    chosen_model_df: pd.DataFrame,
    ignore_mode: str,
) -> pd.DataFrame:
    """
    Add a biological interpretation of chromosome-number dependence.

    Pairwise Constant-vs-Linear and Constant-vs-EXP comparisons distinguish
    no detected dependence, dependence supported only by Linear, only by EXP,
    or by both functional forms.

    When IGNORE participates in model selection and is the overall winner, the
    family is labelled 'Transition ignored' rather than being interpreted as
    evidence for or against chromosome-number dependence.
    """
    output_df = core_df.copy()

    exp_beats_constant = output_df[M0_VS_M1_EXP_COL] == LABEL_EXP
    linear_beats_constant = output_df[M0_VS_M1_LINEAR_COL] == LABEL_LINEAR

    output_df[DEPENDENCE_DETECTION_STATUS_COL] = NO_DETECTED_DEPENDENCE_STATUS

    output_df.loc[
        linear_beats_constant & ~exp_beats_constant,
        DEPENDENCE_DETECTION_STATUS_COL,
    ] = LINEAR_ONLY_DEPENDENCE_STATUS

    output_df.loc[
        exp_beats_constant & ~linear_beats_constant,
        DEPENDENCE_DETECTION_STATUS_COL,
    ] = EXP_ONLY_DEPENDENCE_STATUS

    output_df.loc[
        exp_beats_constant & linear_beats_constant,
        DEPENDENCE_DETECTION_STATUS_COL,
    ] = BOTH_FORMS_DEPENDENCE_STATUS

    chosen_lookup = chosen_model_df[
        [FAMILY_NAME_COL, CHOSEN_FUNCTION_LABEL_COL]
    ].drop_duplicates(subset=[FAMILY_NAME_COL])

    output_df = output_df.merge(
        chosen_lookup,
        on=FAMILY_NAME_COL,
        how="left",
        validate="one_to_one",
    )

    if ignore_mode == IGNORE_MODE_INCLUDE:
        ignore_selected = output_df[CHOSEN_FUNCTION_LABEL_COL] == LABEL_IGNORE
        output_df.loc[
            ignore_selected,
            DEPENDENCE_DETECTION_STATUS_COL,
        ] = TRANSITION_IGNORED_STATUS

    return output_df


def get_detection_order(ignore_mode: str) -> tuple[str, ...]:
    order = (
        NO_DETECTED_DEPENDENCE_STATUS,
        LINEAR_ONLY_DEPENDENCE_STATUS,
        EXP_ONLY_DEPENDENCE_STATUS,
        BOTH_FORMS_DEPENDENCE_STATUS,
    )
    if ignore_mode == IGNORE_MODE_INCLUDE:
        return (*order, TRANSITION_IGNORED_STATUS)
    return order


def get_detection_status_color_map() -> dict[str, str]:
    return {
        NO_DETECTED_DEPENDENCE_STATUS: MODEL_COLOR_MAP[LABEL_CONSTANT],
        LINEAR_ONLY_DEPENDENCE_STATUS: MODEL_COLOR_MAP[LABEL_LINEAR],
        EXP_ONLY_DEPENDENCE_STATUS: MODEL_COLOR_MAP[LABEL_EXP],
        BOTH_FORMS_DEPENDENCE_STATUS: "#8E7CC3",
        TRANSITION_IGNORED_STATUS: MODEL_COLOR_MAP[LABEL_IGNORE],
    }


# -----------------------------------------------------------------------------
# Summary/output helpers
# -----------------------------------------------------------------------------

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


def build_aggregated_weights(
    model_summary_with_weights_df: pd.DataFrame,
    model_labels: Sequence[str],
) -> pd.Series:
    agg_weights = model_summary_with_weights_df.groupby(
        LABEL_FUNC_TYPE_COL
    )[AKAIKE_WEIGHT_COL].sum()

    return pd.Series(
        {
            model_label: float(agg_weights.get(model_label, 0.0))
            for model_label in model_labels
        }
    )


def save_table(df: pd.DataFrame, out_file: Path, output_paths: List[str]) -> None:
    df.to_csv(out_file, index=False)
    output_paths.append(str(out_file))


def save_count_bar_and_pie(
    counts: pd.Series,
    label_col: str,
    title: str,
    output_prefix: Path,
    color_map: dict[str, str],
    total_n: int,
    output_paths: List[str],
    use_dependence_legend_pie: bool = False,
) -> None:
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

    if use_dependence_legend_pie:
        plot_dependence_pie_with_legend(
            counts=counts,
            title=title,
            out_file=pie_file,
            color_map=color_map,
        )
    else:
        plot_pie_counts(
            counts=counts,
            title=title,
            out_file=pie_file,
            color_map=color_map,
            show_counts=True,
            total_n=total_n,
        )
    output_paths.append(str(pie_file))


def get_mode_suffix(ignore_mode: str) -> str:
    return "with_ignore" if ignore_mode == IGNORE_MODE_INCLUDE else "without_ignore"


# -----------------------------------------------------------------------------
# Main analysis
# -----------------------------------------------------------------------------

def run_model_selection_analysis(
    transition_label: str,
    core_table_path: Path,
    baseline_model_summary_table_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    ignore_mode: str = IGNORE_MODE_INCLUDE,
) -> List[str]:
    if ignore_mode not in IGNORE_MODE_CHOICES:
        raise ValueError(
            f"ignore_mode must be one of {IGNORE_MODE_CHOICES}; got {ignore_mode!r}"
        )

    if ignore_mode == IGNORE_MODE_INCLUDE and baseline_model_summary_table_path is None:
        raise ValueError(
            "baseline_model_summary_table_path is required when ignore_mode='include'."
        )

    output_dir = (
        output_dir
        or ANALYSIS_DIR
        / EXPONENTIAL_VS_LINEAR_SUBDIR
        / transition_label
        / MODEL_SELECTION_SUBDIR
    )
    ensure_dir(output_dir)

    output_paths: List[str] = []
    mode_suffix = get_mode_suffix(ignore_mode)

    core_df = pd.read_csv(core_table_path)
    validate_core_table(core_df)

    baseline_model_summary_df: Optional[pd.DataFrame] = None
    if baseline_model_summary_table_path is not None:
        baseline_model_summary_df = pd.read_csv(baseline_model_summary_table_path)

    # 1. Build the model set requested by --ignore-mode and recompute weights.
    model_labels = (
        MODEL_LABELS_WITH_EXP_AND_IGNORE
        if ignore_mode == IGNORE_MODE_INCLUDE
        else MODEL_LABELS_WITH_EXP_NO_IGNORE
    )

    model_summary_df = build_model_summary(
        core_df=core_df,
        transition_label=transition_label,
        ignore_mode=ignore_mode,
        baseline_model_summary_df=baseline_model_summary_df,
    )
    model_summary_with_weights_df = add_akaike_weights(model_summary_df)

    model_summary_with_weights_file = (
        output_dir
        / f"{transition_label}_model_summary_with_exp_and_weights_{mode_suffix}.csv"
    )
    save_table(
        model_summary_with_weights_df,
        model_summary_with_weights_file,
        output_paths,
    )

    # 2. Chosen model under the requested model set.
    chosen_with_exp_df = build_chosen_model_table_with_exp(
        model_summary_with_weights_df=model_summary_with_weights_df,
    )
    chosen_with_exp_file = (
        output_dir
        / f"{transition_label}_chosen_model_with_exp_{mode_suffix}.csv"
    )
    save_table(chosen_with_exp_df, chosen_with_exp_file, output_paths)

    chosen_counts = get_ordered_counts(
        chosen_with_exp_df[CHOSEN_FUNCTION_LABEL_COL],
        order=model_labels,
    )
    save_count_bar_and_pie(
        counts=chosen_counts,
        label_col=CHOSEN_FUNCTION_LABEL_COL,
        title=get_model_selection_title(transition_label, ignore_mode),
        output_prefix=(
            output_dir / f"{transition_label}_chosen_model_with_exp_{mode_suffix}"
        ),
        color_map=MODEL_COLOR_MAP,
        total_n=len(chosen_with_exp_df),
        output_paths=output_paths,
    )

    # 3. Biological interpretation of chromosome-number dependence.
    core_with_detection_df = add_dependence_detection_status(
        core_df=core_df,
        chosen_model_df=chosen_with_exp_df,
        ignore_mode=ignore_mode,
    )
    core_with_detection_file = (
        output_dir
        / f"{transition_label}_core_table_with_dependence_detection_{mode_suffix}.csv"
    )
    save_table(core_with_detection_df, core_with_detection_file, output_paths)

    detection_order = get_detection_order(ignore_mode)
    detection_counts = get_ordered_counts(
        core_with_detection_df[DEPENDENCE_DETECTION_STATUS_COL],
        detection_order,
    )
    save_count_bar_and_pie(
        counts=detection_counts,
        label_col=DEPENDENCE_DETECTION_STATUS_COL,
        title=get_dependence_title(transition_label, ignore_mode),
        output_prefix=(
            output_dir / f"{transition_label}_dependence_detection_{mode_suffix}"
        ),
        color_map=get_detection_status_color_map(),
        total_n=len(core_with_detection_df),
        output_paths=output_paths,
        use_dependence_legend_pie=True,
    )
    # 4. Aggregated Akaike weights under the requested model set.
    agg_weights = build_aggregated_weights(
        model_summary_with_weights_df=model_summary_with_weights_df,
        model_labels=model_labels,
    )
    agg_weights_file = (
        output_dir
        / f"{transition_label}_aggregated_akaike_weights_with_exp_{mode_suffix}.csv"
    )
    save_table(
        pd.DataFrame(
            {
                "model": agg_weights.index,
                "aggregated_akaike_weight": agg_weights.values,
            }
        ),
        agg_weights_file,
        output_paths,
    )

    agg_weights_pie_file = (
        output_dir
        / f"{transition_label}_aggregated_akaike_weights_with_exp_{mode_suffix}_pie.png"
    )
    plot_pie_counts(
        counts=agg_weights,
        title=get_akaike_title(transition_label, ignore_mode),
        out_file=agg_weights_pie_file,
        color_map=MODEL_COLOR_MAP,
        show_counts=False,
        total_n=len(chosen_with_exp_df),
    )
    output_paths.append(str(agg_weights_pie_file))

    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Constant/Linear/EXP model-selection analysis with optional "
            "inclusion of the IGNORE model."
        )
    )

    parser.add_argument(
        "--transition_label",
        type=str,
        required=True,
        choices=[LABEL_GAIN, LABEL_LOSS, LABEL_DUPL],
    )
    parser.add_argument("--core_table_path", type=Path, required=True)
    parser.add_argument(
        "--baseline_model_summary_table_path",
        type=Path,
        default=None,
        help=(
            "Baseline model-summary CSV. Required only when "
            "--ignore-mode include."
        ),
    )
    parser.add_argument(
        "--ignore-mode",
        choices=IGNORE_MODE_CHOICES,
        default=IGNORE_MODE_INCLUDE,
        help=(
            "Whether IGNORE participates in chosen-model selection and Akaike "
            "weights. Default: include (backward-compatible)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        type=Path,
        default=None,
        help=(
            "Optional output directory. If omitted, uses the original "
            "exponential_vs_linear analysis directory."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_paths = run_model_selection_analysis(
        transition_label=args.transition_label,
        core_table_path=args.core_table_path,
        baseline_model_summary_table_path=args.baseline_model_summary_table_path,
        output_dir=args.output_dir,
        ignore_mode=args.ignore_mode,
    )

    print(
        f"[✓] EXP-LINEAR model-selection analysis completed "
        f"(ignore_mode={args.ignore_mode})."
    )
    for output_path in output_paths:
        print(output_path)

    log_run(
        step="analysis",
        script=Path(__file__),
        params={
            "analysis_type": EXPONENTIAL_VS_LINEAR_SUBDIR,
            "transition_label": args.transition_label,
            "core_table_path": str(args.core_table_path),
            "baseline_model_summary_table_path": (
                str(args.baseline_model_summary_table_path)
                if args.baseline_model_summary_table_path is not None
                else None
            ),
            "ignore_mode": args.ignore_mode,
            "output_dir": str(args.output_dir) if args.output_dir is not None else None,
        },
        outputs=output_paths,
        description=(
            "Created Constant/Linear/EXP model-selection tables and plots, "
            "with optional inclusion of IGNORE."
        ),
        notes=(
            "When IGNORE is excluded, Constant/Linear/EXP AICc values are read "
            "directly from the core comparison table and Akaike weights are "
            "recomputed over those three models. When IGNORE is included, the "
            "baseline model summary supplies Constant/Linear/Ignore and EXP is "
            "added from the core table. Biological dependence labels distinguish "
            "no detected dependence, Linear-only support, EXP-only support, "
            "support by both forms, and (when applicable) an ignored transition."
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
