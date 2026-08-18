"""
Create transition-specific family lists for a filtered baseline reanalysis.

A family is retained for a transition only if it satisfies all three criteria:
    1. family_size >= min_family_size
    2. chromosome range (diff_chrom) >= min_chrom_range
    3. expected number of events for that transition in the M0 constant run >= min_events

The expected event count is taken from parsed_results_M0_all_const.csv, so the filter is defined from the same constant model for every family.
"""

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from source_code.constants import *
from source_code.logger import log_run


DEFAULT_TRANSITIONS = (LABEL_GAIN, LABEL_LOSS, LABEL_DUPL)
DEFAULT_METADATA_FILE = INPUT_DATA_DIR / "families_for_analysis_metadata.csv"
DEFAULT_M0_PARSED_FILE = PARSED_RESULTS_ROOT / f"{PARSED_RESULTS_FILE_PREFIX}_{M0_LABEL}.csv"
DEFAULT_OUTPUT_DIR = INPUT_DATA_DIR / "filtered_families" / "range25_events15_size50"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create transition-specific family lists using metadata and M0 expected events.")
    parser.add_argument("--metadata-file", type=Path, default=DEFAULT_METADATA_FILE, help="Family metadata CSV containing family_name, family_size, and diff_chrom.")
    parser.add_argument("--m0-parsed-file", type=Path, default=DEFAULT_M0_PARSED_FILE, help="Parsed M0_all_const results CSV containing expected event counts.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for transition family lists and audit tables.")
    parser.add_argument("--transitions", nargs="+", choices=DEFAULT_TRANSITIONS, default=list(DEFAULT_TRANSITIONS), help="Transitions for which to create family lists.")
    parser.add_argument("--min-chrom-range", type=float, default=25.0)
    parser.add_argument("--min-events", type=float, default=15.0)
    parser.add_argument("--min-family-size", type=float, default=50.0)
    return parser.parse_args()


def load_inputs(metadata_file: Path, m0_parsed_file: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    if not m0_parsed_file.exists():
        raise FileNotFoundError(f"M0 parsed-results file not found: {m0_parsed_file}")

    metadata = pd.read_csv(metadata_file)
    m0 = pd.read_csv(m0_parsed_file)

    metadata_required = [FAMILY_NAME_COL, FAMILY_SIZE_COL, DIFF_COL]
    m0_required = [FAMILY_NAME_COL, EXP_GAIN_COL, EXP_LOSS_COL, EXP_DUPL_COL]

    missing_metadata = [col for col in metadata_required if col not in metadata.columns]
    missing_m0 = [col for col in m0_required if col not in m0.columns]

    if missing_metadata:
        raise ValueError(f"Metadata file is missing required columns: {missing_metadata}")
    if missing_m0:
        raise ValueError(f"M0 parsed-results file is missing required columns: {missing_m0}")

    if metadata[FAMILY_NAME_COL].duplicated().any():
        duplicated = sorted(metadata.loc[metadata[FAMILY_NAME_COL].duplicated(), FAMILY_NAME_COL].astype(str).unique())
        raise ValueError(f"Duplicate families in metadata file: {duplicated}")

    if m0[FAMILY_NAME_COL].duplicated().any():
        duplicated = sorted(m0.loc[m0[FAMILY_NAME_COL].duplicated(), FAMILY_NAME_COL].astype(str).unique())
        raise ValueError(f"Duplicate families in M0 parsed-results file: {duplicated}")

    metadata = metadata[metadata_required].copy()
    m0 = m0[m0_required].copy()

    metadata[FAMILY_SIZE_COL] = pd.to_numeric(metadata[FAMILY_SIZE_COL], errors="coerce")
    metadata[DIFF_COL] = pd.to_numeric(metadata[DIFF_COL], errors="coerce")

    for col in [EXP_GAIN_COL, EXP_LOSS_COL, EXP_DUPL_COL]:
        m0[col] = pd.to_numeric(m0[col], errors="coerce")

    return metadata, m0


def build_filter_table(metadata: pd.DataFrame, m0: pd.DataFrame, transition: str, min_family_size: float, min_chrom_range: float, min_events: float) -> pd.DataFrame:
    events_col = TRANSITION_TO_EVENTS_COL[transition]

    table = metadata.merge(m0[[FAMILY_NAME_COL, events_col]], on=FAMILY_NAME_COL, how="left", validate="one_to_one").rename(columns={events_col: "m0_expected_events"})

    table["passes_family_size"] = table[FAMILY_SIZE_COL].ge(min_family_size)
    table["passes_chrom_range"] = table[DIFF_COL].ge(min_chrom_range)
    table["passes_events"] = table["m0_expected_events"].ge(min_events)
    table["retained"] = (table["passes_family_size"] & table["passes_chrom_range"] & table["passes_events"])

    failed_reasons: List[str] = []
    for row in table.itertuples(index=False):
        reasons: List[str] = []
        if pd.isna(getattr(row, FAMILY_SIZE_COL)):
            reasons.append("missing_family_size")
        elif not getattr(row, "passes_family_size"):
            reasons.append("family_size")

        if pd.isna(getattr(row, DIFF_COL)):
            reasons.append("missing_chrom_range")
        elif not getattr(row, "passes_chrom_range"):
            reasons.append("chrom_range")

        if pd.isna(getattr(row, "m0_expected_events")):
            reasons.append("missing_m0_events")
        elif not getattr(row, "passes_events"):
            reasons.append("events")

        failed_reasons.append(";".join(reasons))

    table["failed_filters"] = failed_reasons
    table = table.sort_values(FAMILY_NAME_COL).reset_index(drop=True)
    return table


def write_family_list(table: pd.DataFrame, output_file: Path) -> List[str]:
    families = sorted(table.loc[table["retained"], FAMILY_NAME_COL].astype(str).tolist())
    output_file.write_text("\n".join(families) + ("\n" if families else ""), encoding="utf-8")
    return families


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata, m0 = load_inputs(args.metadata_file, args.m0_parsed_file)
    outputs: List[str] = []

    print(
        "Filtering criteria: "
        f"family_size >= {args.min_family_size:g}, "
        f"chromosome_range >= {args.min_chrom_range:g}, "
        f"M0 expected events >= {args.min_events:g}"
    )

    for transition in args.transitions:
        table = build_filter_table(metadata, m0, transition, args.min_family_size, args.min_chrom_range, args.min_events)
        families_file = args.output_dir / f"{transition}_families.txt"
        audit_file = args.output_dir / f"{transition}_filter_audit.csv"

        retained_families = write_family_list(table, families_file)
        table.to_csv(audit_file, index=False)

        outputs.extend([str(families_file), str(audit_file)])

        missing_events = int(table["m0_expected_events"].isna().sum())
        print(
            f"[✓] {transition}: retained {len(retained_families)} / {len(table)} families"
            + (f" | missing M0 event count: {missing_events}" if missing_events else "")
        )
        print(f"    families: {families_file}")
        print(f"    audit:    {audit_file}")

    log_run(
        step="analysis",
        script=Path(__file__),
        params={
            "metadata_file": str(args.metadata_file),
            "m0_parsed_file": str(args.m0_parsed_file),
            "output_dir": str(args.output_dir),
            "transitions": list(args.transitions),
            "min_family_size": args.min_family_size,
            "min_chrom_range": args.min_chrom_range,
            "min_events": args.min_events,
        },
        outputs=outputs,
        description="Created transition-specific family lists for filtered baseline reanalysis.",
        notes="Expected event counts are taken from the M0_all_const run for the corresponding transition.",
        log_relative_path=Path("filtered_baseline") / "create_filtered_family_lists.log",
    )


if __name__ == "__main__":
    main()
