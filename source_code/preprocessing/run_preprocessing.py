import argparse
from pathlib import Path

from .preprocessing_families import *
from ..logger import log_run

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocessing pipeline runner")

    p.add_argument(
        "--run",
        choices=[
            "all_families_data_summary",
            "filter_min_family_size",
            "preprocessing_summary_stats",
            "find_species_with_extreme_chr_num",
        ],
        required=True,
        help="Which step to run",
    )

    p.add_argument(
        "--min-family-size",
        type=int,
        default=50,
        help="Minimum family size for filtering (used in 'filter_min_family_size')",
    )

    p.add_argument(
        "--extreme-type",
        choices=["min", "max"],
        default="min",
        help="Extreme type for 'find_species_with_extreme_chr_num' (min/max)",
    )

    p.add_argument(
        "--notes",
        type=str,
        default=None,
        help="Optional free-text notes to store in the run log",
    )

    return p.parse_args()

def main() -> None:
    args = parse_args()

    all_families_csv: Path = INPUT_DATA_DIR / ALL_FAMILIES_DATA_SUMMARY_FILE
    families_for_analysis_csv: Path = INPUT_DATA_DIR / "families_for_analysis.csv"
    summary_txt: Path = INPUT_DATA_DIR / "families_for_analysis_summary.txt"
    extreme_txt: Path = INPUT_DATA_DIR / f"{args.extreme_type}_chrom_species.txt"

    ran_steps: list[str] = []
    outputs: list[str] = []
    params: dict[str, object] = {
        "run": args.run,
        "database_chr_count_input": str(DATABASE_INPUT_DIR),
        "preprocessing_dir": str(INPUT_DATA_DIR),
    }
    notes_parts: list[str] = []

    if args.run == "all_families_data_summary":
        total_fams, skipped_fams, kept_fams = families_data_summary_to_csv(
            database_chr_count_input=DATABASE_INPUT_DIR,
            preprocessing_dir=INPUT_DATA_DIR,
        )
        ran_steps.append("all_families_data_summary")
        outputs.append(str(all_families_csv))
        notes_parts.append(
            f"Summary CSV: iterated {total_fams}; skipped {skipped_fams} missing {CHROM_COUNTS_FILE_NAME}; kept {kept_fams}."
        )

    if args.run == "filter_min_family_size":
        params["min_family_size"] = args.min_family_size

        total_filter, excluded_filter, kept_filter = filter_families_by_minimum_num_of_species(
            families_data_summary=all_families_csv,
            min_family_size=args.min_family_size,
            preprocessing_dir=INPUT_DATA_DIR,
        )
        ran_steps.append("filter_min_family_size")
        outputs.append(str(families_for_analysis_csv))
        outputs.append(str(INPUT_DATA_DIR / f"excluded_families_min_size_{args.min_family_size}.txt"))
        notes_parts.append(
            f"Filter(min_family_size={args.min_family_size}): excluded {excluded_filter}/{total_filter}; kept {kept_filter}."
        )

    if args.run == "preprocessing_summary_stats":
        preprocessing_summary(
            families_for_analysis_file=families_for_analysis_csv,
            preprocessing_dir=INPUT_DATA_DIR,
            columns=PREPROCESSING_COLUMNS,
        )
        ran_steps.append("preprocessing_summary_stats")
        outputs.append(str(summary_txt))
        params["families_for_analysis_file"] = str(families_for_analysis_csv)
        params["columns"] = PREPROCESSING_COLUMNS

    if args.run == "find_species_with_extreme_chr_num":
        params["extreme_type"] = args.extreme_type
        params["families_for_analysis_csv_file"] = str(families_for_analysis_csv)

        find_species_with_extreme_chr_num(
            extreme_type=args.extreme_type,
            families_for_analysis_csv_file=families_for_analysis_csv,
            database_chr_count_input=DATABASE_INPUT_DIR,
            preprocessing_dir=INPUT_DATA_DIR,
        )
        ran_steps.append("find_species_with_extreme_chr_num")
        outputs.append(str(extreme_txt))
        notes_parts.append(
            f"Extreme species report: extreme_type={args.extreme_type}; output={extreme_txt.name}."
        )

    if not ran_steps:
        return

    params["ran_steps"] = ran_steps

    if args.notes:
        notes_parts.append(f"User notes: {args.notes}")

    notes = " ".join(notes_parts) if notes_parts else None

    log_run(
        step="preprocessing",
        script=Path(__file__),
        params=params,
        outputs=outputs,
        description=(
            "Preprocessing pipeline (controlled by --run): "
            "all_families_data_summary, filter_min_family_size, "
            "preprocessing_summary_stats, find_species_with_extreme_chr_num."
        ),
        notes=notes,
    )


if __name__ == "__main__":
    main()
