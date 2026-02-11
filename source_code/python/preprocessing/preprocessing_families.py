from typing import Dict, List, Any

from ..constants import *
from .preprocessing_utils import *

def families_data_summary_to_csv(database_chr_count_input: Path, preprocessing_dir: Path) -> tuple[int, int, int]:
    skipped_log: Path = preprocessing_dir / f"skipped_families_missing_counts_fasta.txt"
    data: Dict[str, List[Any]] = {k: [] for k in PREPROCESSING_COLUMNS}

    total_families = 0
    skipped_families = 0

    for folder in database_chr_count_input.iterdir():
        if not folder.is_dir():
            continue
        if folder.name == ".ipynb_checkpoints" or folder.name.startswith("."):
            continue

        total_families += 1
        family_name: str = folder.name
        chrom_counts_file_path: Path = folder / CHROM_COUNTS_FILE_NAME

        if not chrom_counts_file_path.is_file():
            print(f"---{family_name} skipped")
            skipped_log.parent.mkdir(parents=True, exist_ok=True)
            skipped_log.open("a").write(f"{family_name}\n")
            skipped_families += 1
            continue

        print(family_name)
        data[FAMILY_NAME_COL].append(family_name)

        if chrom_counts_file_path.is_file():
            with chrom_counts_file_path.open('r') as file:
                tree_size: int = sum(1 for line in file if line.startswith('>'))
            data[FAMILY_SIZE_COL].append(tree_size)

            min_chrom, max_chrom, diff = family_chrom_range(chrom_counts_file_path)
            data[MIN_CHROM_COL].append(min_chrom)
            data[MAX_CHROM_COL].append(max_chrom)
            data[DIFF_COL].append(diff)

            ploidb_by_family, ploidb_by_genus = family_ploidy_level(family_name, PLOIDB_BY_FAMILY_FILE, PLOIDB_BY_GENUS_FILE)

            data[POLYPLOIDY_BY_FAMILY].append(ploidb_by_family)
            data[POLYPLOIDY_BY_GENUS].append(ploidb_by_genus)

    preprocessing_dir.mkdir(parents=True, exist_ok=True)
    output_path: Path = preprocessing_dir / "all_families_data_summary.csv"
    pd.DataFrame(data).to_csv(output_path, index=False)

    kept_families = total_families - skipped_families
    return total_families, skipped_families, kept_families


def filter_families_by_minimum_num_of_species(families_data_summary: Path, min_family_size: int, preprocessing_dir: Path) -> tuple[int, int, int]:
    all_families_df: pd.DataFrame = pd.read_csv(families_data_summary)

    excluded_df: pd.DataFrame = all_families_df.loc[all_families_df[FAMILY_SIZE_COL] < min_family_size,[FAMILY_NAME_COL, FAMILY_SIZE_COL]].copy()
    preprocessing_dir.mkdir(parents=True, exist_ok=True)
    excluded_log: Path = preprocessing_dir / f"excluded_families_min_size_{min_family_size}.txt"
    excluded_df.to_csv(excluded_log, sep="\t", index=False)

    filtered_df: pd.DataFrame = all_families_df.loc[all_families_df[FAMILY_SIZE_COL] >= min_family_size]
    preprocessing_dir.mkdir(parents=True, exist_ok=True)
    output_path: Path = preprocessing_dir / "families_for_analysis.csv"
    filtered_df.to_csv(output_path, index=False)

    total = int(len(all_families_df))
    excluded = int(len(excluded_df))
    kept = int(len(filtered_df))
    return total, excluded, kept


def preprocessing_summary(families_for_analysis_file: Path, preprocessing_dir: Path, columns: list[str]) -> None:
    analysis_families_df = pd.read_csv(families_for_analysis_file)

    for col in columns:
        if col == FAMILY_NAME_COL:
            continue
        analysis_families_df[col] = pd.to_numeric(analysis_families_df[col])

    summary_output_path = preprocessing_dir / "families_for_analysis_summary.txt"

    with summary_output_path.open("w") as f:
        f.write("Families for analysis — summary statistics\n")
        f.write("=" * 45 + "\n\n")

        for col in columns:
            if col == FAMILY_NAME_COL:
                continue
            series: pd.Series = analysis_families_df[col].dropna()

            if series.empty:
                f.write(f"{col}:\n  no valid values\n\n")
                continue

            f.write(f"{col}:\n")
            f.write(f"  min     = {series.min()}\n")
            f.write(f"  max     = {series.max()}\n")
            f.write(f"  average = {series.mean()}\n\n")

        f.write("Global counts:\n")
        f.write(f"  number of families = {len(analysis_families_df)}\n")


