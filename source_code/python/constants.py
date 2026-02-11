from pathlib import Path

# -------- paths --------
DATABASE_INPUT_DIR: Path = Path("/groups/itay_mayrose/noybandel/Master_ChromEvol_project/input_data/families_chrom_input/")
INPUT_DATA_DIR: Path = Path("/groups/itay_mayrose/noybandel/Master_ChromEvol_project/input_data/")
ALL_FAMILIES_DATA_SUMMARY_FILE: Path = Path("all_families_data_summary.csv")


PLOIDB_BY_FAMILY_FILE: Path = Path("/groups/itay_mayrose/noybandel/Master_ChromEvol_project/input_data/ploidb_by_family_without_missing.csv")
PLOIDB_BY_GENUS_FILE: Path = Path("/groups/itay_mayrose/noybandel/Master_ChromEvol_project/input_data/ploidb_by_genus_without_missing.csv")


LOGS_ROOT: Path = Path("/groups/itay_mayrose/noybandel/Master_ChromEvol_project/logs/")


# -------- preprocessing --------
# columns
FAMILY_NAME_COL: str = "family_name"
FAMILY_SIZE_COL: str = "family_size"
MIN_CHROM_COL: str = "min_chrom"
MAX_CHROM_COL: str = "max_chrom"
DIFF_COL: str = "diff_chrom"
POLYPLOIDY_BY_FAMILY: str = "num_of_polyploid_species_by_family"
POLYPLOIDY_BY_GENUS: str = "num_of_polyploid_species_by_genus"

PREPROCESSING_COLUMNS: list[str] = [FAMILY_NAME_COL, FAMILY_SIZE_COL, MIN_CHROM_COL, MAX_CHROM_COL, DIFF_COL, POLYPLOIDY_BY_FAMILY, POLYPLOIDY_BY_GENUS]


# input data
CHROM_COUNTS_FILE_NAME:str = "counts.fasta"