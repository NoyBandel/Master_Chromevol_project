from pathlib import Path
from typing import Tuple
import pandas as pd

def family_chrom_range(counts_file_path: Path) -> tuple[int, int, int]:
    min_chrom: int | None = None
    max_chrom: int | None = None

    with counts_file_path.open("r") as file:
        for line in file:
            if line.startswith(">"):
                continue

            value = line.strip()
            if not value:
                continue

            chrom_num: int = int(value)

            if min_chrom is None or chrom_num < min_chrom:
                min_chrom = chrom_num
            if max_chrom is None or chrom_num > max_chrom:
                max_chrom = chrom_num

    if min_chrom is None or max_chrom is None:
        return 0, 0, 0

    return min_chrom, max_chrom, max_chrom - min_chrom


def family_ploidy_level(family_name: str,ploidb_by_family_file: Path,ploidb_by_genus_file: Path) -> Tuple[int, int]:
    ploidb_by_family_df: pd.DataFrame = pd.read_csv(ploidb_by_family_file)
    ploidb_by_genus_df: pd.DataFrame = pd.read_csv(ploidb_by_genus_file)

    fam_mask = ploidb_by_family_df["Family"].astype(str) == family_name
    gen_mask = ploidb_by_genus_df["Family"].astype(str) == family_name

    ploidb_by_family: int = int((fam_mask & (ploidb_by_family_df["Ploidy inference"] == 1)).sum())
    ploidb_by_genus: int = int((gen_mask & (ploidb_by_genus_df["Ploidy inference"] == 1)).sum())

    return ploidb_by_family, ploidb_by_genus

