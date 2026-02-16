from pathlib import Path
from typing import Tuple
import pandas as pd
from Bio import SeqIO

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

def min_max_pars(extreme_type: str, families_df: pd.DataFrame, min_chrom_col: str, max_chrom_col: str) -> Tuple[str, int, str]:
    if extreme_type == "min":
        extreme_col = min_chrom_col
        extreme_value = int(pd.to_numeric(families_df[extreme_col], errors="coerce").min())
        header_line = f"minimal chrom count is {extreme_value}"
    else:
        extreme_col = max_chrom_col
        extreme_value = int(pd.to_numeric(families_df[extreme_col], errors="coerce").max())
        header_line = f"maximal chrom count is {extreme_value}"
    return extreme_col, extreme_value, header_line

def write_species_with_extreme_chr_num_file(out_path: Path, header_line: str, extreme_rows: pd.DataFrame, family_name_col: str, database_chr_count_input: Path, chrom_count_file_name, extreme_value: int) -> None:
    with out_path.open("w", encoding="utf-8") as out:
        out.write(header_line + "\n\n")

        for _, row in extreme_rows.iterrows():
            family_name = row[family_name_col]
            chrom_counts_file_path = (database_chr_count_input/ family_name/ chrom_count_file_name)
            if not chrom_counts_file_path.is_file():
                continue
            out.write(f"family: {family_name}\n")

            for record in SeqIO.parse(chrom_counts_file_path, "fasta"):
                try:
                    chrom_value = int(str(record.seq).strip())
                except ValueError:
                    continue
                if chrom_value == extreme_value:
                    full_header = record.description.strip()
                    species_name = " ".join(full_header.split()[:2])
                    out.write(f"  - {species_name}\n")

            out.write("\n")