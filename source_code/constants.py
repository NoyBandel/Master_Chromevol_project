from pathlib import Path
from typing import Dict

# -------- paths --------
LOGS_ROOT: Path = Path("/groups/itay_mayrose/noybandel/Master_ChromEvol_project/logs/")

DATABASE_INPUT_DIR: Path = Path("/groups/itay_mayrose/noybandel/Master_ChromEvol_project/input_data/families_chrom_input/")
INPUT_DATA_DIR: Path = Path("/groups/itay_mayrose/noybandel/Master_ChromEvol_project/input_data/")
ALL_FAMILIES_DATA_SUMMARY_FILE: Path = Path("all_families_data_summary.csv")
PLOIDB_BY_FAMILY_FILE: Path = Path("/groups/itay_mayrose/noybandel/Master_ChromEvol_project/input_data/ploidb_by_family_without_missing.csv")
PLOIDB_BY_GENUS_FILE: Path = Path("/groups/itay_mayrose/noybandel/Master_ChromEvol_project/input_data/ploidb_by_genus_without_missing.csv")

CHROMEVOL_RAW_RESULTS_ROOT: Path = Path("/groups/itay_mayrose/noybandel/Master_ChromEvol_project/chromevol_raw_results/")
CHROMEVOL_EXE: Path = Path("/groups/itay_mayrose/noybandel/ChromEvol_project/chromevol_program/chromevol/ChromEvol/chromEvol")
CONDA_ENV: Path = Path("source /groups/itay_mayrose/noybandel/miniconda3/etc/profile.d/conda.sh; conda activate chromevol")
CONDA_EXPORT: Path = Path("export LD_LIBRARY_PATH=/groups/itay_mayrose/noybandel/miniconda3/envs/chromevol/lib:$LD_LIBRARY_PATH\n")

RAW_RESULTS_DIR = Path("/groups/itay_mayrose/noybandel/Master_ChromEvol_project/chromevol_raw_results/")


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
CHROM_COUNTS_FILE_NAME: str = "counts.fasta"
TREE_FILE_NAME: str = "tree.nwk"


# -------- chromevol run --------
# cheomevol functions names
CE_IGNORE = "IGNORE"
CE_CONSTANT = "CONST"
CE_LINEAR = "LINEAR"
CE_EXP = "EXP"
CE_LOGNORMAL = "LOGNORMAL"
CE_FUNCTIONS = [CE_IGNORE, CE_CONSTANT, CE_LINEAR, CE_EXP, CE_LOGNORMAL]

# cheomevol transitions names
CE_GAIN = "_gainFunc"
CE_LOSS = "_lossFunc"
CE_DUPL = "_duplFunc"
CE_DEMI_DUPL = "_demiDuplFunc"
CE_BASE_NUM = "_baseNumRFunc"
CE_TRANSITIONS_ORDERED = (CE_GAIN, CE_LOSS, CE_DUPL, CE_DEMI_DUPL, CE_BASE_NUM)

# cheomevol transitions names for initialization
CE_GAIN_INIT = "_gain_1"
CE_LOSS_INIT = "_loss_1"
CE_DUPL_INIT = "_dupl_1"
CE_DEMI_INIT = "_demiPloidyR_1"
CE_BASE_NUM_R_INIT = "_baseNumR_1"

CE_BASE_CHROM_NUM_INIT = "_baseNum_1"

CE_TRANSITIONS_INIT_ORDERED = (CE_GAIN_INIT, CE_LOSS_INIT, CE_DUPL_INIT, CE_DEMI_INIT, CE_BASE_NUM_R_INIT)

DEFAULT_INIT_VALUES = {CE_IGNORE: "", CE_CONSTANT: "1.0", CE_LINEAR: "1.0,0.1", CE_EXP: "1.0,0.01", CE_LOGNORMAL: "1.0,0.1,0.1", CE_BASE_CHROM_NUM_INIT:"6"}
INIT_VALUES_COUNT_PER_FUNCTION = {CE_IGNORE: 0, CE_CONSTANT: 1, CE_LINEAR: 2, CE_EXP: 2, CE_LOGNORMAL: 3, CE_BASE_CHROM_NUM_INIT: 1}

# CE_TRANSITION_NAME <--> CE_TRANSITION_INIT
CE_TRANSITIONS_TO_INIT = {
    CE_GAIN: CE_GAIN_INIT,
    CE_LOSS: CE_LOSS_INIT,
    CE_DUPL: CE_DUPL_INIT,
    CE_DEMI_DUPL: CE_DEMI_INIT,
    CE_BASE_NUM: CE_BASE_NUM_R_INIT
}

CE_INIT_TO_TRANSITIONS = {v: k for k, v in CE_TRANSITIONS_TO_INIT.items()}



# -------- labels --------
# config df columns
LABEL_TRANSITION_COL = "transition_label"
LABEL_FUNC_TYPE_COL = "func_type_label"
CE_TRANSITION_COL = "chromevol_transition"
CE_FUNC_TYPE_COL = "chromevol_func_type"
CE_TRANSITION_INIT_COL = "chromevol_transition_init"
CE_PARAMS_COL = "chromevol_parameters"
CONFIG_DF_COLS = [LABEL_TRANSITION_COL, LABEL_FUNC_TYPE_COL, CE_TRANSITION_COL, CE_FUNC_TYPE_COL, CE_TRANSITION_INIT_COL, CE_PARAMS_COL]

# analysis functions names - for folders and csv files
LABEL_IGNORE = "ignore"
LABEL_CONSTANT = "constant"
LABEL_LINEAR = "linear"
LABEL_EXP = "exponential"
LABEL_LOGNORMAL = "log-normal"
LABEL_FUNCTIONS_LST = [LABEL_IGNORE, LABEL_CONSTANT, LABEL_LINEAR, LABEL_EXP, LABEL_LOGNORMAL]

CE_FUNC_TO_LABEL_FUNC: Dict[str, str] = {
    CE_IGNORE: LABEL_IGNORE,
    CE_CONSTANT: LABEL_CONSTANT,
    CE_LINEAR: LABEL_LINEAR,
    CE_EXP: LABEL_EXP,
    CE_LOGNORMAL: LABEL_LOGNORMAL
}
LABEL_FUNC_TO_CE_FUNC: Dict[str, str] = {v: k for k, v in CE_FUNC_TO_LABEL_FUNC.items()}



# analysis transitions names - for folders and csv files
LABEL_GAIN = "gain"
LABEL_LOSS = "loss"
LABEL_DUPL = "dupl"
LABEL_DEMI = "demi"
LABEL_BASE_NUM = "baseNum"
LABEL_TRANSITIONS_ORDERED = (LABEL_GAIN, LABEL_LOSS, LABEL_DUPL, LABEL_DEMI, LABEL_BASE_NUM)


PARAM_FILE = "paramFile"

# -------- configuration types --------
M0_LABEL = "M0_all_const"
M1_LABEL = "M1_tested_linear_w_M0_init"

CONFIG_TYPES = [M0_LABEL, M1_LABEL]

