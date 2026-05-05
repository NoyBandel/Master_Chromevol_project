import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from source_code.constants import *
from source_code.logger import log_run

ROOT_PATTERN = re.compile(r"Ancestral chromosome number at the root:\s*([0-9]+)")
LIKELIHOOD_PATTERN = re.compile(r"Final optimized likelihood is:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
AICC_PATTERN = re.compile(r"AICc of the best model\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
BASE_CHROM_NUM_PATTERN = re.compile(r"^Chromosome\.baseNum_1\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$")
TRANSITION_PARAM_PATTERN = re.compile(r"^Chromosome\.(gain|loss|dupl|demi|baseNumR)(\d*)_1\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$")
EXPECTATION_PATTERN = re.compile(r"^(GAIN|LOSS|DUPLICATION|DEMI-DUPLICATION|BASE-NUMBER):\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$")

ParseChromevolResReturn = Tuple[
    Optional[int],
    Optional[float],
    Optional[float],
    Dict[str, List[float]],
    Optional[int],
]


def read_families_file(families_file: Path) -> List[str]:
    with open(families_file, "r", encoding="utf-8") as f:
        families: List[str] = [line.strip() for line in f if line.strip()]

    if not families:
        raise ValueError(f"No families found in file: {families_file}")

    return families


def parse_configuration_metadata(configuration: str) -> Tuple[str, Optional[str], str]:
    if configuration == M0_LABEL:
        return M0_LABEL, None, LABEL_CONSTANT

    parts: List[str] = configuration.split("_")
    if len(parts) != 3:
        raise ValueError(
            f"Invalid non-{M0_LABEL} configuration format: {configuration}. "
            f"Expected format like M1_linear_gain"
        )

    model_type: str = parts[0]
    label_func_type: str = parts[1]
    label_tested_transition: str = parts[2]

    if model_type not in MODEL_TYPES:
        raise ValueError(f"Invalid model type in configuration: {configuration}")

    if label_func_type not in LABEL_FUNCTIONS_LST:
        raise ValueError(f"Invalid function type in configuration: {configuration}")

    if label_tested_transition not in LABEL_TRANSITIONS_ORDERED:
        raise ValueError(
            f"Invalid tested transition in configuration: {configuration}. "
            f"Got '{label_tested_transition}', expected one of {LABEL_TRANSITIONS_ORDERED}"
        )

    return model_type, label_tested_transition, label_func_type

def parse_chromevol_res_file(chromevol_res_path: Path) -> ParseChromevolResReturn:
    with open(chromevol_res_path, "r", encoding="utf-8") as f:
        file_text: str = f.read()

    root_match: Optional[re.Match[str]] = ROOT_PATTERN.search(file_text)
    likelihood_match: Optional[re.Match[str]] = LIKELIHOOD_PATTERN.search(file_text)
    aicc_match: Optional[re.Match[str]] = AICC_PATTERN.search(file_text)

    root_chrom_num: Optional[int] = int(root_match.group(1)) if root_match else None
    likelihood: Optional[float] = float(likelihood_match.group(1)) if likelihood_match else None
    aicc: Optional[float] = float(aicc_match.group(1)) if aicc_match else None

    base_chrom_num: Optional[int] = None
    params_by_transition: Dict[str, Dict[int, float]] = {transition_label: {} for transition_label in LABEL_TRANSITIONS_ORDERED}

    for raw_line in file_text.splitlines():
        line: str = raw_line.strip()

        base_match: Optional[re.Match[str]] = BASE_CHROM_NUM_PATTERN.match(line)
        if base_match:
            base_chrom_num = int(round(float(base_match.group(1))))
            continue

        param_match: Optional[re.Match[str]] = TRANSITION_PARAM_PATTERN.match(line)
        if not param_match:
            continue

        raw_name: str = param_match.group(1)
        raw_index: str = param_match.group(2)
        raw_value: str = param_match.group(3)

        if raw_name not in TRANSITIONS_CE_RES_TO_LABEL:
            raise ValueError(f"Unknown ChromEvol transition name in result file: {raw_name}")

        transition_label: str = TRANSITIONS_CE_RES_TO_LABEL[raw_name]
        param_index: int = int(raw_index) if raw_index else 0
        value: float = float(raw_value)

        params_by_transition[transition_label][param_index] = value

    all_params_dict: Dict[str, List[float]] = {
        transition_label: [value for _, value in sorted(param_dict.items(), key=lambda item: item[0])]
        for transition_label, param_dict in params_by_transition.items()
    }

    return root_chrom_num, likelihood, aicc, all_params_dict, base_chrom_num


def parse_expectations_file(expectations_file_path: Path) -> Dict[str, Optional[float]]:
    expectations: Dict[str, Optional[float]] = {
        EXP_GAIN_COL: None,
        EXP_LOSS_COL: None,
        EXP_DUPL_COL: None,
        EXP_DEMI_COL: None,
        EXP_BASE_NUM_COL: None,
    }

    with open(expectations_file_path, "r", encoding="utf-8") as f:
        file_text: str = f.read()

    for raw_line in file_text.splitlines():
        line: str = raw_line.strip()
        expectation_match: Optional[re.Match[str]] = EXPECTATION_PATTERN.match(line)
        if not expectation_match:
            continue

        ce_output_transition_name: str = expectation_match.group(1)
        exp_num_of_events: float = float(expectation_match.group(2))

        if ce_output_transition_name not in CE_OUTPUT_TO_EXP_COL_DICT:
            raise ValueError(f"Unknown expectation transition name: {ce_output_transition_name}")

        expectations[CE_OUTPUT_TO_EXP_COL_DICT[ce_output_transition_name]] = exp_num_of_events

    return expectations

def extract_tested_transition_params(all_params_dict: Dict[str, List[float]], tested_transition_label: Optional[str]) -> Tuple[int, Optional[float], Optional[float], Optional[float]]:
    if tested_transition_label is None:
        return 0, None, None, None

    params: List[float] = all_params_dict.get(tested_transition_label, [])
    n_params: int = len(params)

    param_0: Optional[float] = params[0] if n_params > 0 else None
    param_1: Optional[float] = params[1] if n_params > 1 else None
    param_2: Optional[float] = params[2] if n_params > 2 else None

    return n_params, param_0, param_1, param_2


def parse_single_family_run(family_name: str, family_raw_results_dir: Path, configuration: str, label_tested_transition: Optional[str], label_func_type: str) -> Dict[str, Any]:
    row: Dict[str, Any] = {col: None for col in PARSED_RESULTS_COLS}

    row[FAMILY_NAME_COL] = family_name
    row[CONFIG_COL] = configuration
    row[LABEL_TESTED_TRANSITION_COL] = label_tested_transition
    row[LABEL_FUNC_TYPE_COL] = label_func_type

    if not family_raw_results_dir.exists():
        raise FileNotFoundError(f"Missing family raw results directory: {family_raw_results_dir}")

    results_dir: Path = family_raw_results_dir / "Results"
    if not results_dir.exists():
        raise FileNotFoundError(f"Missing Results directory: {results_dir}")

    chromevol_res_path: Path = results_dir / "chromEvol.res"
    expectations_file_path: Path = results_dir / "expectations_second_round.txt"

    if not chromevol_res_path.exists():
        raise FileNotFoundError(f"Missing file: {chromevol_res_path}")

    root_chrom_num, likelihood, aicc, all_params_dict, base_chrom_num = parse_chromevol_res_file(chromevol_res_path)

    if label_func_type == LABEL_IGNORE:
        expectations = {col: None for col in [
            EXP_GAIN_COL, EXP_LOSS_COL, EXP_DUPL_COL, EXP_DEMI_COL, EXP_BASE_NUM_COL
        ]}
        expectations[TRANSITION_TO_EVENTS_COL[label_tested_transition]] = 0.0

    else:
        if not expectations_file_path.exists():
            raise FileNotFoundError(f"Missing file: {expectations_file_path}")
        expectations: Dict[str, Optional[float]] = parse_expectations_file(expectations_file_path)
    n_params, param_0, param_1, param_2 = extract_tested_transition_params(all_params_dict, label_tested_transition)

    row[ROOT_CHROM_NUM_COL] = root_chrom_num
    row[BASE_CHROM_NUM_COL] = base_chrom_num
    row[LIKELIHOOD_COL] = likelihood
    row[AICC_COL] = aicc

    row[EXP_GAIN_COL] = expectations[EXP_GAIN_COL]
    row[EXP_LOSS_COL] = expectations[EXP_LOSS_COL]
    row[EXP_DUPL_COL] = expectations[EXP_DUPL_COL]
    row[EXP_DEMI_COL] = expectations[EXP_DEMI_COL]
    row[EXP_BASE_NUM_COL] = expectations[EXP_BASE_NUM_COL]

    row[TESTED_TRANSITION_N_PARAMS_COL] = n_params
    row[PARAM_0_COL] = param_0
    row[PARAM_1_COL] = param_1
    row[PARAM_2_COL] = param_2

    row[PARAMS_JSON_COL] = json.dumps(all_params_dict, sort_keys=True)

    return row

def parse_configuration_results(families_file: Path, configuration: str, raw_results_dir: Path) -> Tuple[Path, List[str]]:
    if not raw_results_dir.exists():
        raise FileNotFoundError(f"Raw results directory does not exist: {raw_results_dir}")

    _, label_tested_transition, label_func_type = parse_configuration_metadata(configuration)
    families: List[str] = read_families_file(families_file)

    rows: List[Dict[str, Any]] = []
    failed_families: List[str] = []

    for family in families:
        family_raw_results_dir: Path = raw_results_dir / family
        try:
            row: Dict[str, Any] = parse_single_family_run(family, family_raw_results_dir, configuration, label_tested_transition, label_func_type)
            rows.append(row)
            print(f"[✓] Parsed family: {family}")
        except Exception as e:
            failed_families.append(family)
            print(f"[!] Failed parsing family {family}: {e}")

    df: pd.DataFrame = pd.DataFrame(rows)
    df = df.reindex(columns=PARSED_RESULTS_COLS)

    if not df.empty:
        df = df.sort_values(FAMILY_NAME_COL).reset_index(drop=True)

    output_dir: Path = PARSED_RESULTS_ROOT
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file_path: Path = output_dir / f"{PARSED_RESULTS_FILE_PREFIX}_{configuration}.csv"
    df.to_csv(output_file_path, index=False)

    print(f"[✓] Saved parsed results: {output_file_path}")
    print(f"[✓] Parsed successfully: {len(rows)} / {len(families)}")

    if failed_families:
        print(f"[!] Failed families ({len(failed_families)}): {', '.join(failed_families)}")

    return output_file_path, failed_families


def parse_args() -> Tuple[Path, str, Path]:
    parser = argparse.ArgumentParser(description="Parse ChromEvol result files for one configuration and save a summary CSV.")
    parser.add_argument("--families_file", required=True, type=Path, help="Path to txt file with one family name per line")
    parser.add_argument("--configuration", required=True, type=str, help="Configuration name, for example: M0_all_const or M1_linear_gain")
    parser.add_argument("--raw_results_dir", required=True, type=Path, help="Path to raw results directory of this configuration")

    args = parser.parse_args()
    return args.families_file, args.configuration, args.raw_results_dir


def main() -> None:
    families_file, configuration, raw_results_dir = parse_args()
    model_type, label_tested_transition, label_func_type = parse_configuration_metadata(configuration)
    parsed_results_file_path, failed_families = parse_configuration_results(families_file, configuration, raw_results_dir)

    if configuration == M0_LABEL:
        log_relative_path = Path(M0_LABEL) / f"{M0_LABEL}.log"
    else:
        log_relative_path = Path(model_type) / f"{configuration}.log"

    log_run(
        step="chromevol_run",
        script=Path(__file__),
        params={
            "families_file": str(families_file),
            "configuration": configuration,
            "raw_results_dir": str(raw_results_dir),
            "tested_transition": label_tested_transition,
            "function_type": label_func_type,
            "n_failed_families": len(failed_families),
            "failed_families": failed_families,
        },
        outputs=[str(parsed_results_file_path)],
        description=f"Parsed {configuration} raw ChromEvol results into CSV.",
        log_relative_path=log_relative_path,
    )


if __name__ == "__main__":
    main()