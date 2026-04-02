import pandas as pd


from source_code.constants import *
from source_code.logger import log_run

#-------------
def extract_params(row, tested_transition):
    params_dict = json.loads(row[ALL_PARAMS_DICT_STR_COL])
    params = params_dict.get(tested_transition, [])
    n = len(params)
    p0 = params[0] if n > 0 else None
    p1 = params[1] if n > 1 else None
    p2 = params[2] if n > 2 else None

    return pd.Series([n, p0, p1, p2])

def build_transition_results_file(parsed_results_file: Path, tested_transition: str, output_file: Path) -> None:
    df = pd.read_csv(parsed_results_file)
    df[LABEL_TESTED_TRANSITION_COL] = tested_transition
    df[[TESTED_TRANSITION_N_PARAMS_COL, PARAM_0_COL, PARAM_1_COL, PARAM_2_COL]] = df.apply( lambda row: extract_params(row, tested_transition), axis=1)
    df.to_csv(output_file, index=False)

def parse_args() -> Tuple[str, Path, str, Path, str, str]:
    parser = argparse.ArgumentParser(
        description="Parse ChromEvol result files for one configuration and save a summary CSV.")
    parser.add_argument("--run_type", required=True,
                        choices=["parse_configuration_results", "build_transition_results_file"],
                        help="Which parsing function to run")
    parser.add_argument("--families_file", required=True, type=Path,
                        help="Path to txt file with one family name per line")
    parser.add_argument("--configuration", required=True, type=str, help="Configuration name")
    parser.add_argument("--raw_results_dir", required=True, type=Path,
                        help="Path to raw results directory of this configuration")
    parser.add_argument("--label_tested_transition", required=True, type=str, help="Tested transition label")
    parser.add_argument("--label_func_type", required=True, type=str, help="Function type label")

    args = parser.parse_args()

    return args.run_type, args.families_file, args.configuration, args.raw_results_dir, args.label_tested_transition, args.label_func_type

def main() -> None:
    run_type, families_file, configuration, raw_results_dir, label_tested_transition, label_func_type = parse_args()

    parsed_results_file_path: Path = (PARSED_RESULTS_ROOT / configuration / f"{PARSED_RESULTS_FILE_PREFIX}_{configuration}.csv")
    transition_results_file: Path = (parsed_results_file_path.parent / f"transition_results_{configuration}.csv")

    build_transition_results_file(parsed_results_file_path, label_tested_transition, transition_results_file)

    model_type = M0_LABEL if configuration == M0_LABEL else configuration.split("_")[0]
    log_relative_path = Path(model_type) / f"{configuration}.log"

    log_run(
        step="chromevol_run",
        script=Path(__file__),
        params={
            "run": "build_transition_results_file",
            "parsed_results_file": parsed_results_file_path,
            "tested_transition": label_tested_transition,
            "output_file": transition_results_file,
            "configuration": configuration,
        },
        outputs=[str(transition_results_file)],
        description="Extracted tested transition parameters from parsed results.",
        log_relative_path=log_relative_path,
    )