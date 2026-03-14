import argparse
from pathlib import Path
import pandas as pd

from source_code.constants import *
from source_code.logger import log_run
from source_code.prepare_data_and_config_for_runs.model_specific_utils import (
    build_M0_all_const_configuration_file,
    build_M1_configuration_file,
    build_M2_configuration_file
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODEL_TYPES)
    parser.add_argument("--input_parsed_results_file", required=True, type=Path)
    parser.add_argument("--families_file", required=True, type=Path, help="TXT file with one family name per line.")
    parser.add_argument("--tested_transition_label", required=False, choices=LABEL_TRANSITIONS_ORDERED)
    parser.add_argument("--tested_function_label", required=False, choices=LABEL_FUNCTIONS_LST)
    return parser.parse_args()

def build_output_file(model_label: str, tested_transition_label: str | None, tested_function_label: str | None) -> Path:
    if model_label == M0_LABEL:
        return MODEL_SPECIFIC_CONFIG_ROOT / f"{M0_LABEL}.csv"
    return MODEL_SPECIFIC_CONFIG_ROOT / f"{model_label}_{tested_function_label}_{tested_transition_label}.csv"


def main():
    args = parse_args()
    if args.model == M0_LABEL:
        config_df = build_M0_all_const_configuration_file(args.input_parsed_results_file, args.families_file)

    elif args.model == M1_LABEL:
        if args.tested_transition_label is None or args.tested_function_label is None:
            raise ValueError("M1 requires --tested_transition_label and --tested_function_label")
        config_df = build_M1_configuration_file(args.input_parsed_results_file, args.families_file, args.tested_transition_label, args.tested_function_label)

    elif args.model == M2_LABEL:
        if args.tested_transition_label is None or args.tested_function_label is None:
            raise ValueError("M2 requires --tested_transition_label and --tested_function_label")
        config_df = build_M2_configuration_file(args.input_parsed_results_file, args.families_file, args.tested_transition_label, args.tested_function_label)

    else:
        raise ValueError(f"Unsupported model {args.model}")

    output_file = build_output_file(args.model, args.tested_transition_label, args.tested_function_label)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    config_df.to_csv(output_file, index=False)

    configuration_name = args.model if args.model == M0_LABEL else f"{args.model}_{args.tested_function_label}_{args.tested_transition_label}"
    log_relative_path = Path(args.model) / f"{configuration_name}.log"

    log_run(
        step="prepare_model_specific_config",
        script=Path(__file__),
        params={
            "model": args.model,
            "input_parsed_results_file": str(args.input_parsed_results_file),
            "families_file": str(args.families_file),
            "tested_transition_label": args.tested_transition_label,
            "tested_function_label": args.tested_function_label,
            "n_families": len(config_df),
        },
        outputs=[str(output_file)],
        description=f"Built model-specific configuration file from parsed results: {configuration_name}.",
        log_relative_path=log_relative_path,
    )

    print(f"[✓] Saved {output_file}")


if __name__ == "__main__":
    main()

