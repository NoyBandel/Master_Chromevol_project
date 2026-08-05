"""
Transition-event extraction from ChromEvol simulatedEvolutionPaths.txt files.

Original script by Anat Shafir, adapted and extended by Noy Bandel for the
ChromEvol simulation-analysis pipeline.

Expected input structure:
The simulations directory must contain numbered subdirectories from 0 to
number_of_simulations - 1. Each simulation directory must contain:

```
<simulations_dir>/
├── 0/
│   └── Results/
│       └── simulatedEvolutionPaths.txt
├── 1/
│   └── Results/
│       └── simulatedEvolutionPaths.txt
└── ...
```

For example, when number_of_simulations is 100, the script expects simulation
directories named 0 through 99.
"""

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd

from source_code.constants import *
from source_code.simulations.simulation_constants import *
from source_code.logger import log_run


def get_number_of_true_transitions(evo_path: Path, base_number: int) -> Dict[str, int]:
    transition_types_events = {transition: 0 for transition in TRANSITION_TO_TRUE_EVENTS_COL}
    pattern = re.compile(r"from state:\s+(\d+)\s+t\s*=\s*\S+\s+to state\s*=\s*(\d+)")
    content = evo_path.read_text(encoding="utf-8")
    from_to_states = pattern.findall(content)

    for from_state_str, to_state_str in from_to_states:
        from_state = int(from_state_str)
        to_state = int(to_state_str)

        if from_state + 1 == to_state:
            transition_types_events[LABEL_GAIN] += 1
        elif from_state - 1 == to_state:
            transition_types_events[LABEL_LOSS] += 1
        elif from_state * 2 == to_state:
            transition_types_events[LABEL_DUPL] += 1
        elif from_state % 2 == 0 and to_state == 1.5 * from_state:
            transition_types_events[LABEL_DEMI] += 1
        elif from_state % 2 != 0 and to_state in {math.floor(from_state * 1.5), math.ceil(from_state * 1.5)}:
            transition_types_events[LABEL_DEMI] += 1
        elif base_number > 0 and (to_state - from_state) % base_number == 0:
            transition_types_events[LABEL_BASE_NUM] += 1

    return transition_types_events


def collect_transition_counts(simulations_dir: Path, number_of_simulations: int, base_number: int) -> pd.DataFrame:
    simulation_records: List[Dict[str, int]] = []

    for simulation_id in range(number_of_simulations):
        evo_path = simulations_dir / str(simulation_id) / RESULTS_DIR_NAME / SIMULATED_EVOLUTION_PATHS_FILENAME

        if not evo_path.is_file():
            raise FileNotFoundError(f"Could not find simulation evolution paths file: {evo_path}")

        transition_counts = get_number_of_true_transitions(evo_path, base_number)
        simulation_record = {SIMULATION_ID_COL: simulation_id}

        for transition, output_col in TRANSITION_TO_TRUE_EVENTS_COL.items():
            simulation_record[output_col] = transition_counts[transition]

        simulation_records.append(simulation_record)

    return pd.DataFrame(simulation_records, columns=TRUE_TRANSITION_COUNTS_COLS)


def save_transition_summary(transition_counts_df: pd.DataFrame, summary_file: Path, base_number: int) -> None:
    event_counts_df = transition_counts_df.drop(columns=SIMULATION_ID_COL)
    statistics_df = pd.DataFrame({
        "total": event_counts_df.sum(),
        "mean": event_counts_df.mean(),
        "std": event_counts_df.std(ddof=0),
        "min": event_counts_df.min(),
        "median": event_counts_df.median(),
        "max": event_counts_df.max(),
    })

    with summary_file.open("w", encoding="utf-8") as file:
        file.write(f"Number of simulations: {len(transition_counts_df)}\n")
        file.write(f"Base number: {base_number if base_number > 0 else 'ignored'}\n\n")
        file.write(statistics_df.to_string())
        file.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract true transition counts from ChromEvol simulations")
    parser.add_argument("--simulations_dir", "-s", type=Path, required=True, help="Directory containing numbered simulation directories")
    parser.add_argument("--number_of_simulations", "-n", type=int, required=True, help="Number of simulations")
    parser.add_argument("--base_number", "-b", type=int, required=True, help="Base number; use a negative value to ignore base-number transitions")
    parser.add_argument("--output_file", "-o", type=Path, default=None, help="Optional output CSV path")
    args = parser.parse_args()

    output_file = args.output_file or args.simulations_dir / TRUE_TRANSITION_COUNTS_FILENAME
    summary_file = output_file.parent / TRUE_TRANSITION_COUNTS_SUMMARY_FILENAME

    transition_counts_df = collect_transition_counts(args.simulations_dir, args.number_of_simulations, args.base_number)
    transition_counts_df.to_csv(output_file, index=False)
    save_transition_summary(transition_counts_df, summary_file, args.base_number)

    output_paths = [output_file.as_posix(), summary_file.as_posix()]

    log_run(
        step="simulation_transition_counts",
        script=Path(__file__),
        params={
            "simulations_dir": args.simulations_dir,
            "number_of_simulations": args.number_of_simulations,
            "base_number": args.base_number,
            "output_file": output_file,
        },
        outputs=output_paths,
        description="Extracted true transition-event counts from ChromEvol simulations.",
        notes="Saved per-simulation event counts to CSV and aggregate event-count statistics to TXT.",
    )

    print(f"Transition counts written to: {output_file}")
    print(f"Transition summary written to: {summary_file}")


if __name__ == "__main__":
    main()