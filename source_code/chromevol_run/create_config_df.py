import pandas as pd
from ..constants import *

# A chromevol run configuration is a tuple:
# pd.DataFrame - A configuration dataframe where columns are CONFIG_DF_COLS
# str - An initial base chromosome number value

# Models:
# ---M0---
# - All transitions are of constant function
# - Params initialized from DEFAULT_INIT_VALUES

# ---M1--- - All transitions are constant, and init values are set to the inferred value of M0
#       except for the tested_transition which is set to a linear function with parameters: slope = 0.1 and intersection = inferred value of M0

def create_M0_config() -> tuple[pd.DataFrame, str]:
    config_df = pd.DataFrame({
        LABEL_TRANSITION_COL: LABEL_TRANSITIONS_ORDERED,
        LABEL_FUNC_TYPE_COL: [LABEL_CONSTANT] * len(LABEL_TRANSITIONS_ORDERED),
        CE_TRANSITION_COL: CE_TRANSITIONS_ORDERED,
        CE_FUNC_TYPE_COL: [CE_CONSTANT] * len(CE_TRANSITIONS_ORDERED),
        CE_TRANSITION_INIT_COL: CE_TRANSITIONS_INIT_ORDERED,
        CE_PARAMS_COL: [DEFAULT_INIT_VALUES[CE_CONSTANT]] * len(CE_TRANSITIONS_ORDERED),
    })
    base_num_init = DEFAULT_INIT_VALUES[CE_BASE_CHROM_NUM_INIT]
    return config_df, base_num_init


def create_M1_config_df_from_M0_output(tested_transition: str, M0_results_summary_file_path: Path) -> pd.DataFrame:

### M0_results_summary_file_path needs to have a clear uniform format

    # transition to function dictionaries, by run type configuration
    #
    # def const_except_for_tested_transition_to_function_dict(tested_transition_type: str, tested_function_type: str) -> \
    # Dict[str, str]:
    #     if tested_transition_type not in CE_TRANSITIONS_ORDERED:
    #         raise ValueError(
    #             f"tested_transition_type must be one of {CE_TRANSITIONS_ORDERED}, got: {tested_transition_type}")
    #     if tested_function_type not in CE_FUNCTIONS:
    #         raise ValueError(f"tested_function_type must be one of {CE_FUNCTIONS}, got: {tested_function_type}")
    #
    #     transition_to_function_dict = {}
    #     for transition in CE_TRANSITIONS_ORDERED:
    #         if transition == tested_transition_type:
    #             transition_to_function_dict[transition] = tested_function_type
    #         else:
    #             transition_to_function_dict[transition] = CE_CONSTANT
    #     return transition_to_function_dict
    return



# chosen except for tested note:
   # if transition_to_function_dict[CE_BASE_NUM] == CE_IGNORE:
   #      start_point_values[CE_BASE_CHROM_NUM_INIT] = ""


def create_configuration(configuration_type: str, tested_transition: str, config_data_file: Path) -> pd.DataFrame:
    config_type_to_func = {M0_LABEL: create_M0_config(),
                           M1_LABEL: create_M1_config_df_from_M0_output(tested_transition, config_data_file)}
    output_df: pd.DataFrame = config_type_to_func[configuration_type]
    return output_df
