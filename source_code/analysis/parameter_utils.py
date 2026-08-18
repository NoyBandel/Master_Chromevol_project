
RAW_PARAMETER_SCALE = "raw"
CORRECTED_PARAMETER_SCALE = "corrected"

VALID_PARAMETER_SCALES = RAW_PARAMETER_SCALE, CORRECTED_PARAMETER_SCALE

TREE_SCALE_CORRECTED_SUFFIX = "_tree_scale_corrected"


def resolve_parameter_column(raw_column: str,parameter_scale: str) -> str:
    if parameter_scale == RAW_PARAMETER_SCALE:
        return raw_column

    if parameter_scale == CORRECTED_PARAMETER_SCALE:
        return f"{raw_column}{TREE_SCALE_CORRECTED_SUFFIX}"

    raise ValueError(
        f"Unsupported parameter scale '{parameter_scale}'. "
        f"Expected one of {VALID_PARAMETER_SCALES}."
    )