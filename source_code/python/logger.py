from datetime import datetime
from typing import Any

from .constants import *


def log_run(step: str, script: Path, params: dict[str, Any], outputs: list[str], description: str, notes: str | None = None) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    # --- project log ---
    project_log = LOGS_ROOT / "project.log"
    project_log.parent.mkdir(parents=True, exist_ok=True)
    run_mode = params.get("run", "NA")
    project_log.open("a", encoding="utf-8").write(
        f"{ts} | {step} | run={run_mode} | {script.name} | {description}\n"
    )

    # --- step log ---
    step_dir = LOGS_ROOT / step
    step_dir.mkdir(parents=True, exist_ok=True)
    step_log = step_dir / f"{step}.log"

    block = []
    block.append("=" * 80)
    block.append(f"DATE        : {ts}")
    block.append(f"SCRIPT      : {script.as_posix()}")
    block.append("")
    block.append("PARAMETERS:")
    for k, v in params.items():
        block.append(f"  - {k}: {v}")
    block.append("")
    block.append("OUTPUTS:")
    for o in outputs:
        block.append(f"  - {o}")
    block.append("")
    block.append("DESCRIPTION:")
    block.append(f"  {description}")
    if notes:
        block.append("")
        block.append("NOTES:")
        block.append(f"  {notes}")

    block.append("")

    step_log.open("a", encoding="utf-8").write("\n".join(block))