from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from .constants import *

def log_run(step: str, script: Path, params: dict[str, Any], outputs: list[str], description: str, notes: str = "", log_relative_path: Optional[Path] = None) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    # project log
    project_log: Path = LOGS_ROOT / "project.log"
    project_log.parent.mkdir(parents=True, exist_ok=True)
    run_mode: Optional[str] = params.get("run", None)

    log: str = f"{ts} | {step} "
    if run_mode:
        log += f"| run:{run_mode} "
    log += f"| {script.name} | {description}\n"

    project_log.open("a", encoding="utf-8").write(log)

    # step log
    step_dir: Path = LOGS_ROOT / step

    if log_relative_path is None:
        step_log: Path = step_dir / f"{step}.log"
    else:
        step_log = step_dir / log_relative_path

    step_log.parent.mkdir(parents=True, exist_ok=True)

    block: list[str] = []
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