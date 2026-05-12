import json
import os
from typing import Any, Dict, Optional


def write_output(path: Optional[str], data: Dict[str, Any]) -> None:
    text = json.dumps(data, indent=2, default=str)
    if path:
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w") as f:
            f.write(text)
        print(f"wrote {path}")
    else:
        print(text)


def maybe_log_wandb(
    project: Optional[str],
    name: Optional[str],
    config: Dict[str, Any],
    metrics: Dict[str, Any],
) -> None:
    if not project:
        return
    import wandb

    wandb.init(project=project, name=name, config=config)
    wandb.log(metrics)
    wandb.finish()
