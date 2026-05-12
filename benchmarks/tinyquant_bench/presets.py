from typing import Dict, List, Optional, Tuple

PRESETS: Dict[str, Tuple[List[str], int]] = {
    "quick": (["arc_easy"], 1),
    "zero_shot": (
        ["winogrande", "piqa", "hellaswag", "arc_easy", "arc_challenge"],
        1,
    ),
    "mmlu": (["mmlu"], 5),
    "ppl": (["wikitext"], 0),
}


def resolve_tasks(spec: str) -> Tuple[List[str], Optional[int]]:
    """Resolve a preset name or csv of task names to (tasks, default_num_fewshot).

    Returns num_fewshot=None for csv specs so lm-eval-harness picks per-task defaults.
    """
    if spec in PRESETS:
        return PRESETS[spec]
    tasks = [t.strip() for t in spec.split(",") if t.strip()]
    if not tasks:
        raise ValueError(f"empty task spec: {spec!r}")
    return tasks, None
