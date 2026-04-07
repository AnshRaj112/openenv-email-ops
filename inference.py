from __future__ import annotations

from baseline.run_baseline import HardTask, EasyTask, MediumTask, run_task


def run_inference():
    """
    Root-level inference entrypoint expected by some OpenEnv checkers.
    Emits structured stdout blocks expected by OpenEnv validators.
    """
    task_scores = {}
    for task in [EasyTask(), MediumTask(), HardTask()]:
        print(f"[START] task={task.name}", flush=True)
        score = float(run_task(task))
        # Baseline task runner returns a final episode score only.
        print(f"[STEP] step=1 reward={score:.4f}", flush=True)
        print(f"[END] task={task.name} score={score:.4f} steps=1", flush=True)
        task_scores[task.name] = score
    overall = sum(task_scores.values()) / len(task_scores) if task_scores else 0.0
    return {"task_scores": task_scores, "overall": overall}


def main() -> None:
    run_inference()


if __name__ == "__main__":
    main()

