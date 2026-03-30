from __future__ import annotations

from baseline.run_baseline import run_all_tasks


def run_inference():
    """
    Root-level inference entrypoint expected by some OpenEnv checkers.
    Delegates to the repository baseline implementation.
    """
    return run_all_tasks()


def main() -> None:
    res = run_inference()
    for name, score in res.get("task_scores", {}).items():
        print(f"{name}: {score:.4f}")
    print(f"overall: {res.get('overall', 0.0):.4f}")


if __name__ == "__main__":
    main()

