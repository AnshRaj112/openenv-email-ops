import json
from pathlib import Path

FILE = Path("data/leaderboard.json")


def _read_data():
    if not FILE.exists():
        return []
    try:
        with FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return []
    except (json.JSONDecodeError, OSError):
        return []


def save_score(model, score):
    FILE.parent.mkdir(parents=True, exist_ok=True)
    data = _read_data()

    data.append({"model": model, "score": score})
    data = sorted(data, key=lambda x: x["score"], reverse=True)[:10]

    with FILE.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_leaderboard():
    return _read_data()
