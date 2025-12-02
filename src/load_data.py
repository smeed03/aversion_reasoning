import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

def load_restaurants(filename="restaurants_baltimore.json"):
    path = DATA_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_scenarios(filename="scenarios.json"):
    path = DATA_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
