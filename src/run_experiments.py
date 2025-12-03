import json
from pathlib import Path
from datetime import datetime
from load_data import load_restaurants, load_scenarios
from llm_calls import build_prompt, call_llm

# Folder to store raw model outputs/reccomendations
BASE_RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "raw_responses"
BASE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_one_experiment(restaurants, scenario, condition, model="gpt-4.1-mini"):
    """
    Runs model on single (scenario, condition) pair,
    saves prompt + output to text file under model-specific folder.
    """
    scenario_id = scenario["id"]

    prompt = build_prompt(restaurants, scenario, condition)
    output_text = call_llm(prompt, model=model)

    # Create a folder for each model
    model_dir = BASE_RESULTS_DIR / model
    model_dir.mkdir(parents=True, exist_ok=True)

    # Structuring output filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = model_dir / f"{scenario_id}_{condition}_{timestamp}.txt"

    with open(filename, "w", encoding="utf-8") as f:
        f.write("=== MODEL ===\n")
        f.write(model)
        f.write("\n\n=== SCENARIO ID ===\n")
        f.write(scenario_id)
        f.write("\n\n=== CONDITION ===\n")
        f.write(condition)
        f.write("\n\n=== PROMPT ===\n")
        f.write(prompt)
        f.write("\n\n=== MODEL OUTPUT ===\n")
        f.write(output_text)

    print(f"Saved: {filename}")

def main():
    restaurants = load_restaurants()
    scenarios = load_scenarios()

    models_to_test = [
        "gpt-4.1-mini",
        "gpt-3.5-turbo"
    ]
    conditions = ["A", "B", "C"]  # preference only, aversions, CoT

    for scenario in scenarios:
        for condition in conditions:
            for model in models_to_test:
                print(f"\nRunning {scenario['id']} condition {condition} model {model}...")
                run_one_experiment(restaurants, scenario, condition, model)


if __name__ == "__main__":
    main()
