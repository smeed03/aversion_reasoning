import json
from pathlib import Path
from datetime import datetime
from load_data import load_restaurants, load_scenarios
from llm_calls import build_prompt, call_llm

# Directory to store raw model outputs
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "raw_responses"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_one_experiment(restaurants, scenario, condition, model="gpt-4.1-mini"):
    """
    Runs the model on a single (scenario, condition) pair,
    saves the prompt + output to a text file.
    """
    scenario_id = scenario["id"]

    prompt = build_prompt(restaurants, scenario, condition)
    output_text = call_llm(prompt, model=model)

    # Filename structure: S1_A_gpt-4.1-mini_TIMESTAMP.txt
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = RESULTS_DIR / f"{scenario_id}_{condition}_{model}_{timestamp}.txt"

    with open(filename, "w", encoding="utf-8") as f:
        f.write("=== PROMPT ===\n")
        f.write(prompt)
        f.write("\n\n=== MODEL OUTPUT ===\n")
        f.write(output_text)

    print(f"Saved: {filename}")


def main():
    restaurants = load_restaurants()
    scenarios = load_scenarios()

    models_to_test = ["gpt-4.1-mini"]      # expand later if needed
    conditions = ["A", "B", "C"]           # preference-only, aversions, CoT

    for scenario in scenarios:
        for condition in conditions:
            for model in models_to_test:
                print(f"\nRunning {scenario['id']} condition {condition} model {model}...")
                run_one_experiment(restaurants, scenario, condition, model)


if __name__ == "__main__":
    main()
