import pandas as pd

def load_data(path="../results/annotated_results.csv"):
    df = pd.read_csv(path)
    return df

def compute_basic_metrics(df):
    metrics = {}

    metrics['total_rows'] = len(df)

    metrics['pref_accuracy'] = df['pref_ok'].mean()
    metrics['aversion_accuracy'] = df['aversion_ok'].mean()
    metrics['joint_accuracy'] = ((df['pref_ok'] == 1) & (df['aversion_ok'] == 1)).mean()

    metrics['avg_reasoning_score'] = df['reasoning_score'].mean()
    metrics['avg_cuisine_diversity'] = df['diversity_cuisines'].mean()
    metrics['avg_neighborhood_diversity'] = df['diversity_neighborhoods'].mean()

    return metrics

def compute_breakdowns(df):
    # Group by model
    by_model = df.groupby("model").agg({
        "pref_ok": "mean",
        "aversion_ok": "mean",
        "reasoning_score": "mean",
        "diversity_cuisines": "mean",
        "diversity_neighborhoods": "mean"
    })

    # Group by prompt_condition
    by_condition = df.groupby("prompt_condition").agg({
        "pref_ok": "mean",
        "aversion_ok": "mean",
        "reasoning_score": "mean",
        "diversity_cuisines": "mean",
        "diversity_neighborhoods": "mean"
    })

    # Group by scenario
    by_scenario = df.groupby("scenario_id").agg({
        "pref_ok": "mean",
        "aversion_ok": "mean",
        "reasoning_score": "mean"
    })

    # Model × prompt_condition interaction
    model_condition = df.groupby(["model", "prompt_condition"]).agg({
        "pref_ok": "mean",
        "aversion_ok": "mean",
        "reasoning_score": "mean",
        "diversity_cuisines": "mean",
        "diversity_neighborhoods": "mean"
    })

    # Scenario × prompt_condition interaction
    scenario_condition = df.groupby(["scenario_id", "prompt_condition"]).agg({
        "pref_ok": "mean",
        "aversion_ok": "mean",
        "reasoning_score": "mean",
    })

    return {
        "by_model": by_model,
        "by_condition": by_condition,
        "by_scenario": by_scenario,
        "model_condition": model_condition,
        "scenario_condition": scenario_condition
    }

def print_summary(basic_metrics, breakdowns):
    print("\n================= SUMMARY METRICS =================\n")

    print("Total rows:", basic_metrics['total_rows'])
    print(f"Preference accuracy: {basic_metrics['pref_accuracy']:.3f}")
    print(f"Aversion accuracy: {basic_metrics['aversion_accuracy']:.3f}")
    print(f"Joint accuracy (pref + aversion): {basic_metrics['joint_accuracy']:.3f}")

    print(f"\nAverage reasoning score: {basic_metrics['avg_reasoning_score']:.3f}")
    print(f"Avg cuisine diversity: {basic_metrics['avg_cuisine_diversity']:.3f}")
    print(f"Avg neighborhood diversity: {basic_metrics['avg_neighborhood_diversity']:.3f}")

    print("\n================= BREAKDOWNS =================\n")

    print("---- Accuracy by Model ----")
    print(breakdowns['by_model'])
    print("\n")

    print("---- Accuracy by Condition (A, B, C) ----")
    print(breakdowns['by_condition'])
    print("\n")

    print("---- Accuracy by Scenario (S1-S6) ----")
    print(breakdowns['by_scenario'])
    print("\n")

    print("---- Model x Condition ----")
    print(breakdowns['model_condition'])
    print("\n")

    print("---- Scenario x Condition ----")
    print(breakdowns['scenario_condition'])
    print("\n")

def main():
    df = load_data()
    basic_metrics = compute_basic_metrics(df)
    breakdowns = compute_breakdowns(df)
    print_summary(basic_metrics, breakdowns)

if __name__ == "__main__":
    main()
