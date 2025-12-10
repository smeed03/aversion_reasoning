# Aversion Reasoning in LLM Restaurant Recommendations

This project explores how LLMs reason about user preferences and aversions when making recommendations. We evaluate two models across six scenarios and three prompting 
conditions to investigate how well they filter, generate, and justify recommendations under considering different user constraints.


## Project Structure

aversion_reasoning/
│
├── data/
│   ├── restaurants_baltimore.json  # List of restaurants
│   ├── scenarios.json              # List of scenarios (example users)
│
├── notebooks/
│   ├── exploration.ipynb           # Visualizations & analysis of results
│
├── results/
│   ├── raw_responses               # Contains one folder for recommendations made with GPT-4.1-mini, and another for 3.5-turbo
│   ├── annotated_results.csv       # Evaluation dataset for raw responses
│
├── src/
│   ├── llm_calls.py             # Outlines different prompting conditions and sets up use of OpenAI API
│   ├── analyze_results.py       # Computes summary metrics
│   ├── load_data.py             # Loads restaurants and scenarios
│   ├── run_experiments.py       # Runs different scenario and condition pairs for models and stores the results accordingly
│
├── requirements.txt             # Dependencies
└── README.md


## Experiments Overview

We test:

Two LLMs:

gpt-3.5-turbo
gpt-4.1-mini

Six scenarios (S1–S6) with different user constraints

Three prompting conditions:

A: Simple recommendation
B: Explicit aversions
C: Chain-of-thought

Metrics recorded:

Preference accuracy
Aversion accuracy
Joint accuracy
Reasoning quality score
Cuisine & neighborhood diversity


## Running Analysis

From root directory:

cd src
python analyze_results.py

This loads results/annotated_results.csv and prints:

overall accuracy metrics
breakdowns by model
breakdowns by scenario
breakdowns by prompting conditions


## Generate Visualizations

Open notebooks/exploration.ipynb

Run all cells to output:

accuracy bar graphs
scenario difficulty analysis
interaction between models and conditions
diversity visualizations


## Installation

Create an environment:

conda create -n aversion_env python=3.10
conda activate aversion_env

Install dependencies:

pip install -r requirements.txt


## Running Experiments

To generate new raw model outputs:

python src/run_experiments.py

This will create JSON files inside:

results/raw_responses

You must set your OpenAI API key in an .env file:

OPENAI_API_KEY=your_key_here
