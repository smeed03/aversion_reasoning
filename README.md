# Reasoning with Aversions: Evaluating How Generative Models Reason Through Preference Constraints in Real-World Recommendations

## 📖 Project Overview
This project evaluates how well LLMs generate recommendations when users specify both **preferences** (what they want) and **aversions** (what they want to avoid).

Using a controlled dataset of structured restaurant data and user scenarios, we analyze how different prompting strategies and models impact:
- Constraint satisfaction  
- Reasoning quality  
- Recommendation diversity

---

## 🎯 Purpose / Problem
Most recommendation systems prioritize **positive preferences** while neglecting **negative constraints (aversions)**.

This leads to:
- Recommendations that violate user requirements  
- Poor personalization  
- Potentially unsafe or insensitive suggestions  

This project investigates:
- Can LLMs **reason over both preferences and aversions**?
- Does **explicit reasoning** improve performance?
- What happens when **constraints conflict**?

---

## ⚙️ Technologies Used
- Python
- Jupyter Notebook  
- OpenAI API
- JSON datasets

---

## 📊 Dataset & Scenarios

### Restaurants
- 21 structured restaurant entries  
- Attributes include:
  - cuisine, price, rating  
  - tags (e.g., *romantic*, *lively*)  
  - noise level, accessibility  
  - chain status, neighborhood  

Example:
```json
{
  "name": "Golden Lotus",
  "categories": ["Vietnamese", "Vegetarian"],
  "price": "$$",
  "noise_level": "quiet",
  "wheelchair_accessible": true
}
```

---

### Scenarios
Each scenario defines:
- structured preferences  
- structured aversions  

Example:
```json
{
  "id": "S4",
  "description": "Wants a sports bar but dislikes loud environments",
  "preferences": { "categories_any_of": ["Sports Bar", "Pub"] },
  "aversions": { "noise_level_any_of": ["loud"] }
}
```

---

## 🧪 Experimental Design

### Prompting Conditions
- **A**: Preferences only (implicit reasoning)  
- **B**: Preferences + aversions  
- **C**: Step-by-step reasoning + structured JSON output  

### Models Tested
- `gpt-4.1-mini`  
- `gpt-3.5-turbo`  

---

## 🔧 How It Works (Pipeline)

1. Load data  
   - `load_data.py` reads JSON datasets  

2. Build prompts  
   - `llm_calls.py` formats restaurants + scenarios into structured prompts  

3. Run experiments  
   - `run_experiments.py` loops through:
     - all scenarios  
     - all prompting conditions  
     - all models  
   - Saves outputs to timestamped files  

4. Analyze results  
   - `analyze_results.py` computes:
     - preference accuracy  
     - aversion accuracy  
     - joint accuracy  
     - reasoning score  
     - diversity metrics  

---

## 🚀 Setup & Installation

### 1. Clone repo
```bash
git clone https://github.com/smeed03/aversion_reasoning.git
cd aversion_reasoning
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set OpenAI API key
```bash
export OPENAI_API_KEY="your_api_key_here"
```

---

## ▶️ Usage

### Run experiments
```bash
python src/run_experiments.py
```

Output:
```
results/raw_responses/<model>/<scenario>_<condition>_<timestamp>.txt
```

Each file contains:
- model  
- scenario  
- prompt  
- raw LLM output  

---

### Analyze results
```bash
python src/analyze_results.py
```

Outputs:
- overall accuracy metrics  
- breakdowns by:
  - model  
  - condition  
  - scenario  

---

### Explore
```bash
jupyter notebook notebooks/exploration.ipynb
``` 

---

## 👤 My Role
- Designed experimental framework  
- Created dataset + user scenarios  
- Implemented prompt engineering and pipeline  
- Ran experiments and analyzed results  
- Co-authored research paper  

---

## 🧠 Reflection / Lessons Learned
- LLMs are better at **avoiding bad options** than selecting optimal ones  
- Explicit reasoning improves constraint satisfaction but reduces diversity  
- Models struggle with **contradictory constraints**  
- LLMs behave more like **filters** than true recommendation optimizers
