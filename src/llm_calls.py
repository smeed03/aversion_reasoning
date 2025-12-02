import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------------------------------------
# Prompt builder for all 3 conditions A / B / C
# ---------------------------------------------------------
def build_prompt(restaurants, scenario, condition):
    desc = scenario.get("description", "")

    restaurant_block = "\n".join(
        f"- {r['name']} | rating {r.get('rating', '?')} | price {r.get('price', '?')} "
        f"| categories: {', '.join(r.get('categories', []))}"
        for r in restaurants
    )

    prefs_text = scenario.get("preferences", {})
    aversions_text = scenario.get("aversions", {})

    if condition == "A":
        return f"""
You are a restaurant recommendation assistant.

Here is a list of available restaurants:
{restaurant_block}

User preferences (NO aversions given): {prefs_text}

Recommend 5 restaurants from this list.
Return a numbered list with a one-sentence justification for each.
"""

    if condition == "B":
        return f"""
You are a restaurant recommendation assistant.

Here is a list of available restaurants:
{restaurant_block}

User preferences: {prefs_text}
User aversions: {aversions_text}

Recommend 5 restaurants that respect ALL aversions.
Return a numbered list with a one-sentence justification for each.
"""

    if condition == "C":
        return f"""
You are a careful restaurant recommendation assistant.

Here is a list of available restaurants:
{restaurant_block}

User preferences: {prefs_text}
User aversions: {aversions_text}

First, think step-by-step in bullet points:
1. Restate preferences and aversions.
2. Filter restaurants based on aversions.
3. Select the top 5 matching restaurants.

Then, at the VERY END, output ONLY valid JSON:
{{"recommendations": ["Name1", "Name2", "Name3", "Name4", "Name5"]}}
"""

    raise ValueError(f"Unknown condition: {condition}")

# ---------------------------------------------------------
# Wrapper for Chat Completions API
# ---------------------------------------------------------
def call_llm(prompt, model="gpt-4.1-mini"):
    """
    Uses ChatCompletion API (for older SDK compatibility).
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful reasoning assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    # Some older SDK versions return: response.choices[0].message.content
    return response.choices[0].message.content

