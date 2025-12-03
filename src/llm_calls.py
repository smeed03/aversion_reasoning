import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Prompt builder for all 3 conditions (A, B, C)
def build_prompt(restaurants, scenario, condition):
    desc = scenario.get("description", "")

    # Format restaurants into lines of bullets with attributes
    lines = []
    for r in restaurants:
        base = f"{r['name']} | rating {r.get('rating', '?')}/5 | price {r.get('price', '?')} | neighborhood: {r.get('neighborhood', '?')}"
        categories = ", ".join(r.get("categories", []))
        tags = ", ".join(r.get("tags", []))

        extras = []
        if r.get("dog_friendly"):
            extras.append("dog-friendly")
        if r.get("outdoor_seating"):
            extras.append("outdoor seating")
        if r.get("wheelchair_accessible"):
            extras.append("wheelchair accessible")
        if r.get("kid_friendly"):
            extras.append("kid-friendly")
        if r.get("serves_alcohol"):
            extras.append("serves alcohol")
        if r.get("spicy_focus"):
            extras.append("spicy-focused")
        noise = r.get("noise_level")
        if noise:
            extras.append(f"noise level: {noise}")
        if r.get("is_chain"):
            extras.append("chain restaurant")

        extras_str = ", ".join(extras) if extras else "no notable extra attributes"

        line = f"- {base} | categories: {categories} | tags: {tags} | {extras_str}"
        lines.append(line)

    restaurant_block = "\n".join(lines)

    prefs_text = scenario.get("preferences", {})
    aversions_text = scenario.get("aversions", {})

    if condition == "A":
        return f"""
You are a restaurant recommendation assistant.

Here is a list of available restaurants:
{restaurant_block}

User description: {desc}

User preferences (NO aversions given explicitly): {prefs_text}

Recommend 5 restaurants from this list.
Return a numbered list with a one-sentence justification for each.
"""

    if condition == "B":
        return f"""
You are a restaurant recommendation assistant.

Here is a list of available restaurants:
{restaurant_block}

User description: {desc}

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

User description: {desc}

User preferences: {prefs_text}
User aversions: {aversions_text}

First, think step-by-step in bullet points:
1. Restate preferences and aversions.
2. Filter restaurants based on aversions (explaining which ones you remove and why).
3. From the remaining restaurants, select the top 5 that best satisfy preferences.

Then, at the VERY END, output ONLY valid JSON on a SINGLE LINE:
{{"recommendations": ["Name1", "Name2", "Name3", "Name4", "Name5"]}}
"""

    raise ValueError(f"Unknown condition: {condition}")

# API completions wrapper
def call_llm(prompt, model="gpt-4.1-mini"):
    """
    Uses ChatCompletion API (for older SDK compatibility).
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful reasoning assistant that carefully follows user constraints."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content
