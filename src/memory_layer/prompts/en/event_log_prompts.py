"""
Event Log Extraction Prompts - English Version
This module contains prompts for extracting structured event logs from episodic memory text.
"""

EVENT_LOG_PROMPT = """You are an expert episodic memory extraction analyst and information architect.  
Your task is to analyze the given narrative or multi-turn conversation (called "EPISODE_TEXT") and produce an event log optimized for factual retrieval.

---

### INPUT
- EPISODE_TEXT: The memory text saved through the "episode memory".
- TIME: The start time of the episode, e.g., "March 10, 2024(Sunday) at 2:00 PM".

---

### OUTPUT
Return **only** one valid JSON object, with the following exact structure:

{
  "event_log": {
    "time": "<THE EXACT TIME STRING FROM INPUT TIME>",
    "atomic_fact": [
      "<Atomic fact sentence 1>",
      "<Atomic fact sentence 2>",
      ...
    ]
  }
}

---

### EXTRACTION RULES

#### 1. Atomicity
* Each entry in `"atomic_fact"` must express **exactly one coherent unit of meaning** — an action, emotion, reason, plan, decision, or statement.
* If a speaker expresses multiple ideas (e.g., an event and its reason), split them into multiple atomic facts.
* Each atomic_fact must be **independent and retrievable on its own**.

#### 2. Time & Date Handling
* Do **not** add or prepend timestamps such as "On March 10, 2024".
* The `"time"` field (at the top level) already represents the episode's start time.
* However, when the original text mentions **explicit or relative time expressions**, you must:
  - **Preserve** explicit dates verbatim (e.g., "October 6, 2023").
  - **Resolve** relative or vague times (e.g., "yesterday", "last week", "two months ago") relative to `TIME`, and **append the resolved absolute date in parentheses**.  
    Example: "Gina said she launched the campaign yesterday (March 9, 2024)."
  - If the exact resolution is uncertain, use a normalized vague phrase (e.g., "in early 2023", "during the summer of 2021") instead of guessing.

#### 3. Content Preservation
* Preserve **all semantically meaningful information**, including:
  - emotions, attitudes, reasons, intentions, consequences, conditions, and comparisons
* Resolve pronouns into explicit names or entities when unambiguous.
* Keep original wording as much as possible — only fix grammar for clarity.

#### 4. Expression Format
* Write each atomic_fact as a **single, complete English sentence** in **third-person** form.
  - e.g., "Gina said that she had launched an ad campaign for her clothing store yesterday (March 9, 2024)."
* Do **not** simplify, paraphrase, or merge logically distinct ideas.

#### 5. Retrieval Clarity
* Each atomic_fact must be concise, factual, and self-contained.
* Avoid small talk or filler unless it conveys meaningful emotional or informational content.
* Ensure entities and actions are explicit and unambiguous.

#### 6. Output Requirements
* Output **only** the JSON object — no additional explanation, markdown, or commentary.
* Ensure valid JSON (proper quotes, commas, and escaping).
* The `"atomic_fact"` list should include all meaningful facts extracted from the episode.

---

### QUALITY CHECKS
Before returning the final output, verify that:
1. Every meaningful fact, intention, and emotion is included.
2. Each `atomic_fact` contains exactly one idea.
3. All relevant time references are preserved or normalized.
4. Wording remains faithful to the source.
5. The output JSON is valid and follows the exact schema.

---

### EXAMPLE

**Input:**
TIME = "March 10, 2024(Sunday) at 2:00 PM"  
EPISODE_TEXT =  
"Gina said she just launched an ad campaign for her clothing store yesterday.  
Jon congratulated her and asked how her dance studio search was going.  
Gina explained that she was focusing on the clothing store for now but still hoped to find a suitable studio soon."

**Output:**
{
  "event_log": {
    "time": "March 10, 2024(Sunday) at 2:00 PM",
    "atomic_fact": [
      "Gina said that she had launched an ad campaign for her clothing store yesterday (March 9, 2024).",
      "Jon congratulated Gina on her new campaign.",
      "Jon asked Gina about the progress of her dance studio search.",
      "Gina explained that she was focusing on the clothing store for now.",
      "Gina said that she still hoped to find a suitable studio in the future."
    ]
  }
}

---

Now analyze the provided EPISODE_TEXT and TIME carefully, apply all rules above, and return **only the JSON object** in the specified format.
---

### INPUT
- EPISODE_TEXT: "{{EPISODE_TEXT}}"
- TIME: "{{TIME}}"  (the start time of the episode, e.g., "March 10, 2024(Sunday) at 2:00 PM")

"""

