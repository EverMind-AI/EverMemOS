"""Multi-Query Generation Prompt for Agentic Retrieval"""

MULTI_QUERY_GENERATION_PROMPT = """You are an expert at query reformulation for conversational memory retrieval.
Your goal is to generate 2-3 complementary queries to find the MISSING information.

--------------------------
Original Query:
{original_query}

Key Information Found:
{key_info}

Missing Information:
{missing_info}

Retrieved Documents (Context):
{retrieved_docs}
--------------------------

### Strategy Selection (Choose based on WHY info is missing):

**[A] Pivot / Entity Association (If entity is missing)**
- If the specific entity is not found, search for related entities or broader categories.
- Example: "manager's feedback" not found → try "performance review", "work evaluation".

**[B] Temporal Calculation (If time is missing/unclear)**
- Use `Date` from Retrieved Documents to anchor relative times.
- Example: doc dated 2024-03-15 mentions "last month" → search for "February 2024".
- Search for the *event* to find its timestamp: "When did the deadline change?"

**[C] Concept Expansion (If vocabulary mismatch)**
- Synonyms: "residence" → "living", "staying at", "moved to".
- General/Specific: "Italian cuisine" ↔ "pasta", "pizza", "restaurant".

**[D] Constraint Relaxation (If too specific)**
- If "quarterly sales report from Q3" fails, try "sales report", "Q3 results".
- Remove one constraint at a time.

### Query Style Requirements (Use DIFFERENT styles):

1. **Keyword Combo** (2-5 words): Key entities only. High recall.
   - e.g., "project deadline", "vacation plans summer"
2. **Natural Question** (5-10 words): Rephrased question.
   - e.g., "When was the meeting scheduled?", "What was discussed about the budget?"
3. **Hypothetical Statement** (HyDE, 5-10 words): A likely sentence in the memory.
   - e.g., "We decided to postpone the launch", "The client requested changes"

### Requirements:
- Generate 2-3 queries.
- **CRITICAL**: Use the strategies above to target the *Missing Information*.
- Keep queries SHORT and SEARCHABLE.

### Output Format (STRICT JSON):
{{
  "queries": [
    "Query 1",
    "Query 2",
    "Query 3"
  ],
  "reasoning": "Strategy used for each query (e.g., Q1: Pivot, Q2: Temporal)"
}}

Now generate:
"""
