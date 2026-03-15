"""
LongMemEval-specific Multi-Query Generation Prompt with time-awareness support.

This module provides prompts optimized for LongMemEval dataset, which includes
temporal-reasoning questions that require knowing the "current time" (question_date).
"""

MULTI_QUERY_GENERATION_PROMPT_LME = """You are an expert at query reformulation for conversational memory retrieval.
Your goal is to generate 2-3 complementary queries to find the MISSING information.

# CRITICAL: CURRENT TIME CONTEXT
**The current date and time is: {current_time}**
Use this as the reference point for ALL relative time expressions.

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

**[B] Temporal Calculation (If time is missing/unclear) - USE CURRENT TIME!**
- **CRITICAL**: Convert relative times to ABSOLUTE dates using the current time above!
- Current time: {current_time}
- Examples:
  - "two weeks ago" → Calculate: {current_time} - 14 days → Search for that specific date
  - "last month" → Calculate: previous month from {current_time}
  - "yesterday" → Calculate: {current_time} - 1 day
- Generate queries with SPECIFIC dates or date ranges when possible.
- Example: Original "What did I do last week?" → Generate "events May 22 to May 28" (if current is May 30)

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
3. **Date-Specific Query** (For temporal questions): Include calculated dates.
   - e.g., "events on May 16 2023", "what happened in April 2023"
4. **Hypothetical Statement** (HyDE, 5-10 words): A likely sentence in the memory.
   - e.g., "We decided to postpone the launch", "The client requested changes"

### Requirements:
- Generate 2-3 queries.
- **CRITICAL FOR TEMPORAL QUERIES**: At least ONE query should include a calculated absolute date/date range.
- Use the strategies above to target the *Missing Information*.
- Keep queries SHORT and SEARCHABLE.

### Output Format (STRICT JSON):
{{
  "queries": [
    "Query 1",
    "Query 2",
    "Query 3"
  ],
  "time_calculation": "If temporal: 'Current time {current_time} → Query asks about X → Calculated target: Y'",
  "reasoning": "Strategy used for each query (e.g., Q1: Temporal with date, Q2: Pivot)"
}}

Now generate:
"""


def format_lme_multi_query_prompt(
    original_query: str,
    retrieved_docs: str,
    missing_info: str,
    key_info: str,
    current_time: str
) -> str:
    """
    Format the LongMemEval multi-query generation prompt with time context.
    
    Args:
        original_query: Original user question
        retrieved_docs: Formatted retrieved documents
        missing_info: Comma-separated missing information
        key_info: Comma-separated key information found
        current_time: The question_date from LongMemEval dataset (e.g., "2023/05/30 (Tue) 23:40")
    
    Returns:
        Formatted prompt string
    """
    return MULTI_QUERY_GENERATION_PROMPT_LME.format(
        original_query=original_query,
        retrieved_docs=retrieved_docs,
        missing_info=missing_info,
        key_info=key_info,
        current_time=current_time
    )

