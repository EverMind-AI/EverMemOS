"""
LongMemEval-specific Sufficiency Check Prompt with time-awareness support.

This module provides prompts optimized for LongMemEval dataset, which includes
temporal-reasoning questions that require knowing the "current time" (question_date).
"""

SUFFICIENCY_CHECK_PROMPT_LME = """You are an expert in information retrieval evaluation. Assess whether the retrieved documents provide sufficient information to answer the user's query.

# CRITICAL: CURRENT TIME CONTEXT
**The current date and time is: {current_time}**
Use this as the reference point for ALL relative time calculations in the query (e.g., "last week", "3 days ago", "two months ago", "yesterday").

--------------------------
User Query:
{query}

Retrieved Documents:
{retrieved_docs}
--------------------------

### Instructions:

1. **Analyze the Query's Needs**
   - **Entities**: Who/What is being asked about?
   - **Attributes**: What specific details (color, time, location, quantity)?
   - **Time**: Does it ask for a specific time or use relative time expressions?
     - **IMPORTANT**: Convert relative times using the CURRENT TIME above!
     - Example: If current time is "2023/05/30" and query asks "two weeks ago", calculate → around "2023/05/16"

2. **Evaluate Document Evidence**
   - Check **Content**: Do the documents mention the entities and attributes?
   - Check **Dates**: 
     - Use the `Date` field of each document.
     - For relative time queries, verify if document dates fall within the calculated timeframe.
     - Example: Query "What did I do last week?" with current time 2023/05/30 → need docs from ~2023/05/22 to 2023/05/28
   
3. **Time Calculation Examples**
   - Current: 2023/05/30, "two weeks ago" → ~2023/05/16
   - Current: 2023/05/30, "last month" → ~April 2023
   - Current: 2023/05/30, "yesterday" → 2023/05/29
   - Current: 2023/05/30, "three months ago" → ~February 2023

4. **Judgment Logic**
   - **Sufficient**: You can answer the query *completely* and *precisely* using ONLY the provided documents.
   - **Insufficient**: 
     - The specific entity is not found.
     - The entity is found, but the specific attribute (e.g., "price", "date") is missing.
     - The time reference cannot be resolved (no documents from the calculated timeframe).
     - Conflicting information without resolution.

### Output Format (strict JSON):
{{
  "is_sufficient": true or false,
  "reasoning": "Brief explanation. Include time calculations if relevant. If insufficient, state WHY.",
  "time_calculation": "If query has relative time, show: 'Query asks about X → Current time {current_time} → Target date range: Y'",
  "key_information_found": ["Fact 1 (Source: Doc 1, Date: YYYY/MM/DD)", "Fact 2 (Source: Doc 2)"],
  "missing_information": ["Specific gap 1", "Specific gap 2"]
}}

Now evaluate:"""


def format_lme_sufficiency_prompt(query: str, retrieved_docs: str, current_time: str) -> str:
    """
    Format the LongMemEval sufficiency check prompt with time context.
    
    Args:
        query: User question
        retrieved_docs: Formatted retrieved documents
        current_time: The question_date from LongMemEval dataset (e.g., "2023/05/30 (Tue) 23:40")
    
    Returns:
        Formatted prompt string
    """
    return SUFFICIENCY_CHECK_PROMPT_LME.format(
        query=query,
        retrieved_docs=retrieved_docs,
        current_time=current_time
    )

