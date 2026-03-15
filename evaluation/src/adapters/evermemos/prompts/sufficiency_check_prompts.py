"""Sufficiency Check Prompt for Agentic Retrieval"""

SUFFICIENCY_CHECK_PROMPT = """You are an expert in information retrieval evaluation. Assess whether the retrieved documents provide sufficient information to answer the user's query.

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
   - **Time**: Does it ask for a specific time (absolute or relative like "last week")?

2. **Evaluate Document Evidence**
   - Check **Content**: Do the documents mention the entities and attributes?
   - Check **Dates**: 
     - Use the `Date` field of each document.
     - For relative time queries (e.g., "last week", "yesterday"), verify if document dates fall within that timeframe.
     - If the query asks "When did X happen?", do you have the specific date or just a vague mention?

3. **Judgment Logic**
   - **Sufficient**: You can answer the query *completely* and *precisely* using ONLY the provided documents.
   - **Insufficient**: 
     - The specific entity is not found.
     - The entity is found, but the specific attribute (e.g., "price") is missing.
     - The time reference cannot be resolved (e.g., doc says "yesterday" but has no date, or doc date doesn't match query timeframe).
     - Conflicting information without resolution.

### Output Format (strict JSON):
{{
  "is_sufficient": true or false,
  "reasoning": "Brief explanation. If insufficient, state WHY (e.g., 'Found X but missing date', 'No mention of Y').",
  "key_information_found": ["Fact 1 (Source: Doc 1)", "Fact 2 (Source: Doc 2)"],
  "missing_information": ["Specific gap 1", "Specific gap 2"]
}}

Now evaluate:"""
