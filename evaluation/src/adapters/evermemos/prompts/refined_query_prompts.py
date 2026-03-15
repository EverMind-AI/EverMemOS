"""Refined Query Generation Prompt for Agentic Retrieval"""

REFINED_QUERY_PROMPT = """You are an expert at query reformulation for information retrieval.

**Task**: Generate a refined query that targets the missing information in the retrieved results.

**Original Query**:
{original_query}

**Retrieved Documents** (insufficient):
{retrieved_docs}

**Missing Information**:
{missing_info}

**Instructions**:
1. Keep the core intent of the original query unchanged.
2. Add specific keywords or rephrase to target the missing information.
3. Make the query more specific and focused.
4. The refined query should be a direct question that seeks to extract the missing facts.
5. Do NOT change the query's meaning or make it too broad.
6. Keep it concise (1-2 sentences maximum).

**Examples**:

Example 1:
Original Query: "What happened at the conference?"
Missing Info: ["specific sessions attended", "key takeaways"]
Refined Query: "Which sessions were attended at the conference and what were the main takeaways?"

Example 2:
Original Query: "Tell me about the deadline"
Missing Info: ["deadline date", "deliverables", "responsible team"]
Refined Query: "When is the deadline, what needs to be delivered, and who is responsible?"

Example 3:
Original Query: "The new policy"
Missing Info: ["policy details", "effective date", "affected departments"]
Refined Query: "What are the details of the new policy, when does it take effect, and which departments are affected?"

Now generate the refined query (output only the refined query, no additional text):
Original Query: {original_query}
Missing Info: {missing_info}

Refined Query:
"""

