"""
LongMemEval-specific answer prompts with time-awareness support.

This module provides prompts optimized for LongMemEval dataset, which includes
temporal-reasoning questions that require knowing the "current time" (question_date).
"""

# Time-aware prompt for LongMemEval
# Includes {current_time} placeholder for temporal reasoning
ANSWER_PROMPT_LME = """
You are an intelligent memory assistant tasked with retrieving accurate information from episodic memories.

# CURRENT TIME CONTEXT:
**The current date and time is: {current_time}**
Use this as the reference point for ALL relative time calculations (e.g., "last week", "3 days ago", "how many weeks ago").

# CONTEXT:
You have access to episodic memories from conversations between two speakers. These memories contain
timestamped information that may be relevant to answering the question.

# INSTRUCTIONS:
Your goal is to synthesize information from all relevant memories to provide a comprehensive and accurate answer.
You MUST follow a structured Chain-of-Thought process to ensure no details are missed.
Actively look for connections between people, places, and events to build a complete picture. Synthesize information from different memories to answer the user's question.
It is CRITICAL that you move beyond simple fact extraction and perform logical inference. When the evidence strongly suggests a connection, you must state that connection. Do not dismiss reasonable inferences as "speculation." Your task is to provide the most complete answer supported by the available evidence.

# CRITICAL REQUIREMENTS:
1. NEVER omit specific names - use "Amy's colleague Rob" not "a colleague"
2. ALWAYS include exact numbers, amounts, prices, percentages, dates, times
3. PRESERVE frequencies exactly - "every Tuesday and Thursday" not "twice a week"
4. MAINTAIN all proper nouns and entities as they appear
5. EXPLICITLY state confidence levels for inferences (High/Medium/Low)
6. **FOR TEMPORAL QUESTIONS**: Use the CURRENT TIME above as your reference point. Calculate time differences precisely.

# RESPONSE FORMAT (You MUST follow this structure):

## STEP 1: RELEVANT MEMORIES EXTRACTION
[List each memory that relates to the question, with its timestamp]
- Memory [ID]: [timestamp] - [content snippet]

## STEP 2: KEY INFORMATION IDENTIFICATION
[Extract ALL specific details from the memories]
- Names mentioned: [list all person names, place names, company names]
- Numbers/Quantities: [list all amounts, prices, percentages]
- Dates/Times: [list all temporal information]
- Frequencies: [list any recurring patterns]
- Other entities: [list brands, products, etc.]

## STEP 3: CROSS-MEMORY LINKING & INFERENCE
[Identify entities that appear in multiple memories and link related information. Make reasonable inferences when entities are strongly connected.]
- Shared entities: [list people, places, events mentioned across different memories]
- Connections found: [e.g., "Memory 1 mentions A moved from hometown -> Memory 2 mentions A's hometown is LA -> Therefore A moved from LA"]
- Inferences: [Connect the dots. Label confidence: (Confidence: High/Medium/Low)]

## STEP 4: TIME REFERENCE CALCULATION
[CRITICAL for temporal reasoning questions]
- Current time (reference point): {current_time}
- Event timestamp from memories: [extracted timestamp]
- Time difference calculation: [Show step-by-step: e.g., "From Jan 15 to Feb 1 = 17 days = ~2.4 weeks"]
- Final result: [e.g., "approximately 2 weeks ago"]

## STEP 5: CONTRADICTION & GAP ANALYSIS
[Check for conflicts and missing details]
- Conflicting information: [describe conflicts and resolution strategy]
- Missing information: [explicitly state what details are requested but missing from context]

## STEP 6: DETAIL VERIFICATION CHECKLIST
- [ ] All person names included?
- [ ] All locations included?
- [ ] All numbers exact?
- [ ] All frequencies specific?
- [ ] All dates/times precise?
- [ ] All proper nouns preserved?
- [ ] Time calculations verified against current time?

## STEP 7: FINAL ANSWER
[Provide the concise answer with ALL specific details preserved. Do not include the internal checklist in this section, just the final synthesized answer.]

---

{context}

Question: {question}

Now, follow the Chain-of-Thought process above to answer the question:
"""


def get_lme_prompt_template(current_time: str) -> str:
    """
    Return LME prompt template with only {current_time} filled in.

    Leaves {context} and {question} placeholders intact for locomo_response() to fill.

    Args:
        current_time: The question_date from LongMemEval dataset (e.g., "2023/05/30 (Tue) 23:40")

    Returns:
        Partially formatted prompt template (still contains {context} and {question})
    """
    return ANSWER_PROMPT_LME.replace("{current_time}", current_time)

