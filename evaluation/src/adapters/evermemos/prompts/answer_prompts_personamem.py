PERSONAMEM_MCQ_PROMPT = """
You are an intelligent assistant answering a multiple-choice question about a user's preferences, habits, or personal profile.
Your task is to select the BEST option that matches the user's known characteristics from their conversation memories and profile.

# CONTEXT (Conversation Memories):
{context}

# USER PROFILE:
{profile}

# QUESTION:
{question}

# OPTIONS:
{options}

# INSTRUCTIONS:
Follow this structured reasoning process to select the best answer:

## STEP 1: PROFILE & MEMORY ANALYSIS
[Extract relevant user characteristics from the profile and memories]
- Explicit preferences: [hobbies, habits, dietary preferences, routines, etc.]
- Implicit traits: [personality, communication style, values, etc.]
- Key facts: [profession, location, relationships, experiences, etc.]

## STEP 2: OPTION EVALUATION
[Evaluate each option against the user's known characteristics]
- (a): [How well does this match the user's profile/memories? Cite evidence]
- (b): [How well does this match the user's profile/memories? Cite evidence]
- (c): [How well does this match the user's profile/memories? Cite evidence]
- (d): [How well does this match the user's profile/memories? Cite evidence]
(Continue for all available options)

## STEP 3: BEST MATCH SELECTION
[Identify the option with the strongest evidence]
- Best match: [option letter]
- Reasoning: [Why this option best reflects the user's preferences/profile]
- Confidence: [High/Medium/Low]

## STEP 4: FINAL ANSWER
[Output ONLY the option letter in parentheses, e.g., (a)]

FINAL ANSWER:
"""
