"""
Prompt templates for EverMemOS evaluation.

This module provides prompts for:
- Answer generation (answer_prompts.py, answer_prompts_lme.py)
- Sufficiency check (sufficiency_check_prompts.py, sufficiency_check_prompts_lme.py)
- Multi-query generation (multi_query_prompts.py, multi_query_prompts_lme.py)
- Query refinement (refined_query_prompts.py)

The *_lme.py variants include time-awareness for LongMemEval dataset.
"""

from .answer_prompts import ANSWER_PROMPT
from .answer_prompts_lme import ANSWER_PROMPT_LME, format_lme_prompt
from .sufficiency_check_prompts import SUFFICIENCY_CHECK_PROMPT
from .sufficiency_check_prompts_lme import SUFFICIENCY_CHECK_PROMPT_LME, format_lme_sufficiency_prompt
from .multi_query_prompts import MULTI_QUERY_GENERATION_PROMPT
from .multi_query_prompts_lme import MULTI_QUERY_GENERATION_PROMPT_LME, format_lme_multi_query_prompt
from .refined_query_prompts import REFINED_QUERY_PROMPT

__all__ = [
    # Answer prompts
    "ANSWER_PROMPT",
    "ANSWER_PROMPT_LME",
    "format_lme_prompt",
    
    # Sufficiency check prompts
    "SUFFICIENCY_CHECK_PROMPT",
    "SUFFICIENCY_CHECK_PROMPT_LME",
    "format_lme_sufficiency_prompt",
    
    # Multi-query prompts
    "MULTI_QUERY_GENERATION_PROMPT",
    "MULTI_QUERY_GENERATION_PROMPT_LME",
    "format_lme_multi_query_prompt",
    
    # Query refinement prompts
    "REFINED_QUERY_PROMPT",
]
