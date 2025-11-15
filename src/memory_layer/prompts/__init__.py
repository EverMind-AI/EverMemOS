"""
多语言提示词模块

通过环境变量 MEMORY_LANGUAGE 控制语言，支持 'en' 和 'zh'
默认使用英文 ('en')

使用方法：
1. 设置环境变量：export MEMORY_LANGUAGE=zh
2. 现有代码无需修改，直接从 memory_layer.prompts 导入使用

示例：
    from memory_layer.prompts import (
        EPISODE_GENERATION_PROMPT,
        CONVERSATION_PROFILE_PART1_EXTRACTION_PROMPT,
        get_semantic_generation_prompt,
    )
"""

import os

# 获取语言设置，默认为英文
MEMORY_LANGUAGE = os.getenv('MEMORY_LANGUAGE', 'en').lower()

# 支持的语言
SUPPORTED_LANGUAGES = ['en', 'zh']

if MEMORY_LANGUAGE not in SUPPORTED_LANGUAGES:
    print(f"Warning: Unsupported language '{MEMORY_LANGUAGE}', falling back to 'en'")
    MEMORY_LANGUAGE = 'en'

# 根据语言设置动态导入提示词
if MEMORY_LANGUAGE == 'zh':
    # ===== 中文提示词 =====
    # 对话相关
    from .zh.conv_prompts import CONV_BOUNDARY_DETECTION_PROMPT, CONV_SUMMARY_PROMPT
    
    # Episode 相关
    from .zh.episode_mem_prompts import (
        EPISODE_GENERATION_PROMPT,
        GROUP_EPISODE_GENERATION_PROMPT,
        DEFAULT_CUSTOM_INSTRUCTIONS,
    )
    
    # Profile 相关
    from .zh.profile_mem_prompts import CONVERSATION_PROFILE_EXTRACTION_PROMPT
    from .zh.profile_mem_part1_prompts import CONVERSATION_PROFILE_PART1_EXTRACTION_PROMPT
    from .zh.profile_mem_part2_prompts import CONVERSATION_PROFILE_PART2_EXTRACTION_PROMPT
    from .zh.profile_mem_part3_prompts import CONVERSATION_PROFILE_PART3_EXTRACTION_PROMPT
    from .zh.profile_mem_evidence_completion_prompt import (
        CONVERSATION_PROFILE_EVIDENCE_COMPLETION_PROMPT,
    )
    
    # Group Profile 相关
    from .zh.group_profile_prompts import (
        CONTENT_ANALYSIS_PROMPT,
        BEHAVIOR_ANALYSIS_PROMPT,
    )
    
    # Semantic Memory 相关
    from .zh.semantic_mem_prompts import (
        get_group_semantic_generation_prompt,
        get_semantic_generation_prompt,
    )
    
    # Event Log 相关
    from .zh.event_log_prompts import EVENT_LOG_PROMPT
    
else:
    # ===== 英文提示词（默认） =====
    # 对话相关
    from .en.conv_prompts import CONV_BOUNDARY_DETECTION_PROMPT, CONV_SUMMARY_PROMPT
    
    # Episode 相关
    from .en.episode_mem_prompts import (
        EPISODE_GENERATION_PROMPT,
        GROUP_EPISODE_GENERATION_PROMPT,
        DEFAULT_CUSTOM_INSTRUCTIONS,
    )
    
    # Profile 相关
    from .en.profile_mem_prompts import CONVERSATION_PROFILE_EXTRACTION_PROMPT
    from .en.profile_mem_part1_prompts import CONVERSATION_PROFILE_PART1_EXTRACTION_PROMPT
    from .en.profile_mem_part2_prompts import CONVERSATION_PROFILE_PART2_EXTRACTION_PROMPT
    from .en.profile_mem_part3_prompts import CONVERSATION_PROFILE_PART3_EXTRACTION_PROMPT
    from .en.profile_mem_evidence_completion_prompt import (
        CONVERSATION_PROFILE_EVIDENCE_COMPLETION_PROMPT,
    )
    
    # Group Profile 相关
    from .en.group_profile_prompts import (
        CONTENT_ANALYSIS_PROMPT,
        BEHAVIOR_ANALYSIS_PROMPT,
    )
    
    # Semantic Memory 相关
    from .en.semantic_mem_prompts import (
        get_group_semantic_generation_prompt,
        get_semantic_generation_prompt,
    )
    
    # Event Log 相关
    from .en.event_log_prompts import EVENT_LOG_PROMPT

# 导出当前语言信息
CURRENT_LANGUAGE = MEMORY_LANGUAGE


def get_current_language():
    """获取当前语言"""
    return CURRENT_LANGUAGE


def set_language(language: str):
    """设置语言（需要重启应用才能生效）"""
    global MEMORY_LANGUAGE, CURRENT_LANGUAGE
    if language.lower() in SUPPORTED_LANGUAGES:
        MEMORY_LANGUAGE = language.lower()
        CURRENT_LANGUAGE = MEMORY_LANGUAGE
        print(f"Language set to: {MEMORY_LANGUAGE}")
    else:
        print(f"Unsupported language: {language}. Supported: {SUPPORTED_LANGUAGES}")
