"""Language utilities module

Unified management of prompt language settings. All logic that needs to get the
default language should call the functions in this module.
"""

import os

# Supported language list
SUPPORTED_LANGUAGES = ["en", "zh"]

# Default language
DEFAULT_LANGUAGE = "en"


def get_prompt_language() -> str:
    """Get the current prompt language setting

    Gets the language setting from the MEMORY_LANGUAGE environment variable.
    If not set or unsupported, returns the default value "en".
    Language setting should be configured via environment variable at startup
    and cannot be modified at runtime.

    Returns:
        The current language setting, defaults to "en"
    """
    language = os.getenv("MEMORY_LANGUAGE", DEFAULT_LANGUAGE).lower()
    if language not in SUPPORTED_LANGUAGES:
        return DEFAULT_LANGUAGE
    return language


def is_supported_language(language: str) -> bool:
    """Check if a language is supported

    Args:
        language: Language code

    Returns:
        Whether the language is supported
    """
    return language.lower() in SUPPORTED_LANGUAGES

