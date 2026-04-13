"""
Unit tests for AgentSkill relevance post-verification in SearchMemoryService.

Tests the _verify_skill_relevance method with mocked LLM responses.
"""

import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from api_specs.dtos.memory import SearchAgentSkillItem as AgentSkillItem


def _make_service():
    """Create a SearchMemoryService instance with all repos mocked out."""
    with patch("agentic_layer.search_mem_service.EpisodicMemoryEsRepository"), \
         patch("agentic_layer.search_mem_service.EpisodicMemoryMilvusRepository"), \
         patch("agentic_layer.search_mem_service.UserProfileMilvusRepository"), \
         patch("agentic_layer.search_mem_service.AgentCaseEsRepository"), \
         patch("agentic_layer.search_mem_service.AgentSkillEsRepository"), \
         patch("agentic_layer.search_mem_service.AgentCaseMilvusRepository"), \
         patch("agentic_layer.search_mem_service.AgentSkillMilvusRepository"), \
         patch("agentic_layer.search_mem_service.MemoryManager"), \
         patch("agentic_layer.search_mem_service.RawMessageService"):
        from agentic_layer.search_mem_service import SearchMemoryService
        return SearchMemoryService()


def _make_skill(name: str, description: str = "desc", content: str = "content", score: float = 0.8) -> AgentSkillItem:
    """Helper to create an AgentSkillItem instance."""
    return AgentSkillItem(
        id=f"skill_{name}",
        user_id="test_user",
        name=name,
        description=description,
        content=content,
        score=score,
    )


@pytest.fixture
def service():
    return _make_service()


@pytest.mark.asyncio
async def test_empty_skills_returns_empty(service):
    """Empty input returns empty output without calling LLM."""
    result = await service._verify_skill_relevance(
        query="how to fix database connection",
        skills=[],
    )
    assert result == []


@pytest.mark.asyncio
async def test_empty_query_returns_all(service):
    """Empty query returns all skills without filtering."""
    skills = [_make_skill("skill1")]
    result = await service._verify_skill_relevance(query="", skills=skills)
    assert result == skills


@pytest.mark.asyncio
async def test_filters_irrelevant_skills(service):
    """LLM gives high score to relevant skill and low score to irrelevant — only high-scoring one is returned."""
    skills = [
        _make_skill(
            "Database connection pool tuning",
            "Optimize DB connection pool settings",
            "## Steps\n1. Check pool size\n2. Adjust max connections",
        ),
        _make_skill(
            "CSS grid layout",
            "Build responsive layouts with CSS grid",
            "## Steps\n1. Define grid container\n2. Set grid-template-columns",
        ),
    ]

    llm_response = json.dumps({
        "results": [
            {"index": 0, "score": 0.9, "reason": "directly addresses DB connection issues"},
            {"index": 1, "score": 0.1, "reason": "CSS layout is unrelated to DB connections"},
        ]
    })

    mock_provider = AsyncMock()
    mock_provider.generate = AsyncMock(return_value=llm_response)

    with patch("memory_layer.llm.llm_provider.build_default_provider", return_value=mock_provider), \
         patch("memory_layer.prompts.get_prompt_by", return_value="{query}{skills_json}"):
        result = await service._verify_skill_relevance(
            query="how to fix database connection pool timeout",
            skills=skills,
        )

    assert len(result) == 1
    assert result[0].name == "Database connection pool tuning"
    assert result[0].score == 0.9


@pytest.mark.asyncio
async def test_all_skills_high_score(service):
    """When LLM gives all skills high scores, all are returned sorted by score descending."""
    skills = [_make_skill("skill_a"), _make_skill("skill_b")]

    llm_response = json.dumps({
        "results": [
            {"index": 0, "score": 0.7, "reason": "relevant"},
            {"index": 1, "score": 0.9, "reason": "very relevant"},
        ]
    })

    mock_provider = AsyncMock()
    mock_provider.generate = AsyncMock(return_value=llm_response)

    with patch("memory_layer.llm.llm_provider.build_default_provider", return_value=mock_provider), \
         patch("memory_layer.prompts.get_prompt_by", return_value="{query}{skills_json}"):
        result = await service._verify_skill_relevance(
            query="some query", skills=skills,
        )

    assert len(result) == 2
    assert result[0].name == "skill_b"
    assert result[1].name == "skill_a"


@pytest.mark.asyncio
async def test_all_skills_low_score(service):
    """When LLM gives all skills low scores, empty list is returned."""
    skills = [_make_skill("skill_a")]

    llm_response = json.dumps({
        "results": [
            {"index": 0, "score": 0.2, "reason": "not relevant"},
        ]
    })

    mock_provider = AsyncMock()
    mock_provider.generate = AsyncMock(return_value=llm_response)

    with patch("memory_layer.llm.llm_provider.build_default_provider", return_value=mock_provider), \
         patch("memory_layer.prompts.get_prompt_by", return_value="{query}{skills_json}"):
        result = await service._verify_skill_relevance(
            query="some query", skills=skills,
        )

    assert len(result) == 0


@pytest.mark.asyncio
async def test_llm_failure_returns_all(service):
    """When LLM call fails, all original skills are returned as fallback."""
    skills = [_make_skill("skill_a"), _make_skill("skill_b")]

    mock_provider = AsyncMock()
    mock_provider.generate = AsyncMock(side_effect=Exception("LLM API error"))

    with patch("memory_layer.llm.llm_provider.build_default_provider", return_value=mock_provider), \
         patch("memory_layer.prompts.get_prompt_by", return_value="{query}{skills_json}"):
        result = await service._verify_skill_relevance(
            query="some query", skills=skills,
        )

    assert len(result) == 2


@pytest.mark.asyncio
async def test_malformed_llm_json_returns_all(service):
    """When LLM returns invalid JSON, all original skills are returned as fallback."""
    skills = [_make_skill("skill_a")]

    mock_provider = AsyncMock()
    mock_provider.generate = AsyncMock(return_value="not valid json {{{")

    with patch("memory_layer.llm.llm_provider.build_default_provider", return_value=mock_provider), \
         patch("memory_layer.prompts.get_prompt_by", return_value="{query}{skills_json}"):
        result = await service._verify_skill_relevance(
            query="some query", skills=skills,
        )

    assert len(result) == 1


# ---------------------------------------------------------------------------
# Edge cases: None / missing fields / boundary conditions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_none_skills_returns_none(service):
    """None input is returned as-is (falsy short-circuit)."""
    result = await service._verify_skill_relevance(query="some query", skills=None)
    assert result is None


@pytest.mark.asyncio
async def test_none_fields_in_skill_use_empty_string(service):
    """Skills with None name/description/content are serialised as empty strings in prompt."""
    skills = [AgentSkillItem(id="s1", user_id="u1", name=None, description=None, content=None, score=0.5)]

    llm_response = json.dumps({"results": [{"index": 0, "score": 0.8, "reason": "ok"}]})
    mock_provider = AsyncMock()

    captured_prompt = None

    async def _capture_generate(prompt, **kwargs):
        nonlocal captured_prompt
        captured_prompt = prompt
        return llm_response

    mock_provider.generate = _capture_generate

    with patch("memory_layer.llm.llm_provider.build_default_provider", return_value=mock_provider), \
         patch("memory_layer.prompts.get_prompt_by", return_value="{query}{skills_json}"):
        result = await service._verify_skill_relevance(query="q", skills=skills)

    assert len(result) == 1
    parsed = json.loads(captured_prompt[len("q"):])
    assert parsed[0]["name"] == ""
    assert parsed[0]["description"] == ""
    assert parsed[0]["content"] == ""


@pytest.mark.asyncio
async def test_out_of_range_index_ignored(service):
    """LLM returns an index beyond the skills list — it is safely ignored."""
    skills = [_make_skill("only_one")]

    llm_response = json.dumps({
        "results": [
            {"index": 0, "score": 0.85, "reason": "ok"},
            {"index": 99, "score": 0.9, "reason": "ghost"},
        ]
    })

    mock_provider = AsyncMock()
    mock_provider.generate = AsyncMock(return_value=llm_response)

    with patch("memory_layer.llm.llm_provider.build_default_provider", return_value=mock_provider), \
         patch("memory_layer.prompts.get_prompt_by", return_value="{query}{skills_json}"):
        result = await service._verify_skill_relevance(query="q", skills=skills)

    assert len(result) == 1
    assert result[0].name == "only_one"


@pytest.mark.asyncio
async def test_missing_results_key_returns_empty(service):
    """LLM returns valid JSON but without 'results' key — no skills pass."""
    skills = [_make_skill("skill_a")]

    llm_response = json.dumps({"answer": "something unexpected"})
    mock_provider = AsyncMock()
    mock_provider.generate = AsyncMock(return_value=llm_response)

    with patch("memory_layer.llm.llm_provider.build_default_provider", return_value=mock_provider), \
         patch("memory_layer.prompts.get_prompt_by", return_value="{query}{skills_json}"):
        result = await service._verify_skill_relevance(query="q", skills=skills)

    assert len(result) == 0


@pytest.mark.asyncio
async def test_missing_score_field_defaults_zero(service):
    """LLM result item without 'score' key defaults to 0.0 — skill is excluded."""
    skills = [_make_skill("skill_a")]

    llm_response = json.dumps({
        "results": [{"index": 0, "reason": "no score field"}]
    })
    mock_provider = AsyncMock()
    mock_provider.generate = AsyncMock(return_value=llm_response)

    with patch("memory_layer.llm.llm_provider.build_default_provider", return_value=mock_provider), \
         patch("memory_layer.prompts.get_prompt_by", return_value="{query}{skills_json}"):
        result = await service._verify_skill_relevance(query="q", skills=skills)

    assert len(result) == 0


@pytest.mark.asyncio
async def test_partial_indices_only_returns_scored_above_threshold(service):
    """LLM only returns judgement for some skills — unjudged ones default to 0.0 and are excluded."""
    skills = [_make_skill("a"), _make_skill("b"), _make_skill("c")]

    llm_response = json.dumps({
        "results": [
            {"index": 1, "score": 0.75, "reason": "relevant"},
        ]
    })
    mock_provider = AsyncMock()
    mock_provider.generate = AsyncMock(return_value=llm_response)

    with patch("memory_layer.llm.llm_provider.build_default_provider", return_value=mock_provider), \
         patch("memory_layer.prompts.get_prompt_by", return_value="{query}{skills_json}"):
        result = await service._verify_skill_relevance(query="q", skills=skills)

    assert len(result) == 1
    assert result[0].name == "b"


@pytest.mark.asyncio
async def test_results_sorted_by_score_descending(service):
    """Results are sorted by LLM relevance score in descending order."""
    skills = [_make_skill("first"), _make_skill("second"), _make_skill("third")]

    llm_response = json.dumps({
        "results": [
            {"index": 0, "score": 0.6, "reason": "ok"},
            {"index": 1, "score": 0.9, "reason": "great"},
            {"index": 2, "score": 0.75, "reason": "good"},
        ]
    })
    mock_provider = AsyncMock()
    mock_provider.generate = AsyncMock(return_value=llm_response)

    with patch("memory_layer.llm.llm_provider.build_default_provider", return_value=mock_provider), \
         patch("memory_layer.prompts.get_prompt_by", return_value="{query}{skills_json}"):
        result = await service._verify_skill_relevance(query="q", skills=skills)

    assert len(result) == 3
    assert result[0].name == "second"
    assert result[1].name == "third"
    assert result[2].name == "first"


@pytest.mark.asyncio
async def test_threshold_boundary(service):
    """Score exactly at 0.4 passes, score below 0.4 is excluded."""
    skills = [_make_skill("at_boundary"), _make_skill("below_boundary")]

    llm_response = json.dumps({
        "results": [
            {"index": 0, "score": 0.4, "reason": "borderline"},
            {"index": 1, "score": 0.39, "reason": "just below"},
        ]
    })
    mock_provider = AsyncMock()
    mock_provider.generate = AsyncMock(return_value=llm_response)

    with patch("memory_layer.llm.llm_provider.build_default_provider", return_value=mock_provider), \
         patch("memory_layer.prompts.get_prompt_by", return_value="{query}{skills_json}"):
        result = await service._verify_skill_relevance(query="q", skills=skills)

    assert len(result) == 1
    assert result[0].name == "at_boundary"
    assert result[0].score == 0.4


@pytest.mark.asyncio
async def test_prompt_receives_correct_arguments(service):
    """Verify get_prompt_by is called with correct key and format receives query + skills_json."""
    skills = [_make_skill("db_tuning", "tune db", "step1")]

    llm_response = json.dumps({"results": [{"index": 0, "score": 0.85, "reason": "ok"}]})
    mock_provider = AsyncMock()
    mock_provider.generate = AsyncMock(return_value=llm_response)

    with patch("memory_layer.llm.llm_provider.build_default_provider", return_value=mock_provider), \
         patch("memory_layer.prompts.get_prompt_by", return_value="Q={query} S={skills_json}") as mock_prompt:
        result = await service._verify_skill_relevance(query="fix db", skills=skills)

    mock_prompt.assert_called_once_with("AGENT_SKILL_RELEVANCE_VERIFY_PROMPT")
    call_args = mock_provider.generate.call_args
    prompt_text = call_args[0][0]
    assert prompt_text.startswith("Q=fix db S=")
    assert '"name": "db_tuning"' in prompt_text
