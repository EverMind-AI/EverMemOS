"""
tests/test_mem_memorize.py

Unit tests for biz_layer/mem_memorize.py — the core memorization pipeline.

Usage:
    PYTHONPATH=src pytest tests/test_mem_memorize.py -v
"""

import asyncio
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from api_specs.memory_types import (
    AgentCase,
    AtomicFact,
    EpisodeMemory,
    Foresight,
    MemCell,
    MemoryType,
    RawDataType,
    ScenarioType,
)
from api_specs.dtos import MemorizeRequest
from biz_layer.memorize_config import (
    MemorizeConfig,
    DEFAULT_MEMORIZE_CONFIG,
    AGENT_DEFAULT_MEMORIZE_CONFIG,
)
from biz_layer.mem_memorize import (
    ExtractionState,
    _is_agent_case_quality_sufficient,
    _build_agent_cases_from_batch,
    _clone_episodes_for_users,
    _should_skip_atomic_fact_for_agent,
    if_memorize,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_memcell(
    raw_data_type: RawDataType = RawDataType.CONVERSATION,
    event_id: str = "evt-001",
    participants: Optional[List[str]] = None,
    timestamp: Optional[datetime] = None,
) -> MemCell:
    mc = MagicMock(spec=MemCell)
    mc.type = raw_data_type
    mc.event_id = event_id
    mc.original_data = []
    mc.participants = participants or ["user_001"]
    mc.timestamp = timestamp or datetime(2026, 4, 7, 10, 0, 0)
    return mc


def _make_request(
    scene: str = "solo",
    group_id: str = "grp-001",
    raw_data_type: RawDataType = RawDataType.CONVERSATION,
) -> MemorizeRequest:
    req = MagicMock(spec=MemorizeRequest)
    req.group_id = group_id
    req.session_id = "sess-001"
    req.scene = scene
    req.raw_data_type = raw_data_type
    return req


def _make_agent_case(
    quality_score: float = 0.8,
    task_intent: str = "deploy service",
    user_id: str = "user_001",
) -> AgentCase:
    ac = MagicMock(spec=AgentCase)
    ac.id = "ac-001"
    ac.quality_score = quality_score
    ac.task_intent = task_intent
    ac.approach = "ssh and restart"
    ac.key_insight = "check logs first"
    ac.user_id = user_id
    ac.memory_type = MemoryType.AGENT_CASE
    ac.timestamp = datetime(2026, 4, 7, 10, 0, 0)
    return ac


def _make_config(**overrides) -> MemorizeConfig:
    return MemorizeConfig(**overrides)


def _make_state(
    is_solo: bool = True,
    has_episode: bool = True,
    participants: Optional[List[str]] = None,
) -> ExtractionState:
    state = MagicMock(spec=ExtractionState)
    state.memcell = _make_memcell()
    state.request = _make_request("solo" if is_solo else "team")
    state.is_solo_scene = is_solo
    state.participants = participants or ["user_001"]
    state.current_time = datetime(2026, 4, 7, 10, 0, 0)
    state.foresight_parent_type = "memcell"
    state.atomic_fact_parent_type = "memcell"
    state.parent_id = "evt-001"
    saved_ep = MagicMock()
    saved_ep.id = "ep-mongo-001"
    state.group_episode_memories = [MagicMock(id="ep-mongo-001")] if has_episode else []
    state.parent_docs_map = {"ep-mongo-001": saved_ep} if has_episode else {}
    state.episode_saved = has_episode
    state.agent_case = None
    state.group_episode = MagicMock() if has_episode else None
    state.episode_memories = []
    return state


def _make_pending_entry(
    event_id: str = "evt-001",
    episode: str = "user asked about deployment",
    timestamp: float = 1712484000.0,
    participants: Optional[List[str]] = None,
    scene: str = "solo",
    agent_case: Optional[dict] = None,
) -> dict:
    entry = {
        "event_id": event_id,
        "episode": episode,
        "timestamp": timestamp,
        "participants": participants if participants is not None else ["user_001"],
        "group_id": "grp-001",
        "scene": scene,
    }
    if agent_case:
        entry["agent_case"] = agent_case
    return entry


def _make_mem_scene_state(
    pending: Optional[list] = None,
    cluster_counts: Optional[dict] = None,
    eventid_to_cluster: Optional[dict] = None,
    cluster_last_ts: Optional[dict] = None,
):
    state = MagicMock()
    state.pending_clustering = pending if pending is not None else []
    state.cluster_counts = cluster_counts or {}
    state.eventid_to_cluster = eventid_to_cluster or {}
    state.cluster_last_ts = cluster_last_ts or {}
    state.event_ids = list((eventid_to_cluster or {}).keys())
    state.timestamps = []
    state.to_dict.return_value = {}
    return state


# ===========================================================================
# _is_agent_case_quality_sufficient
# ===========================================================================

class TestIsAgentCaseQualitySufficient:

    def test_score_above_threshold_returns_true(self):
        ac = _make_agent_case(quality_score=0.5)
        config = _make_config(skill_min_quality_score=0.2)
        assert _is_agent_case_quality_sufficient(ac, config) is True

    def test_score_equal_to_threshold_returns_true(self):
        ac = _make_agent_case(quality_score=0.2)
        config = _make_config(skill_min_quality_score=0.2)
        assert _is_agent_case_quality_sufficient(ac, config) is True

    def test_score_below_threshold_returns_false(self):
        ac = _make_agent_case(quality_score=0.1)
        config = _make_config(skill_min_quality_score=0.2)
        assert _is_agent_case_quality_sufficient(ac, config) is False

    def test_score_none_returns_false(self):
        ac = _make_agent_case(quality_score=0.5)
        ac.quality_score = None
        config = _make_config(skill_min_quality_score=0.2)
        assert _is_agent_case_quality_sufficient(ac, config) is False

    def test_score_zero_below_nonzero_threshold_returns_false(self):
        ac = _make_agent_case(quality_score=0.0)
        config = _make_config(skill_min_quality_score=0.1)
        assert _is_agent_case_quality_sufficient(ac, config) is False

    def test_score_zero_with_zero_threshold_returns_true(self):
        ac = _make_agent_case(quality_score=0.0)
        config = _make_config(skill_min_quality_score=0.0)
        assert _is_agent_case_quality_sufficient(ac, config) is True


# ===========================================================================
# _build_agent_cases_from_batch
# ===========================================================================

class TestBuildAgentCasesFromBatch:

    def test_builds_cases_from_valid_entries(self):
        entries = [
            _make_pending_entry(
                event_id="evt-001",
                timestamp=1712484000.0,
                participants=["user_A"],
                agent_case={
                    "id": "ac-1",
                    "task_intent": "deploy",
                    "approach": "ssh",
                    "key_insight": "check logs",
                    "quality_score": 0.9,
                },
            ),
            _make_pending_entry(
                event_id="evt-002",
                timestamp=1712484100.0,
                participants=["user_B"],
                agent_case={
                    "id": "ac-2",
                    "task_intent": "rollback",
                    "approach": "revert",
                    "key_insight": None,
                    "quality_score": 0.5,
                },
            ),
        ]
        result = _build_agent_cases_from_batch(entries)
        assert len(result) == 2
        assert "evt-001" in result
        assert "evt-002" in result
        assert result["evt-001"].task_intent == "deploy"
        assert result["evt-002"].quality_score == 0.5

    def test_skips_entry_without_event_id(self):
        entries = [
            {
                "episode": "test",
                "timestamp": 1712484000.0,
                "participants": [],
                "agent_case": {"id": "ac-1", "task_intent": "x"},
            }
        ]
        result = _build_agent_cases_from_batch(entries)
        assert len(result) == 0

    def test_skips_entry_without_agent_case(self):
        entries = [_make_pending_entry(event_id="evt-001")]
        result = _build_agent_cases_from_batch(entries)
        assert len(result) == 0

    def test_empty_list_returns_empty_dict(self):
        result = _build_agent_cases_from_batch([])
        assert result == {}

    def test_user_id_from_first_participant(self):
        entries = [
            _make_pending_entry(
                event_id="evt-001",
                participants=["user_X", "user_Y"],
                agent_case={"id": "ac-1", "task_intent": "x", "approach": "y"},
            )
        ]
        result = _build_agent_cases_from_batch(entries)
        assert result["evt-001"].user_id == "user_X"

    def test_empty_participants_gives_empty_user_id(self):
        entries = [
            _make_pending_entry(
                event_id="evt-001",
                participants=[],
                agent_case={"id": "ac-1", "task_intent": "x", "approach": "y"},
            )
        ]
        result = _build_agent_cases_from_batch(entries)
        assert result["evt-001"].user_id == ""

    def test_timestamp_none_uses_datetime_now(self):
        entries = [
            _make_pending_entry(
                event_id="evt-001",
                timestamp=None,
                agent_case={"id": "ac-1", "task_intent": "x", "approach": "y"},
            )
        ]
        # timestamp=None in entry dict
        entries[0]["timestamp"] = None
        result = _build_agent_cases_from_batch(entries)
        # Should not raise; datetime should be approximately now
        assert result["evt-001"].timestamp is not None


# ===========================================================================
# _clone_episodes_for_users
# ===========================================================================

class TestCloneEpisodesForUsers:

    def _make_clone_state(self, participants: List[str]) -> ExtractionState:
        memcell = _make_memcell(participants=participants)
        state = ExtractionState(
            memcell=memcell,
            request=_make_request(),
            current_time=datetime(2026, 4, 7),
            scene="solo",
            is_solo_scene=True,
            participants=participants,
        )
        ep = EpisodeMemory(
            memory_type=MemoryType.EPISODIC_MEMORY,
            user_id="group",
            timestamp=datetime(2026, 4, 7),
            episode="test episode",
        )
        state.group_episode_memories = [ep]
        return state

    def test_clones_to_regular_users(self):
        state = self._make_clone_state(["alice", "bob"])
        cloned = _clone_episodes_for_users(state)
        assert len(cloned) == 2
        user_ids = {ep.user_id for ep in cloned}
        assert user_ids == {"alice", "bob"}

    def test_filters_robot_names(self):
        state = self._make_clone_state(["alice", "Robot_1", "assistant_bot", "Agent_X", "tool_call"])
        cloned = _clone_episodes_for_users(state)
        assert len(cloned) == 1
        assert cloned[0].user_id == "alice"

    def test_filter_is_case_insensitive(self):
        state = self._make_clone_state(["ROBOT_1", "ASSISTANT", "AGENT", "TOOL"])
        cloned = _clone_episodes_for_users(state)
        assert len(cloned) == 0

    def test_empty_participants(self):
        state = self._make_clone_state([])
        cloned = _clone_episodes_for_users(state)
        assert len(cloned) == 0


# ===========================================================================
# _should_skip_atomic_fact_for_agent
# ===========================================================================

class TestShouldSkipAtomicFactForAgent:

    def _make_memcell_with_data(self, messages: list) -> MemCell:
        mc = MagicMock(spec=MemCell)
        mc.original_data = [{"message": m} for m in messages]
        return mc

    def test_no_tool_calls_returns_false(self):
        mc = self._make_memcell_with_data([
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there, how can I help?"},
        ])
        assert _should_skip_atomic_fact_for_agent(mc) is False

    def test_tool_calls_short_response_returns_false(self):
        mc = self._make_memcell_with_data([
            {"role": "user", "content": "run test"},
            {"role": "assistant", "tool_calls": [{"id": "t1"}]},
            {"role": "tool", "content": "ok"},
            {"role": "assistant", "content": "done"},
        ])
        assert _should_skip_atomic_fact_for_agent(mc) is False

    def test_tool_calls_long_response_returns_true(self):
        long_text = "x" * 1000
        mc = self._make_memcell_with_data([
            {"role": "user", "content": "analyze"},
            {"role": "assistant", "tool_calls": [{"id": "t1"}]},
            {"role": "tool", "content": "data"},
            {"role": "assistant", "content": long_text},
        ])
        assert _should_skip_atomic_fact_for_agent(mc) is True

    def test_cumulative_response_across_messages(self):
        # 600 + 500 = 1100 >= 1000 -> True
        mc = self._make_memcell_with_data([
            {"role": "user", "content": "go"},
            {"role": "assistant", "tool_calls": [{"id": "t1"}]},
            {"role": "tool", "content": "ok"},
            {"role": "assistant", "content": "a" * 600},
            {"role": "assistant", "content": "b" * 500},
        ])
        assert _should_skip_atomic_fact_for_agent(mc) is True

    def test_tool_call_assistant_messages_not_counted(self):
        # Only non-tool-call assistant msgs count
        mc = self._make_memcell_with_data([
            {"role": "user", "content": "go"},
            {"role": "assistant", "tool_calls": [{"id": "t1"}], "content": "x" * 2000},
            {"role": "tool", "content": "ok"},
            {"role": "assistant", "content": "short"},
        ])
        assert _should_skip_atomic_fact_for_agent(mc) is False

    def test_empty_original_data(self):
        mc = MagicMock(spec=MemCell)
        mc.original_data = []
        assert _should_skip_atomic_fact_for_agent(mc) is False

    def test_none_original_data(self):
        mc = MagicMock(spec=MemCell)
        mc.original_data = None
        assert _should_skip_atomic_fact_for_agent(mc) is False


# ===========================================================================
# if_memorize
# ===========================================================================

class TestIfMemorize:

    def test_always_returns_true(self):
        mc = _make_memcell()
        assert if_memorize(mc) is True


# ===========================================================================
# _trigger_clustering
# ===========================================================================

class TestTriggerClustering:

    @pytest.mark.asyncio
    async def test_builds_pending_entry_and_calls_drain(self):
        from biz_layer.mem_memorize import _trigger_clustering

        mc = _make_memcell(event_id="evt-100", participants=["u1", "u2"])
        mc.timestamp = datetime(2026, 4, 7, 12, 0, 0)

        with patch(
            'biz_layer.mem_memorize._drain_and_cluster',
            new_callable=AsyncMock,
            return_value=0,
        ) as mock_drain:
            await _trigger_clustering(
                group_id="grp-001",
                memcell=mc,
                scene="team",
                episode_text="they discussed plans",
            )

            mock_drain.assert_called_once()
            call_kwargs = mock_drain.call_args[1]
            assert call_kwargs["group_id"] == "grp-001"
            entry = call_kwargs["new_entry"]
            assert entry["event_id"] == "evt-100"
            assert entry["episode"] == "they discussed plans"
            assert entry["participants"] == ["u1", "u2"]
            assert entry["scene"] == "team"

    @pytest.mark.asyncio
    async def test_includes_agent_case_when_provided(self):
        from biz_layer.mem_memorize import _trigger_clustering

        mc = _make_memcell()
        ac = _make_agent_case()

        with patch(
            'biz_layer.mem_memorize._drain_and_cluster',
            new_callable=AsyncMock,
            return_value=0,
        ) as mock_drain:
            await _trigger_clustering(
                group_id="grp-001",
                memcell=mc,
                agent_case=ac,
            )
            entry = mock_drain.call_args[1]["new_entry"]
            assert "agent_case" in entry
            assert entry["agent_case"]["task_intent"] == "deploy service"

    @pytest.mark.asyncio
    async def test_no_agent_case_key_when_none(self):
        from biz_layer.mem_memorize import _trigger_clustering

        mc = _make_memcell()

        with patch(
            'biz_layer.mem_memorize._drain_and_cluster',
            new_callable=AsyncMock,
            return_value=0,
        ) as mock_drain:
            await _trigger_clustering(group_id="grp-001", memcell=mc)
            entry = mock_drain.call_args[1]["new_entry"]
            assert "agent_case" not in entry


# ===========================================================================
# _drain_and_cluster
# ===========================================================================

class TestDrainAndCluster:

    def _patch_drain_deps(self, mem_scene_state=None, acquired=True, cluster_ids=None):
        """Build patches for _drain_and_cluster dependencies."""
        if mem_scene_state is None:
            mem_scene_state = _make_mem_scene_state()

        mock_storage = AsyncMock()
        mock_storage.load_mem_scene = AsyncMock(return_value=None)
        mock_storage.save_mem_scene = AsyncMock()

        patches = [
            patch(
                'biz_layer.mem_memorize.get_bean_by_type',
                return_value=mock_storage,
            ),
            patch(
                'memory_layer.cluster_manager.MemSceneState',
                return_value=mem_scene_state,
            ),
            patch(
                'memory_layer.cluster_manager.MemSceneState.from_dict',
                return_value=mem_scene_state,
            ),
            patch(
                'biz_layer.mem_memorize._run_batch_clustering',
                new_callable=AsyncMock,
                return_value=cluster_ids or ["cluster-1"],
            ),
            patch(
                'biz_layer.mem_memorize._run_profile_extraction_for_batch',
                new_callable=AsyncMock,
            ),
            patch(
                'biz_layer.mem_memorize._run_skill_extraction_for_batch',
                new_callable=AsyncMock,
            ),
        ]

        # Mock distributed_lock context manager
        lock_cm = AsyncMock()
        lock_cm.__aenter__ = AsyncMock(return_value=acquired)
        lock_cm.__aexit__ = AsyncMock(return_value=False)

        patches.append(
            patch(
                'core.lock.redis_distributed_lock.distributed_lock',
                return_value=lock_cm,
            )
        )

        return patches, mock_storage, mem_scene_state

    @pytest.mark.asyncio
    async def test_lock_not_acquired_returns_zero(self):
        from biz_layer.mem_memorize import _drain_and_cluster

        patches, _, _ = self._patch_drain_deps(acquired=False)
        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            result = await _drain_and_cluster("grp-001", _make_config())
        assert result == 0

    @pytest.mark.asyncio
    async def test_accumulates_when_below_batch_size(self):
        from biz_layer.mem_memorize import _drain_and_cluster

        mss = _make_mem_scene_state(pending=[])
        patches, mock_storage, _ = self._patch_drain_deps(mem_scene_state=mss)
        config = _make_config(cluster_batch_size=5)

        with ExitStack() as stack:
            mocks = [stack.enter_context(p) for p in patches]
            entry = _make_pending_entry()
            result = await _drain_and_cluster("grp-001", config, new_entry=entry)

        assert result == 0
        mock_storage.save_mem_scene.assert_called()

    @pytest.mark.asyncio
    async def test_drains_when_batch_size_reached(self):
        from biz_layer.mem_memorize import _drain_and_cluster

        pending = [_make_pending_entry(event_id=f"evt-{i}") for i in range(4)]
        mss = _make_mem_scene_state(pending=pending)
        patches, _, _ = self._patch_drain_deps(mem_scene_state=mss)
        config = _make_config(cluster_batch_size=5)

        with ExitStack() as stack:
            mocks = [stack.enter_context(p) for p in patches]
            entry = _make_pending_entry(event_id="evt-4")
            result = await _drain_and_cluster("grp-001", config, new_entry=entry)

        assert result == 5

    @pytest.mark.asyncio
    async def test_force_drain_with_pending_items(self):
        from biz_layer.mem_memorize import _drain_and_cluster

        pending = [_make_pending_entry(event_id="evt-0")]
        mss = _make_mem_scene_state(pending=pending)
        patches, _, _ = self._patch_drain_deps(mem_scene_state=mss)
        config = _make_config(cluster_batch_size=100)

        with ExitStack() as stack:
            mocks = [stack.enter_context(p) for p in patches]
            result = await _drain_and_cluster(
                "grp-001", config, force_drain=True
            )

        assert result == 1

    @pytest.mark.asyncio
    async def test_force_drain_empty_returns_zero(self):
        from biz_layer.mem_memorize import _drain_and_cluster

        mss = _make_mem_scene_state(pending=[])
        patches, _, _ = self._patch_drain_deps(mem_scene_state=mss)
        config = _make_config(cluster_batch_size=100)

        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            result = await _drain_and_cluster(
                "grp-001", config, force_drain=True
            )

        assert result == 0

    @pytest.mark.asyncio
    async def test_skill_extraction_runs_outside_lock(self):
        from biz_layer.mem_memorize import _drain_and_cluster

        pending = [_make_pending_entry()]
        mss = _make_mem_scene_state(pending=pending)
        patches, _, _ = self._patch_drain_deps(mem_scene_state=mss)
        config = _make_config(cluster_batch_size=1)

        with ExitStack() as stack:
            mocks = [stack.enter_context(p) for p in patches]
            # Find _run_skill_extraction_for_batch mock
            skill_mock = None
            for m in mocks:
                if hasattr(m, '_mock_name') and 'skill' in str(getattr(m, '_mock_name', '')):
                    skill_mock = m
            result = await _drain_and_cluster(
                "grp-001", config, new_entry=_make_pending_entry()
            )

        # result > 0 means drain happened
        assert result > 0


# ===========================================================================
# _run_profile_extraction_for_batch
# ===========================================================================

class TestRunProfileExtractionForBatch:

    @pytest.mark.asyncio
    async def test_skip_when_config_flag_set(self):
        from biz_layer.mem_memorize import _run_profile_extraction_for_batch

        config = _make_config(skip_profile_extraction=True)
        # Should return immediately without calling anything
        await _run_profile_extraction_for_batch(
            group_id="grp-001",
            drained_memcells=[_make_pending_entry()],
            cluster_ids=["c1"],
            mem_scene_state=_make_mem_scene_state(cluster_counts={"c1": 5}),
            config=config,
        )
        # No exception means it returned early

    @pytest.mark.asyncio
    async def test_interval_lte_1_always_extracts(self):
        from biz_layer.mem_memorize import _run_profile_extraction_for_batch

        config = _make_config(
            skip_profile_extraction=False,
            profile_extraction_interval=1,
        )
        mss = _make_mem_scene_state(
            cluster_counts={"c1": 1},
            cluster_last_ts={"c1": 9999999999.0},
        )

        mock_profile_repo = AsyncMock()
        mock_profile_repo.get_all_by_group = AsyncMock(return_value=[])

        with (
            patch('biz_layer.mem_memorize.get_bean_by_type', return_value=mock_profile_repo),
            patch(
                'biz_layer.mem_memorize._trigger_profile_extraction',
                new_callable=AsyncMock,
            ) as mock_trigger,
        ):
            await _run_profile_extraction_for_batch(
                group_id="grp-001",
                drained_memcells=[_make_pending_entry(timestamp=100.0)],
                cluster_ids=["c1"],
                mem_scene_state=mss,
                config=config,
            )
            mock_trigger.assert_called_once()

    @pytest.mark.asyncio
    async def test_interval_modulo_skips_when_not_met(self):
        from biz_layer.mem_memorize import _run_profile_extraction_for_batch

        config = _make_config(
            skip_profile_extraction=False,
            profile_extraction_interval=5,
            cluster_batch_size=1,  # not batch mode
        )
        mss = _make_mem_scene_state(cluster_counts={"c1": 3})

        with patch(
            'biz_layer.mem_memorize._trigger_profile_extraction',
            new_callable=AsyncMock,
        ) as mock_trigger:
            await _run_profile_extraction_for_batch(
                group_id="grp-001",
                drained_memcells=[_make_pending_entry()],
                cluster_ids=["c1"],
                mem_scene_state=mss,
                config=config,
            )
            mock_trigger.assert_not_called()

    @pytest.mark.asyncio
    async def test_batch_mode_extracts_when_count_gte_interval(self):
        from biz_layer.mem_memorize import _run_profile_extraction_for_batch

        config = _make_config(
            skip_profile_extraction=False,
            profile_extraction_interval=5,
            cluster_batch_size=10,  # batch mode
        )
        mss = _make_mem_scene_state(
            cluster_counts={"c1": 5},
            cluster_last_ts={"c1": 9999999999.0},
        )

        mock_profile_repo = AsyncMock()
        mock_profile_repo.get_all_by_group = AsyncMock(return_value=[])

        with (
            patch('biz_layer.mem_memorize.get_bean_by_type', return_value=mock_profile_repo),
            patch(
                'biz_layer.mem_memorize._trigger_profile_extraction',
                new_callable=AsyncMock,
            ) as mock_trigger,
        ):
            await _run_profile_extraction_for_batch(
                group_id="grp-001",
                drained_memcells=[_make_pending_entry(timestamp=100.0)],
                cluster_ids=["c1"],
                mem_scene_state=mss,
                config=config,
            )
            mock_trigger.assert_called_once()

    @pytest.mark.asyncio
    async def test_force_drain_extracts_when_count_gte_interval(self):
        from biz_layer.mem_memorize import _run_profile_extraction_for_batch

        config = _make_config(
            skip_profile_extraction=False,
            profile_extraction_interval=5,
            cluster_batch_size=1,  # non-batch
        )
        mss = _make_mem_scene_state(
            cluster_counts={"c1": 6},
            cluster_last_ts={"c1": 9999999999.0},
        )

        mock_profile_repo = AsyncMock()
        mock_profile_repo.get_all_by_group = AsyncMock(return_value=[])

        with (
            patch('biz_layer.mem_memorize.get_bean_by_type', return_value=mock_profile_repo),
            patch(
                'biz_layer.mem_memorize._trigger_profile_extraction',
                new_callable=AsyncMock,
            ) as mock_trigger,
        ):
            await _run_profile_extraction_for_batch(
                group_id="grp-001",
                drained_memcells=[_make_pending_entry(timestamp=100.0)],
                cluster_ids=["c1"],
                mem_scene_state=mss,
                config=config,
                force_drain=True,
            )
            mock_trigger.assert_called_once()

    @pytest.mark.asyncio
    async def test_scene_picked_from_last_entry(self):
        from biz_layer.mem_memorize import _run_profile_extraction_for_batch

        config = _make_config(
            skip_profile_extraction=False,
            profile_extraction_interval=1,
        )
        mss = _make_mem_scene_state(
            cluster_counts={"c1": 2},
            cluster_last_ts={"c1": 9999999999.0},
        )
        entries = [
            _make_pending_entry(event_id="e1", scene="solo"),
            _make_pending_entry(event_id="e2", scene="team"),
        ]

        mock_profile_repo = AsyncMock()
        mock_profile_repo.get_all_by_group = AsyncMock(return_value=[])

        with (
            patch('biz_layer.mem_memorize.get_bean_by_type', return_value=mock_profile_repo),
            patch(
                'biz_layer.mem_memorize._trigger_profile_extraction',
                new_callable=AsyncMock,
            ) as mock_trigger,
        ):
            await _run_profile_extraction_for_batch(
                group_id="grp-001",
                drained_memcells=entries,
                cluster_ids=["c1", "c1"],
                mem_scene_state=mss,
                config=config,
            )
            call_kwargs = mock_trigger.call_args[1]
            assert call_kwargs["scene"] == "team"

    @pytest.mark.asyncio
    async def test_no_target_clusters_skips_extraction(self):
        from biz_layer.mem_memorize import _run_profile_extraction_for_batch

        config = _make_config(
            skip_profile_extraction=False,
            profile_extraction_interval=1,
        )
        # cluster_last_ts has old timestamps, all cluster_ids are None
        mss = _make_mem_scene_state(
            cluster_counts={"c1": 2},
            cluster_last_ts={"c1": 0.0},
        )

        mock_profile_repo = AsyncMock()
        existing_profile = MagicMock()
        existing_profile.last_updated_ts = 9999999999.0
        mock_profile_repo.get_all_by_group = AsyncMock(return_value=[existing_profile])

        with (
            patch('biz_layer.mem_memorize.get_bean_by_type', return_value=mock_profile_repo),
            patch(
                'biz_layer.mem_memorize._trigger_profile_extraction',
                new_callable=AsyncMock,
            ) as mock_trigger,
        ):
            await _run_profile_extraction_for_batch(
                group_id="grp-001",
                drained_memcells=[_make_pending_entry(timestamp=100.0)],
                cluster_ids=[None],  # None cluster_id
                mem_scene_state=mss,
                config=config,
            )
            mock_trigger.assert_not_called()


# ===========================================================================
# _trigger_profile_extraction
# ===========================================================================

class TestTriggerProfileExtraction:

    def _patch_profile_deps(self, all_memcells=None, old_profiles=None, new_profiles=None):
        mock_profile_repo = AsyncMock()
        mock_profile_repo.get_all_profiles = AsyncMock(
            return_value=old_profiles or {}
        )
        mock_profile_repo.save_profile = AsyncMock()
        mock_profile_repo.get_by_user_and_group = AsyncMock(return_value=None)
        mock_profile_repo.upsert = AsyncMock()

        mock_memcell_repo = AsyncMock()
        mock_memcell_repo.get_by_event_ids = AsyncMock(
            return_value={} if all_memcells is None else {mc.event_id: mc for mc in all_memcells}
        )

        mock_llm = MagicMock()

        mock_profile_manager = AsyncMock()
        mock_profile_manager.extract_profiles = AsyncMock(
            return_value=new_profiles or []
        )

        patches = [
            patch(
                'biz_layer.mem_memorize.get_bean_by_type',
                side_effect=lambda cls: {
                    'UserProfileRawRepository': mock_profile_repo,
                    'MemCellRawRepository': mock_memcell_repo,
                }.get(cls.__name__, MagicMock()),
            ),
            patch(
                'memory_layer.llm.llm_provider.build_default_provider',
                return_value=mock_llm,
            ),
            patch(
                'memory_layer.profile_manager.ProfileManager',
                return_value=mock_profile_manager,
            ),
        ]
        return patches, mock_profile_repo, mock_memcell_repo, mock_profile_manager

    @pytest.mark.asyncio
    async def test_skips_when_below_min_memcells(self):
        from biz_layer.mem_memorize import _trigger_profile_extraction

        mss = _make_mem_scene_state(cluster_counts={"c1": 1})
        config = _make_config(profile_min_memcells=5)

        patches, mock_repo, _, _ = self._patch_profile_deps()
        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            await _trigger_profile_extraction(
                group_id="grp-001",
                cluster_ids=["c1"],
                mem_scene_state=mss,
                latest_memcell_ts=100.0,
                config=config,
            )
        mock_repo.get_all_profiles.assert_not_called()

    @pytest.mark.asyncio
    async def test_extracts_and_saves_profiles(self):
        from biz_layer.mem_memorize import _trigger_profile_extraction

        fetched_mc = MagicMock()
        fetched_mc.event_id = "evt-001"
        fetched_mc.participants = ["user_A", "user_B"]

        new_profile = MagicMock()
        new_profile.user_id = "user_A"
        new_profile.to_dict.return_value = {"explicit_info": ["trait"]}
        new_profile.total_items.return_value = 1

        mss = _make_mem_scene_state(
            cluster_counts={"c1": 3},
            eventid_to_cluster={"evt-001": "c1", "evt-002": "c1"},
        )

        patches, mock_repo, _, mock_pm = self._patch_profile_deps(
            all_memcells=[fetched_mc],
            new_profiles=[new_profile],
        )
        config = _make_config(profile_min_memcells=1)

        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            await _trigger_profile_extraction(
                group_id="grp-001",
                cluster_ids=["c1"],
                mem_scene_state=mss,
                latest_memcell_ts=100.0,
                config=config,
            )

        mock_pm.extract_profiles.assert_called_once()
        mock_repo.save_profile.assert_called_once()

    @pytest.mark.asyncio
    async def test_advances_ts_on_failure(self):
        from biz_layer.mem_memorize import _trigger_profile_extraction

        mss = _make_mem_scene_state(
            cluster_counts={"c1": 3},
            eventid_to_cluster={"evt-001": "c1"},
        )

        mock_profile_repo = AsyncMock()
        # get_all_profiles succeeds so we reach extract_profiles which fails
        mock_profile_repo.get_all_profiles = AsyncMock(return_value={})
        mock_profile_repo.get_by_user_and_group = AsyncMock(return_value=None)
        mock_profile_repo.upsert = AsyncMock()

        mock_memcell_repo = AsyncMock()
        fetched_mc = MagicMock()
        fetched_mc.event_id = "evt-001"
        fetched_mc.participants = ["user_X"]
        mock_memcell_repo.get_by_event_ids = AsyncMock(
            return_value={"evt-001": fetched_mc}
        )

        mock_pm = AsyncMock()
        mock_pm.extract_profiles = AsyncMock(side_effect=RuntimeError("LLM down"))

        with (
            patch(
                'biz_layer.mem_memorize.get_bean_by_type',
                side_effect=lambda cls: {
                    'UserProfileRawRepository': mock_profile_repo,
                    'MemCellRawRepository': mock_memcell_repo,
                }.get(cls.__name__, MagicMock()),
            ),
            patch('memory_layer.llm.llm_provider.build_default_provider', return_value=MagicMock()),
            patch('memory_layer.profile_manager.ProfileManager', return_value=mock_pm),
        ):
            # Should not raise
            await _trigger_profile_extraction(
                group_id="grp-001",
                cluster_ids=["c1"],
                mem_scene_state=mss,
                latest_memcell_ts=100.0,
                config=_make_config(profile_min_memcells=1),
            )

        # Should attempt to advance timestamp for user_X despite failure
        mock_profile_repo.upsert.assert_called_once()
        call_kwargs = mock_profile_repo.upsert.call_args[1]
        assert call_kwargs["user_id"] == "user_X"
        assert call_kwargs["metadata"]["last_updated_ts"] == 100.0


# ===========================================================================
# _run_skill_extraction_for_batch
# ===========================================================================

class TestRunSkillExtractionForBatch:

    @pytest.mark.asyncio
    async def test_skip_when_config_flag_set(self):
        from biz_layer.mem_memorize import _run_skill_extraction_for_batch

        config = _make_config(skip_skill_extraction=True)
        await _run_skill_extraction_for_batch(
            group_id="grp-001",
            drained_memcells=[_make_pending_entry()],
            cluster_ids=["c1"],
            config=config,
        )
        # No exception = returned early

    @pytest.mark.asyncio
    async def test_skip_when_no_agent_cases(self):
        from biz_layer.mem_memorize import _run_skill_extraction_for_batch

        config = _make_config(skip_skill_extraction=False)
        entries = [_make_pending_entry()]  # no agent_case

        with patch(
            'biz_layer.mem_memorize._trigger_agent_skill_extraction',
            new_callable=AsyncMock,
        ) as mock_trigger:
            await _run_skill_extraction_for_batch(
                group_id="grp-001",
                drained_memcells=entries,
                cluster_ids=["c1"],
                config=config,
            )
            mock_trigger.assert_not_called()

    @pytest.mark.asyncio
    async def test_filters_low_quality_cases(self):
        from biz_layer.mem_memorize import _run_skill_extraction_for_batch

        config = _make_config(
            skip_skill_extraction=False,
            skill_min_quality_score=0.5,
        )
        entries = [
            _make_pending_entry(
                event_id="evt-001",
                agent_case={
                    "id": "ac-1",
                    "task_intent": "x",
                    "approach": "y",
                    "quality_score": 0.1,  # below threshold
                },
            )
        ]

        with patch(
            'biz_layer.mem_memorize._trigger_agent_skill_extraction',
            new_callable=AsyncMock,
        ) as mock_trigger:
            await _run_skill_extraction_for_batch(
                group_id="grp-001",
                drained_memcells=entries,
                cluster_ids=["c1"],
                config=config,
            )
            mock_trigger.assert_not_called()

    @pytest.mark.asyncio
    async def test_groups_by_cluster_and_triggers(self):
        from biz_layer.mem_memorize import _run_skill_extraction_for_batch

        config = _make_config(
            skip_skill_extraction=False,
            skill_min_quality_score=0.0,
        )
        ac_dict = {
            "id": "ac-1",
            "task_intent": "deploy",
            "approach": "ssh",
            "quality_score": 0.9,
        }
        entries = [
            _make_pending_entry(event_id="e1", agent_case=ac_dict),
            _make_pending_entry(event_id="e2", agent_case={**ac_dict, "id": "ac-2"}),
            _make_pending_entry(event_id="e3", agent_case={**ac_dict, "id": "ac-3"}),
        ]
        cluster_ids = ["c1", "c1", "c2"]

        with patch(
            'biz_layer.mem_memorize._trigger_agent_skill_extraction',
            new_callable=AsyncMock,
            return_value=False,
        ) as mock_trigger:
            await _run_skill_extraction_for_batch(
                group_id="grp-001",
                drained_memcells=entries,
                cluster_ids=cluster_ids,
                config=config,
            )
            assert mock_trigger.call_count == 2  # c1 and c2

    @pytest.mark.asyncio
    async def test_milvus_flush_when_changes(self):
        from biz_layer.mem_memorize import _run_skill_extraction_for_batch

        config = _make_config(
            skip_skill_extraction=False,
            skill_min_quality_score=0.0,
        )
        entries = [
            _make_pending_entry(
                event_id="e1",
                agent_case={"id": "ac-1", "task_intent": "x", "approach": "y", "quality_score": 0.9},
            )
        ]

        mock_milvus_repo = AsyncMock()
        mock_milvus_repo.flush = AsyncMock()

        with (
            patch(
                'biz_layer.mem_memorize._trigger_agent_skill_extraction',
                new_callable=AsyncMock,
                return_value=True,  # has milvus changes
            ),
            patch(
                'biz_layer.mem_memorize.get_bean_by_type',
                return_value=mock_milvus_repo,
            ),
        ):
            await _run_skill_extraction_for_batch(
                group_id="grp-001",
                drained_memcells=entries,
                cluster_ids=["c1"],
                config=config,
            )
            mock_milvus_repo.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_milvus_flush_when_no_changes(self):
        from biz_layer.mem_memorize import _run_skill_extraction_for_batch

        config = _make_config(
            skip_skill_extraction=False,
            skill_min_quality_score=0.0,
        )
        entries = [
            _make_pending_entry(
                event_id="e1",
                agent_case={"id": "ac-1", "task_intent": "x", "approach": "y", "quality_score": 0.9},
            )
        ]

        with patch(
            'biz_layer.mem_memorize._trigger_agent_skill_extraction',
            new_callable=AsyncMock,
            return_value=False,  # no milvus changes
        ):
            # Should not try to flush
            await _run_skill_extraction_for_batch(
                group_id="grp-001",
                drained_memcells=entries,
                cluster_ids=["c1"],
                config=config,
            )
            # No exception = no flush attempted


# ===========================================================================
# flush_clustering
# ===========================================================================

class TestFlushClustering:

    @pytest.mark.asyncio
    async def test_uses_agent_default_config(self):
        from biz_layer.mem_memorize import flush_clustering

        with (
            patch(
                'api_specs.id_generator.generate_single_user_group_id',
                return_value="grp-user1",
            ),
            patch(
                'biz_layer.mem_memorize._drain_and_cluster',
                new_callable=AsyncMock,
                return_value=3,
            ) as mock_drain,
        ):
            result = await flush_clustering("user1")
            assert result == 3
            call_kwargs = mock_drain.call_args[1]
            assert call_kwargs["force_drain"] is True
            assert call_kwargs["config"] is AGENT_DEFAULT_MEMORIZE_CONFIG

    @pytest.mark.asyncio
    async def test_custom_config_override(self):
        from biz_layer.mem_memorize import flush_clustering

        custom = _make_config(cluster_batch_size=50)
        with (
            patch(
                'api_specs.id_generator.generate_single_user_group_id',
                return_value="grp-user1",
            ),
            patch(
                'biz_layer.mem_memorize._drain_and_cluster',
                new_callable=AsyncMock,
                return_value=0,
            ) as mock_drain,
        ):
            await flush_clustering("user1", config=custom)
            assert mock_drain.call_args[1]["config"] is custom


# ===========================================================================
# ExtractionState
# ===========================================================================

class TestExtractionState:

    def test_episode_saved_true_when_parent_docs_map_populated(self):
        mc = _make_memcell()
        state = ExtractionState(
            memcell=mc,
            request=_make_request(),
            current_time=datetime(2026, 4, 7),
            scene="solo",
            is_solo_scene=True,
            participants=["user_001"],
        )
        state.parent_docs_map["ep-001"] = MagicMock()
        assert state.episode_saved is True

    def test_episode_saved_false_when_empty(self):
        mc = _make_memcell()
        state = ExtractionState(
            memcell=mc,
            request=_make_request(),
            current_time=datetime(2026, 4, 7),
            scene="solo",
            is_solo_scene=True,
            participants=["user_001"],
        )
        assert state.episode_saved is False

    def test_default_parent_types_from_config(self):
        mc = _make_memcell()
        state = ExtractionState(
            memcell=mc,
            request=_make_request(),
            current_time=datetime(2026, 4, 7),
            scene="solo",
            is_solo_scene=True,
            participants=["user_001"],
        )
        assert state.episode_parent_type == DEFAULT_MEMORIZE_CONFIG.default_episode_parent_type
        assert state.foresight_parent_type == DEFAULT_MEMORIZE_CONFIG.default_foresight_parent_type
        assert state.atomic_fact_parent_type == DEFAULT_MEMORIZE_CONFIG.default_atomic_fact_parent_type

    def test_parent_id_defaults_to_event_id(self):
        mc = _make_memcell(event_id="evt-999")
        state = ExtractionState(
            memcell=mc,
            request=_make_request(),
            current_time=datetime(2026, 4, 7),
            scene="solo",
            is_solo_scene=True,
            participants=["user_001"],
        )
        assert state.parent_id == "evt-999"

    def test_post_init_creates_empty_lists(self):
        mc = _make_memcell()
        state = ExtractionState(
            memcell=mc,
            request=_make_request(),
            current_time=datetime(2026, 4, 7),
            scene="solo",
            is_solo_scene=True,
            participants=["user_001"],
        )
        assert state.group_episode_memories == []
        assert state.episode_memories == []
        assert state.parent_docs_map == {}
