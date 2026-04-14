"""
Type definitions for the OpenClaw benchmark adapter.

All TypedDicts here are the contract surface between Python and the Node
bridge (BridgeCommand/Response) and between adapter add()/search() and the
rest of the EverMemOS pipeline (ConversationSandboxHandle, OpenClawIndexHandle).

Class declaration order is chosen so that nested TypedDict references resolve
at runtime without needing `from __future__ import annotations` — dependents
appear below their dependencies.
"""
from typing import Literal, TypedDict


BackendMode = Literal["fts_only", "vector", "hybrid"]
RetrievalRoute = Literal["search_only", "search_then_get"]
VisibilityMode = Literal["settled", "eventual"]
VisibilityState = Literal["prepared", "ingested", "indexed", "settled"]
RunStatus = Literal["pending", "running", "failed", "ready"]
BridgeCommandName = Literal["index", "flush", "search", "get", "status"]


class BridgeArtifactLocator(TypedDict, total=False):
    """
    Points to a specific OpenClaw artifact the Node bridge can return.

    Two kinds:
    - index_doc: an indexed document identified by doc_id
    - memory_file_range: a line span within a markdown memory file (path_rel
      relative to the sandbox workspace_dir)
    """
    kind: Literal["index_doc", "memory_file_range"]
    doc_id: str
    path_rel: str
    line_start: int
    line_end: int


class BridgeSearchHit(TypedDict, total=False):
    score: float
    snippet: str
    artifact_locator: BridgeArtifactLocator
    metadata: dict


class BridgeCommand(TypedDict, total=False):
    command: BridgeCommandName
    config_path: str
    workspace_dir: str
    state_dir: str
    query: str
    top_k: int
    artifact_locator: BridgeArtifactLocator


class BridgeResponse(TypedDict, total=False):
    ok: bool
    command: BridgeCommandName
    hits: list[BridgeSearchHit]
    artifact_locator: BridgeArtifactLocator
    flush_epoch: int
    index_epoch: int
    input_artifacts: list[BridgeArtifactLocator]
    output_artifacts: list[BridgeArtifactLocator]
    active_artifacts: list[BridgeArtifactLocator]
    settled: bool
    error: str


class ConversationSandboxHandle(TypedDict):
    """
    Per-conversation sandbox descriptor.

    run_status and visibility_state are orthogonal:
    - run_status: lifecycle outcome of add() for this conversation
    - visibility_state: whether the data is actually queryable by search()
    """
    conversation_id: str
    workspace_dir: str
    native_store_dir: str
    resolved_config_path: str
    session_manifest_path: str
    prov_units_path: str
    artifact_bindings_path: str
    events_path: str
    metrics_dir: str
    backend_mode: BackendMode
    retrieval_route: RetrievalRoute
    visibility_mode: VisibilityMode
    visibility_state: VisibilityState
    run_status: RunStatus
    last_flush_epoch: int
    last_index_epoch: int
    retrieval_eval_supported: bool


class OpenClawIndexHandle(TypedDict):
    type: str
    run_id: str
    root_dir: str
    conversations: dict[str, ConversationSandboxHandle]
