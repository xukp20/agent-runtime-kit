from __future__ import annotations

from pathlib import Path
from typing import Mapping, Protocol

from .artifacts import (
    ArtifactCaptureRequest,
    ArtifactDescribeRequest,
    ArtifactRestoreRequest,
    ArtifactStabilityRequest,
    ArtifactStabilityResult,
    ProviderArtifactManifest,
    ProviderArtifactRestoreResult,
    ProviderArtifactSnapshot,
)
from .capabilities import ProviderCapabilities
from .homes import (
    HomeInitializationResult,
    HomeMaterializationResult,
    HomeValidationResult,
    ProviderExecutionContext,
    ProviderHomeSpec,
)
from .identities import ModelBackendIdentity
from .locators import ProviderSessionLocator, ProviderTurnLocator
from .models import (
    AgentEvent,
    AgentToolCall,
    Page,
    ProviderContextCompactionResult,
    ProviderContextUsage,
    ProviderControlRequest,
    ProviderControlResult,
    ProviderEventBatch,
    ProviderForkRequest,
    ProviderForkResult,
    ProviderRunRequest,
    ProviderRunState,
    ProviderTurnResult,
)
from .query import (
    AgentSessionView,
    AgentTurnView,
    ProviderContextCompactionRequest,
    ProviderContextQuery,
    ProviderContextReconcileRequest,
    ProviderEventQuery,
    ProviderSessionListQuery,
    ProviderSessionQuery,
    ProviderToolQuery,
    ProviderTurnQuery,
    ProviderUsageQuery,
)
from .usage import AgentSessionUsage, AgentTurnUsage


class HomeRecordView(Protocol):
    provider_type: str
    home_id: str


class ProviderCapabilityResolver(Protocol):
    def resolve_capabilities(
        self,
        home: HomeRecordView,
        model_backend: ModelBackendIdentity | None = None,
    ) -> ProviderCapabilities: ...


class ProviderHomeRenderer(Protocol):
    provider_type: str

    def validate(self, spec: ProviderHomeSpec) -> HomeValidationResult: ...

    def materialize(self, spec: ProviderHomeSpec, home_root: Path) -> HomeMaterializationResult: ...

    def refresh_materialization(
        self,
        home: HomeRecordView,
        home_root: Path,
    ) -> HomeMaterializationResult: ...

    def initialize(
        self,
        home: HomeRecordView,
        ctx: ProviderExecutionContext,
    ) -> HomeInitializationResult: ...

    def build_execution_context(
        self,
        home: HomeRecordView,
        *,
        run_env: Mapping[str, str] | None,
        workdir: str | None,
    ) -> ProviderExecutionContext: ...


class ProviderRunHandle(Protocol):
    @property
    def run_id(self) -> str: ...

    def session_locator(self) -> ProviderSessionLocator | None: ...

    def turn_locator(self) -> ProviderTurnLocator | None: ...

    def poll_state(self) -> ProviderRunState: ...

    def drain_events(self, after_cursor: str | None = None) -> ProviderEventBatch: ...

    def wait_terminal(self, timeout_s: float | None = None) -> ProviderTurnResult: ...

    def interrupt(self, timeout_s: float | None = None) -> ProviderControlResult: ...

    def control(self, request: ProviderControlRequest) -> ProviderControlResult: ...

    def close(self) -> None: ...


class ProviderRuntimeAdapter(Protocol):
    provider_type: str

    def start(self, request: ProviderRunRequest) -> ProviderRunHandle: ...

    def resume(self, request: ProviderRunRequest) -> ProviderRunHandle: ...

    def fork(self, request: ProviderForkRequest) -> ProviderForkResult: ...

    def control(self, request: ProviderControlRequest) -> ProviderControlResult: ...

    def close_session(self, locator: ProviderSessionLocator) -> ProviderControlResult: ...

    def close(self) -> None: ...


class ProviderQueryAdapter(Protocol):
    provider_type: str

    def list_sessions(self, query: ProviderSessionListQuery) -> Page: ...

    def read_session(self, query: ProviderSessionQuery) -> AgentSessionView: ...

    def list_turns(self, query: ProviderTurnQuery) -> Page: ...

    def read_turn(self, query: ProviderTurnQuery) -> AgentTurnView | None: ...

    def list_events(self, query: ProviderEventQuery) -> Page: ...

    def list_tool_calls(self, query: ProviderToolQuery) -> Page: ...

    def read_usage(self, query: ProviderUsageQuery) -> AgentTurnUsage | AgentSessionUsage: ...


class ProviderContextAdapter(Protocol):
    provider_type: str

    def inspect(self, request: ProviderContextQuery) -> ProviderContextUsage: ...

    def compact(self, request: ProviderContextCompactionRequest) -> ProviderContextCompactionResult: ...

    def reconcile(
        self,
        request: ProviderContextReconcileRequest,
    ) -> ProviderContextCompactionResult | None: ...


class ProviderArtifactAdapter(Protocol):
    provider_type: str

    def wait_quiescent(self, request: ArtifactStabilityRequest) -> ArtifactStabilityResult: ...

    def describe(self, request: ArtifactDescribeRequest) -> ProviderArtifactManifest: ...

    def capture(self, request: ArtifactCaptureRequest) -> ProviderArtifactSnapshot: ...

    def prepare_restore(self, request: ArtifactRestoreRequest) -> None: ...

    def restore(self, request: ArtifactRestoreRequest) -> ProviderArtifactRestoreResult: ...

    def rebuild_after_restore(self, request: ArtifactRestoreRequest) -> None: ...
