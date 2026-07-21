from __future__ import annotations

from dataclasses import dataclass

from agent_runtime_kit.agent.provider_contracts import (
    AgentError,
    AgentEvent,
    AgentProviderBundle,
    CapabilityKey,
    CapabilityStatus,
    CapabilitySupport,
    HomeInitializationResult,
    HomeMaterializationResult,
    HomeValidationResult,
    ProviderCapabilities,
    ProviderControlAction,
    ProviderControlRequest,
    ProviderControlResult,
    ProviderDescriptor,
    ProviderEventBatch,
    ProviderExecutionKind,
    ProviderHomeKind,
    ProviderRunRequest,
    ProviderRunState,
    ProviderSessionLocator,
    ProviderTurnLocator,
    ProviderTurnResult,
    TokenUsage,
)


_NOW = "2026-07-21T10:00:00Z"


class FakeHomeRenderer:
    def __init__(self, provider_type: str) -> None:
        self.provider_type = provider_type

    def validate(self, spec):  # noqa: ANN001, ANN201
        return HomeValidationResult(valid=spec.provider_type == self.provider_type)

    def materialize(self, spec, home_root):  # noqa: ANN001, ANN201
        return HomeMaterializationResult(
            provider_type=self.provider_type,
            home_id=spec.home_id,
            renderer_version="fake-1",
            manifest_schema_version=1,
            manifest_hash="fake-manifest",
        )

    def initialize(self, home, ctx):  # noqa: ANN001, ANN201
        return HomeInitializationResult(initialized=True)

    def build_execution_context(self, home, *, run_env=None, workdir=None):  # noqa: ANN001, ANN201
        raise NotImplementedError


class ContractFakeRunHandle:
    def __init__(self, request: ProviderRunRequest, *, initial_state: ProviderRunState) -> None:
        self.request = request
        self._run_id = f"run-{request.agent_id}"
        self._session = request.session_locator or ProviderSessionLocator(
            provider_type=request.provider_type,
            session_id=f"session-{request.agent_id}",
            home_id=request.home_id,
            created_at=_NOW,
        )
        self._turn = ProviderTurnLocator(session=self._session, turn_id=f"turn-{request.agent_id}")
        self._state = initial_state
        self._events = [
            AgentEvent(
                provider_type=request.provider_type,
                session_id=self._session.session_id,
                turn_id=self._turn.turn_id,
                sequence=0,
                timestamp=_NOW,
                kind="turn.started",
            )
        ]
        self.closed = False

    @property
    def run_id(self) -> str:
        return self._run_id

    def session_locator(self) -> ProviderSessionLocator:
        return self._session

    def turn_locator(self) -> ProviderTurnLocator:
        return self._turn

    def poll_state(self) -> ProviderRunState:
        return self._state

    def drain_events(self, after_cursor: str | None = None) -> ProviderEventBatch:
        start = int(after_cursor) if after_cursor is not None else 0
        return ProviderEventBatch(
            events=tuple(self._events[start:]),
            next_cursor=str(len(self._events)),
            terminal=self._state.terminal,
        )

    def _result(self) -> ProviderTurnResult:
        error = None
        if self._state is ProviderRunState.FAILED:
            error = AgentError(error_type="fake_failure", message="fake failure")
        return ProviderTurnResult(
            provider_type=self.request.provider_type,
            run_id=self.run_id,
            session_locator=self._session,
            turn_locator=self._turn,
            status=self._state,
            started_at=_NOW,
            completed_at=_NOW,
            final_text="done" if self._state is ProviderRunState.COMPLETED else None,
            error=error,
        )

    def wait_terminal(self, timeout_s: float | None = None) -> ProviderTurnResult:
        if not self._state.terminal and self._state is not ProviderRunState.NEEDS_INPUT:
            raise TimeoutError("fake run is still active")
        return self._result()

    def interrupt(self, timeout_s: float | None = None) -> ProviderControlResult:
        requested_at = _NOW
        if self._state.terminal:
            return ProviderControlResult(
                action=ProviderControlAction.INTERRUPT,
                accepted=False,
                terminal_confirmed=True,
                resulting_state=self._state,
                requested_at=requested_at,
                completed_at=_NOW,
                session_locator=self._session,
                turn_locator=self._turn,
                reason="already terminal",
            )
        self._state = ProviderRunState.INTERRUPTED
        self._events.append(
            AgentEvent(
                provider_type=self.request.provider_type,
                session_id=self._session.session_id,
                turn_id=self._turn.turn_id,
                sequence=len(self._events),
                timestamp=_NOW,
                kind="terminal.interrupted",
                terminal=True,
            )
        )
        return ProviderControlResult(
            action=ProviderControlAction.INTERRUPT,
            accepted=True,
            terminal_confirmed=True,
            resulting_state=self._state,
            requested_at=requested_at,
            completed_at=_NOW,
            session_locator=self._session,
            turn_locator=self._turn,
        )

    def control(self, request: ProviderControlRequest) -> ProviderControlResult:
        if request.action is ProviderControlAction.INTERRUPT:
            return self.interrupt()
        if self._state is ProviderRunState.NEEDS_INPUT and request.action in {
            ProviderControlAction.RESPOND_APPROVAL,
            ProviderControlAction.RESPOND_INPUT,
            ProviderControlAction.REJECT_INPUT,
        }:
            self._state = ProviderRunState.COMPLETED
            return ProviderControlResult(
                action=request.action,
                accepted=True,
                terminal_confirmed=True,
                resulting_state=self._state,
                requested_at=request.requested_at,
                completed_at=_NOW,
                session_locator=self._session,
                turn_locator=self._turn,
            )
        return ProviderControlResult(
            action=request.action,
            accepted=False,
            terminal_confirmed=self._state.terminal,
            resulting_state=self._state,
            requested_at=request.requested_at,
            completed_at=_NOW,
            reason="unsupported fake action",
        )

    def close(self) -> None:
        self.closed = True


class ContractFakeRuntime:
    provider_type = "contract_fake"

    def __init__(self, initial_state: ProviderRunState = ProviderRunState.COMPLETED) -> None:
        self.initial_state = initial_state
        self.handles: list[ContractFakeRunHandle] = []

    def start(self, request: ProviderRunRequest) -> ContractFakeRunHandle:
        handle = ContractFakeRunHandle(request, initial_state=self.initial_state)
        self.handles.append(handle)
        return handle

    def resume(self, request: ProviderRunRequest) -> ContractFakeRunHandle:
        if request.session_locator is None:
            raise ValueError("resume requires session_locator")
        return self.start(request)

    def fork(self, request):  # noqa: ANN001, ANN201
        raise NotImplementedError

    def control(self, request):  # noqa: ANN001, ANN201
        raise NotImplementedError

    def close_session(self, locator):  # noqa: ANN001, ANN201
        raise NotImplementedError

    def close(self) -> None:
        for handle in self.handles:
            handle.close()


def make_contract_fake_bundle(
    *,
    initial_state: ProviderRunState = ProviderRunState.COMPLETED,
    execution_kind: ProviderExecutionKind = ProviderExecutionKind.SDK,
) -> AgentProviderBundle:
    runtime = ContractFakeRuntime(initial_state=initial_state)
    capabilities = ProviderCapabilities(
        provider_type=runtime.provider_type,
        supports={
            CapabilityKey.SESSION_CREATE: CapabilitySupport(
                capability=CapabilityKey.SESSION_CREATE,
                status=CapabilityStatus.NATIVE,
                available=True,
            ),
            CapabilityKey.RUN_WAIT_TERMINAL: CapabilitySupport(
                capability=CapabilityKey.RUN_WAIT_TERMINAL,
                status=CapabilityStatus.ADAPTABLE,
                available=True,
            ),
            CapabilityKey.RUN_INTERRUPT: CapabilitySupport(
                capability=CapabilityKey.RUN_INTERRUPT,
                status=CapabilityStatus.ADAPTABLE,
                available=True,
            ),
        },
    )
    descriptor = ProviderDescriptor(
        provider_type=runtime.provider_type,
        display_name="Contract Fake",
        adapter_version="1",
        execution_kind=execution_kind,
        home_kind=ProviderHomeKind.ARK_OWNED,
        static_capabilities=capabilities,
    )
    return AgentProviderBundle(
        descriptor=descriptor,
        runtime=runtime,
        home_renderer=FakeHomeRenderer(runtime.provider_type),
    )


def standard_request(*, session: ProviderSessionLocator | None = None) -> ProviderRunRequest:
    return ProviderRunRequest(
        agent_id="a-contract",
        scope_id="scope-contract",
        agent_type="ContractAgent",
        provider_type="contract_fake",
        home_id="contract-home",
        prompt="work",
        session_locator=session,
    )


def assert_immediate_terminal_contract(bundle: AgentProviderBundle) -> None:
    handle = bundle.runtime.start(standard_request())
    assert handle.session_locator() is not None
    assert handle.turn_locator() is not None
    result = handle.wait_terminal(timeout_s=0.1)
    assert result.status.terminal
    assert result.session_locator == handle.session_locator()
    assert result.turn_locator == handle.turn_locator()


def assert_interrupt_contract(bundle: AgentProviderBundle) -> None:
    handle = bundle.runtime.start(standard_request())
    result = handle.interrupt(timeout_s=0.1)
    assert result.accepted
    assert result.terminal_confirmed
    assert handle.poll_state() is ProviderRunState.INTERRUPTED
    assert handle.wait_terminal(timeout_s=0.1).status is ProviderRunState.INTERRUPTED
