from __future__ import annotations

import pytest

from agent_runtime_kit.agent.provider_contracts import (
    ProviderControlAction,
    ProviderControlRequest,
    ProviderExecutionKind,
    ProviderRegistry,
    ProviderRunState,
)

from .provider_contract_harness import (
    assert_immediate_terminal_contract,
    assert_interrupt_contract,
    make_contract_fake_bundle,
    standard_request,
)


@pytest.mark.parametrize(
    "execution_kind",
    [
        ProviderExecutionKind.SDK,
        ProviderExecutionKind.SUBPROCESS_RPC,
        ProviderExecutionKind.PYTHON_LIBRARY,
    ],
)
def test_same_contract_accepts_sdk_subprocess_and_library_shapes(execution_kind) -> None:  # noqa: ANN001
    bundle = make_contract_fake_bundle(execution_kind=execution_kind)
    ProviderRegistry((bundle,))
    assert_immediate_terminal_contract(bundle)


def test_interrupt_contract_waits_for_terminal_barrier() -> None:
    bundle = make_contract_fake_bundle(initial_state=ProviderRunState.RUNNING)
    assert_interrupt_contract(bundle)


def test_timeout_does_not_implicitly_cancel_active_run() -> None:
    bundle = make_contract_fake_bundle(initial_state=ProviderRunState.RUNNING)
    handle = bundle.runtime.start(standard_request())
    with pytest.raises(TimeoutError):
        handle.wait_terminal(timeout_s=0.001)
    assert handle.poll_state() is ProviderRunState.RUNNING


def test_needs_input_is_resumable_through_neutral_control_action() -> None:
    bundle = make_contract_fake_bundle(initial_state=ProviderRunState.NEEDS_INPUT)
    handle = bundle.runtime.start(standard_request())
    waiting = handle.wait_terminal(timeout_s=0.1)
    assert waiting.status is ProviderRunState.NEEDS_INPUT

    control = handle.control(
        ProviderControlRequest(
            action=ProviderControlAction.RESPOND_INPUT,
            requested_at="2026-07-21T10:00:00Z",
            content={"answer": "continue"},
        )
    )
    assert control.accepted
    assert control.terminal_confirmed
    assert handle.wait_terminal(timeout_s=0.1).status is ProviderRunState.COMPLETED


def test_event_cursor_is_monotonic_and_does_not_repeat_events() -> None:
    bundle = make_contract_fake_bundle(initial_state=ProviderRunState.RUNNING)
    handle = bundle.runtime.start(standard_request())
    first = handle.drain_events()
    second = handle.drain_events(after_cursor=first.next_cursor)
    assert [event.sequence for event in first.events] == [0]
    assert second.events == ()
