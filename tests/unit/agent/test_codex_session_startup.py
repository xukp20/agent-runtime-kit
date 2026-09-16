from unittest.mock import Mock

import pytest
from openai_codex.errors import InternalRpcError, TransportClosedError

from agent_runtime_kit.agent.diagnostics import exception_diagnostics
from agent_runtime_kit.agent.providers.codex import (
    CodexProvider, CodexTransientRetryPolicy, _context_rpc,
    _required_mcp_startup_failure,
)


def failure(details='lc_app: MCP client startup timed out after 30s', operation='resuming'):
    return InternalRpcError(-32603, f'error {operation} thread: required MCP servers failed to initialize: {details}')


@pytest.mark.parametrize('method', ['thread/start', 'thread/resume'])
def test_explicit_required_mcp_timeout_retries_once_before_returning_session(method):
    provider = CodexProvider(transient_retry_policy=CodexTransientRetryPolicy(initial_delay_s=0))
    call = Mock(side_effect=[failure(), 'session'])
    events = []
    assert provider._initialize_session_with_retry(method=method, call=call, on_transient_retry=events.append) == 'session'
    assert call.call_count == 2
    assert events[0]['classification'] == 'required_mcp_startup_timeout'
    assert events[0]['turn_id'] is None
    assert events[0]['max_attempts'] == 2


def test_repeated_timeout_preserves_exception_and_final_attempt_diagnostics():
    provider = CodexProvider(transient_retry_policy=CodexTransientRetryPolicy(initial_delay_s=0))
    call = Mock(side_effect=lambda: (_ for _ in ()).throw(failure()))
    with pytest.raises(InternalRpcError) as caught:
        provider._initialize_session_with_retry(method='thread/resume', call=call, on_transient_retry=None)
    assert call.call_count == 2
    context = exception_diagnostics(caught.value)['context']
    assert context['rpc_attempt'] == 2
    assert context['rpc_category'] == 'required_mcp_startup_timeout'
    assert context['rpc_mcp_lc_app'] is True


@pytest.mark.parametrize('error', [
    InternalRpcError(-32603, 'internal error'),
    InternalRpcError(-32603, 'thread operation timed out after 30s'),
    TransportClosedError('closed'),
    failure('lc_app: MCP client startup timed out after 30s; lc_submit: unauthorized'),
    failure('lc_app: timed out waiting for MCP event stream response headers'),
    failure('lc_app: MCP startup cancelled'),
    failure('lc_app: arbitrary failure with secret-token-sentinel'),
])
def test_unknown_or_mixed_failures_are_not_replayed(error):
    provider = CodexProvider(transient_retry_policy=CodexTransientRetryPolicy(initial_delay_s=0))
    call = Mock(side_effect=error)
    with pytest.raises(type(error)):
        provider._initialize_session_with_retry(method='thread/resume', call=call, on_transient_retry=None)
    assert call.call_count == 1
    assert 'secret-token-sentinel' not in str(exception_diagnostics(error))


def test_both_servers_and_fractional_timeout_are_classified_without_raw_names():
    result = _required_mcp_startup_failure(failure(
        'lc_app: MCP client startup timed out after 30s; lc_submit: MCP client startup timed out after 1.5s'))
    assert result == {'rpc_category': 'required_mcp_startup_timeout', 'rpc_mcp_failure_count': 2,
                      'rpc_mcp_lc_app': True, 'rpc_mcp_lc_submit': True}
    error = failure('private_server: unauthorized secret-token-sentinel')
    with pytest.raises(InternalRpcError):
        with _context_rpc('thread/resume'):
            raise error
    safe = str(exception_diagnostics(error))
    assert 'private_server' not in safe and 'secret-token-sentinel' not in safe
    assert 'required_mcp_startup_failed' in safe


def test_policy_can_disable_session_retry():
    provider = CodexProvider(transient_retry_policy=CodexTransientRetryPolicy(max_attempts=1))
    call = Mock(side_effect=failure())
    with pytest.raises(InternalRpcError):
        provider._initialize_session_with_retry(method='thread/start', call=call, on_transient_retry=None)
    assert call.call_count == 1


def test_wrapped_mcp_error_reports_indicators_without_enabling_retry_or_exposing_text():
    error = InternalRpcError(-32603,
        'error resuming thread: session failed: required MCP servers failed to initialize: '
        'lc_app: handshaking timed out; private_server: secret-token-sentinel',
        {'private_key': 'secret-data-sentinel'})
    provider = CodexProvider(transient_retry_policy=CodexTransientRetryPolicy(initial_delay_s=0))
    call = Mock(side_effect=error)
    with pytest.raises(InternalRpcError):
        provider._initialize_session_with_retry(method='thread/resume', call=call, on_transient_retry=None)
    diagnostics = exception_diagnostics(error)
    context = diagnostics['context']
    assert context['rpc_category'] == 'unclassified'
    assert context['rpc_message_required_mcp'] is True
    assert context['rpc_message_handshake'] is True
    assert context['rpc_message_timeout'] is True
    assert context['rpc_message_lc_app'] is True
    assert context['rpc_message_startup_timeout'] is False
    assert call.call_count == 1
    assert 'secret-token-sentinel' not in str(diagnostics)
    assert 'secret-data-sentinel' not in str(diagnostics)
    assert 'private_server' not in str(diagnostics)


def test_oversized_rpc_message_only_exports_length():
    error = InternalRpcError(-32603, 'required MCP servers failed to initialize' + 'x' * 16384)
    with pytest.raises(InternalRpcError):
        with _context_rpc('thread/resume'):
            raise error
    context = exception_diagnostics(error)['context']
    assert context['rpc_message_length'] == len(error.message)
    assert 'rpc_message_required_mcp' not in context
    assert context['rpc_category'] == 'unclassified'


def test_rpc_indicator_values_are_strict_booleans():
    error = InternalRpcError(-32603, 'internal error')
    error.ark_context_diagnostics = {'rpc_message_required_mcp': 'secret-token-sentinel'}
    assert 'rpc_message_required_mcp' not in exception_diagnostics(error)['context']
