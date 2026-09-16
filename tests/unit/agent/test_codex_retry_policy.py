from agent_runtime_kit.agent.providers.codex import CodexTransientRetryPolicy


def test_codex_transient_retry_defaults_use_five_attempts_and_three_minute_cap() -> None:
    policy = CodexTransientRetryPolicy()

    assert policy.max_attempts == 5
    assert policy.initial_delay_s == 30.0
    assert policy.max_delay_s == 180.0
    assert [
        min(policy.max_delay_s, policy.initial_delay_s * (2 ** (attempt - 1)))
        for attempt in range(1, policy.max_attempts)
    ] == [30.0, 60.0, 120.0, 180.0]


def test_channel_unavailable_is_narrowly_retryable_and_auth_quota_are_not():
    from types import SimpleNamespace
    from agent_runtime_kit.agent.providers.codex import _classify_transient_codex_error

    sdk = SimpleNamespace()
    assert _classify_transient_codex_error(sdk, RuntimeError(
        'unexpected status 404 Not Found: no enabled channel for model "gpt-5.6-luna"'
    )) == 'model_channel_unavailable'
    for message in ['unexpected status 404 Not Found: unknown endpoint',
                    'response stream disconnected before completion',
                    'Upstream error, please retry.']:
        assert _classify_transient_codex_error(sdk, RuntimeError(message)) is None
    optimistic_sdk = SimpleNamespace(is_retryable_error=lambda e: True)
    for message in ['Unauthorized', 'Payment required', 'Build usage balance exhausted',
                    'insufficient_quota', 'invalid api key']:
        assert _classify_transient_codex_error(optimistic_sdk, RuntimeError(message)) is None
