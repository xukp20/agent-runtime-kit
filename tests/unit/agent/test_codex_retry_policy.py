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
