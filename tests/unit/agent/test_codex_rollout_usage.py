from agent_runtime_kit.agent.providers.codex_trace import AgentTraceReader
from agent_runtime_kit.agent.providers.codex_query import _turn_usage
from agent_runtime_kit.agent.provider_contracts import TokenUsage


def context(turn):
    return {'type': 'turn_context', 'payload': {'turn_id': turn}}


def count(n, output=0):
    return {'type': 'event_msg', 'payload': {'type': 'token_count', 'info': {
        'total_token_usage': {'input_tokens': n, 'output_tokens': output,
                             'total_tokens': n + output, 'cached_input_tokens': 0,
                             'reasoning_output_tokens': 0, 'cache_write_input_tokens': 0}}}}


def test_cumulative_counts_are_differenced_per_turn_without_replay_double_count():
    events = [context('a'), count(10), count(10), count(20, 5),
              context('b'), count(20, 5), count(30, 7)]
    turns = AgentTraceReader(events=events).list_turns()
    usages = [_turn_usage(t.usage, model_identity=None) for t in turns]
    assert [u.token_usage.total_tokens for u in usages] == [25, 12]
    assert TokenUsage.aggregate_complete(tuple(u.token_usage for u in usages)).total_tokens == 37
    assert all(u.aggregate_complete for u in usages)
    assert usages[0].token_usage.cached_input_tokens == 0
    assert usages[0].token_usage.cache_write_input_tokens == 0


def test_missing_info_keeps_latest_known_count_and_missing_turn_stays_unknown():
    turns = AgentTraceReader(events=[context('a'), count(10),
        {'type': 'event_msg', 'payload': {'type': 'token_count', 'info': None}},
        context('b')]).list_turns()
    assert _turn_usage(turns[0].usage, model_identity=None).token_usage.total_tokens == 10
    assert not _turn_usage(turns[1].usage, model_identity=None).aggregate_complete


def test_counter_reset_does_not_emit_negative_or_claim_complete_usage():
    turns = AgentTraceReader(events=[context('a'), count(20, 5), context('b'), count(2), count(50, 8)]).list_turns()
    usage = _turn_usage(turns[1].usage, model_identity=None)
    assert usage.token_usage.total_tokens is None
    assert not usage.aggregate_complete


def test_missing_previous_turn_counter_does_not_charge_both_turns_to_next():
    turns = AgentTraceReader(events=[context('a'), context('b'), count(50, 3)]).list_turns()
    assert not _turn_usage(turns[1].usage, model_identity=None).aggregate_complete
