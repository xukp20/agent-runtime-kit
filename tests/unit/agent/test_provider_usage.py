from __future__ import annotations

import pytest

from agent_runtime_kit.agent.provider_contracts import (
    AgentTurnUsage,
    ModelBackendIdentity,
    ModelRequestUsage,
    ReportedCost,
    TokenUsage,
)


def _model(name: str = "gpt-example") -> ModelBackendIdentity:
    return ModelBackendIdentity(
        api_provider="openai",
        api_mode="responses",
        requested_model=name,
    )


def test_unknown_usage_stays_none_during_aggregation() -> None:
    requests = (
        ModelRequestUsage(0, _model(), TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15)),
        ModelRequestUsage(1, _model(), TokenUsage(input_tokens=None, output_tokens=7, total_tokens=None)),
    )

    usage = AgentTurnUsage.from_requests(requests)

    assert usage.token_usage.input_tokens is None
    assert usage.token_usage.output_tokens == 12
    assert usage.token_usage.total_tokens is None
    assert usage.aggregate_complete is False


def test_cache_and_reasoning_are_not_added_to_total_again() -> None:
    usage = TokenUsage(
        input_tokens=100,
        output_tokens=20,
        total_tokens=120,
        cached_input_tokens=60,
        reasoning_output_tokens=10,
        semantics={
            "cached_input_tokens": "subset_of_input_tokens",
            "reasoning_output_tokens": "subset_of_output_tokens",
        },
    )

    assert usage.total_tokens == 120


def test_reported_cost_uses_decimal_strings_and_never_computes_missing_price() -> None:
    cost = ReportedCost(currency="USD", input_cost="0.01", total_cost=None)
    assert cost.total_cost is None
    with pytest.raises(ValueError, match="decimal string"):
        ReportedCost(currency="USD", total_cost="not-a-number")


def test_usage_provenance_buckets_are_disjoint() -> None:
    with pytest.raises(ValueError, match="simultaneously"):
        ModelRequestUsage(
            request_index=0,
            model_identity=_model(),
            token_usage=TokenUsage(input_tokens=5),
            reported_fields=("input_tokens",),
            estimated_fields=("input_tokens",),
        )
