from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Mapping, Protocol

from .identities import ModelBackendIdentity, ProviderPayload


_TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "uncached_input_tokens",
    "cached_input_tokens",
    "cache_read_input_tokens",
    "cache_write_input_tokens",
    "cache_creation_input_tokens",
    "cache_creation_5m_input_tokens",
    "cache_creation_1h_input_tokens",
    "reasoning_output_tokens",
    "visible_output_tokens",
)


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    uncached_input_tokens: int | None = None
    cached_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    cache_write_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    cache_creation_5m_input_tokens: int | None = None
    cache_creation_1h_input_tokens: int | None = None
    reasoning_output_tokens: int | None = None
    visible_output_tokens: int | None = None
    input_tokens_by_modality: Mapping[str, int] = field(default_factory=dict)
    output_tokens_by_modality: Mapping[str, int] = field(default_factory=dict)
    other_token_details: Mapping[str, int] = field(default_factory=dict)
    semantics: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in _TOKEN_FIELDS:
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must not be negative")
        for mapping_name in (
            "input_tokens_by_modality",
            "output_tokens_by_modality",
            "other_token_details",
        ):
            if any(value < 0 for value in getattr(self, mapping_name).values()):
                raise ValueError(f"{mapping_name} values must not be negative")

    @classmethod
    def aggregate_complete(cls, usages: tuple["TokenUsage", ...]) -> "TokenUsage":
        if not usages:
            return cls()
        values: dict[str, int | None] = {}
        for name in _TOKEN_FIELDS:
            items = [getattr(usage, name) for usage in usages]
            values[name] = sum(items) if all(item is not None for item in items) else None
        return cls(**values)


@dataclass(frozen=True)
class ReportedCost:
    currency: str
    input_cost: str | None = None
    output_cost: str | None = None
    cache_read_cost: str | None = None
    cache_write_cost: str | None = None
    reasoning_cost: str | None = None
    server_tool_cost: str | None = None
    total_cost: str | None = None
    source: str = "provider_reported"
    provider_payload: ProviderPayload | None = None

    def __post_init__(self) -> None:
        if not self.currency.strip():
            raise ValueError("currency must not be empty")
        if self.source != "provider_reported":
            raise ValueError("ReportedCost source must be provider_reported")
        for name in (
            "input_cost",
            "output_cost",
            "cache_read_cost",
            "cache_write_cost",
            "reasoning_cost",
            "server_tool_cost",
            "total_cost",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            try:
                parsed = Decimal(value)
            except (InvalidOperation, ValueError) as exc:
                raise ValueError(f"{name} must be a decimal string") from exc
            if parsed < 0:
                raise ValueError(f"{name} must not be negative")


@dataclass(frozen=True)
class BillableUnit:
    category: str
    unit: str
    quantity: str
    provider_sku: str | None = None
    reported_cost: ReportedCost | None = None
    provider_payload: ProviderPayload | None = None

    def __post_init__(self) -> None:
        if not self.category.strip() or not self.unit.strip():
            raise ValueError("billable category and unit must not be empty")
        try:
            if Decimal(self.quantity) < 0:
                raise ValueError("billable quantity must not be negative")
        except InvalidOperation as exc:
            raise ValueError("billable quantity must be a decimal string") from exc


@dataclass(frozen=True)
class ModelRequestUsage:
    request_index: int
    model_identity: ModelBackendIdentity
    token_usage: TokenUsage
    request_id: str | None = None
    response_id: str | None = None
    session_id: str | None = None
    turn_id: str | None = None
    reported_cost: ReportedCost | None = None
    billable_units: tuple[BillableUnit, ...] = ()
    status: str | None = None
    stop_reason: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    duration_ms: float | None = None
    time_to_first_token_ms: float | None = None
    reported_fields: tuple[str, ...] = ()
    derived_fields: tuple[str, ...] = ()
    estimated_fields: tuple[str, ...] = ()
    unavailable_fields: Mapping[str, str] = field(default_factory=dict)
    provider_payload: ProviderPayload | None = None

    def __post_init__(self) -> None:
        if self.request_index < 0:
            raise ValueError("request_index must not be negative")
        for name in ("duration_ms", "time_to_first_token_ms"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must not be negative")
        buckets = [set(self.reported_fields), set(self.derived_fields), set(self.estimated_fields)]
        if buckets[0] & buckets[1] or buckets[0] & buckets[2] or buckets[1] & buckets[2]:
            raise ValueError("usage fields cannot be reported, derived, and estimated simultaneously")


@dataclass(frozen=True)
class AgentTurnUsage:
    request_count: int
    requests: tuple[ModelRequestUsage, ...]
    token_usage: TokenUsage
    models_used: tuple[ModelBackendIdentity, ...] = ()
    billable_units: tuple[BillableUnit, ...] = ()
    reported_costs: tuple[ReportedCost, ...] = ()
    aggregate_complete: bool = False

    @classmethod
    def from_requests(cls, requests: tuple[ModelRequestUsage, ...]) -> "AgentTurnUsage":
        models: list[ModelBackendIdentity] = []
        for request in requests:
            if request.model_identity not in models:
                models.append(request.model_identity)
        aggregate = TokenUsage.aggregate_complete(tuple(item.token_usage for item in requests))
        complete = bool(requests) and all(
            getattr(aggregate, field_name) is not None
            for field_name in ("input_tokens", "output_tokens", "total_tokens")
        )
        return cls(
            request_count=len(requests),
            requests=requests,
            token_usage=aggregate,
            models_used=tuple(models),
            billable_units=tuple(unit for request in requests for unit in request.billable_units),
            reported_costs=tuple(
                request.reported_cost for request in requests if request.reported_cost is not None
            ),
            aggregate_complete=complete,
        )


@dataclass(frozen=True)
class AgentSessionUsage:
    turn_count: int | None
    request_count: int | None
    token_usage: TokenUsage
    turns: tuple[AgentTurnUsage, ...] = ()
    latest_context_usage: object | None = None
    aggregate_complete: bool = False


@dataclass(frozen=True)
class TokenEstimateRequest:
    model_identity: ModelBackendIdentity
    content: object
    tool_schema: object | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TokenEstimate:
    tokens: int
    estimator_id: str
    estimator_version: str | None
    tokenizer_id: str | None
    model_identity: ModelBackendIdentity
    content_fingerprint: str | None = None
    includes_tool_schema: bool = False
    includes_provider_overhead: bool = False
    confidence: str = "unknown"
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.tokens < 0:
            raise ValueError("estimated tokens must not be negative")


class TokenEstimator(Protocol):
    estimator_id: str

    def supports(self, model: ModelBackendIdentity) -> bool: ...

    def estimate(self, request: TokenEstimateRequest) -> TokenEstimate: ...


class PricingResolver(Protocol):
    def estimate_cost(self, usage: ModelRequestUsage) -> object | None: ...
