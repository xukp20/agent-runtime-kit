from __future__ import annotations

from types import MappingProxyType
from typing import Any

from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from .models import (
    BaseFlow,
    BaseFlowError,
    BaseFlowInput,
    BaseFlowResult,
    BaseFlowState,
    BaseStep,
    BaseStepError,
    BaseStepResult,
    BaseStepState,
    BaseSubmission,
    FlowRequest,
    FlowStepTypeError,
    FlowStepValidationError,
)


def _model_field_default(model_cls: type[BaseModel], field_name: str) -> Any:
    field = model_cls.model_fields.get(field_name)
    if field is None or field.default is PydanticUndefined:
        return None
    return field.default


def _state_type_for(state_cls: type[BaseModel]) -> str:
    try:
        state = state_cls()
    except Exception:
        state_type = _model_field_default(state_cls, "state_type")
    else:
        state_type = getattr(state, "state_type", None)
    if not isinstance(state_type, str) or not state_type:
        raise FlowStepValidationError(f"state class {state_cls.__name__} must declare non-empty state_type")
    return state_type


def _type_name_for(model_cls: type[BaseModel], field_name: str) -> str:
    type_name = getattr(model_cls, field_name, None)
    if type_name is None:
        type_name = _model_field_default(model_cls, field_name)
    if not isinstance(type_name, str) or not type_name:
        raise FlowStepValidationError(f"{model_cls.__name__} must declare non-empty {field_name}")
    return type_name


def _declared_model_cls(
    owner_cls: type[BaseModel],
    attr_name: str,
    default_cls: type[BaseModel],
    *,
    expected_base: type[BaseModel],
) -> type[BaseModel]:
    model_cls = getattr(owner_cls, attr_name, default_cls)
    if not isinstance(model_cls, type) or not issubclass(model_cls, expected_base):
        raise FlowStepTypeError(
            f"{owner_cls.__name__}.{attr_name} must be a {expected_base.__name__} subclass"
        )
    return model_cls


def _declared_optional_model_cls(
    owner_cls: type[BaseModel],
    attr_name: str,
    *,
    expected_base: type[BaseModel],
) -> type[BaseModel] | None:
    model_cls = getattr(owner_cls, attr_name, None)
    if model_cls is None:
        return None
    if not isinstance(model_cls, type) or not issubclass(model_cls, expected_base):
        raise FlowStepTypeError(
            f"{owner_cls.__name__}.{attr_name} must be a {expected_base.__name__} subclass or None"
        )
    return model_cls


def _declared_model_map(
    owner_cls: type[BaseModel],
    attr_name: str,
    *,
    expected_base: type[BaseModel],
    type_field: str,
) -> dict[str, type[BaseModel]]:
    raw = getattr(owner_cls, attr_name, {})
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise FlowStepTypeError(f"{owner_cls.__name__}.{attr_name} must be a dict")
    parsed: dict[str, type[BaseModel]] = {}
    for key, model_cls in raw.items():
        if not isinstance(key, str) or not key:
            raise FlowStepTypeError(f"{owner_cls.__name__}.{attr_name} keys must be non-empty strings")
        if not isinstance(model_cls, type) or not issubclass(model_cls, expected_base):
            raise FlowStepTypeError(
                f"{owner_cls.__name__}.{attr_name}[{key!r}] must be a {expected_base.__name__} subclass"
            )
        declared_type = _model_field_default(model_cls, type_field)
        if isinstance(declared_type, str) and declared_type and declared_type != key:
            raise FlowStepTypeError(
                f"{owner_cls.__name__}.{attr_name}[{key!r}] points to {model_cls.__name__} "
                f"with {type_field}={declared_type!r}"
            )
        parsed[key] = model_cls
    return parsed


def _validate_payload(model_cls: type[BaseModel], payload: dict[str, Any], *, label: str) -> BaseModel:
    try:
        return model_cls.model_validate(payload)
    except Exception as exc:
        raise FlowStepValidationError(f"{label} failed validation: {exc}") from exc


def _payload_type(payload: dict[str, Any], type_field: str, *, label: str) -> str:
    type_value = payload.get(type_field)
    if not isinstance(type_value, str) or not type_value:
        raise FlowStepValidationError(f"{label} must declare non-empty {type_field}")
    return type_value


class FlowTypeRegistry:
    def __init__(self) -> None:
        self._types: dict[str, type[BaseFlow]] = {}
        self._state_types: dict[str, str] = {}

    @property
    def types(self) -> MappingProxyType[str, type[BaseFlow]]:
        return MappingProxyType(self._types)

    def register(self, flow_cls: type[BaseFlow]) -> None:
        if not isinstance(flow_cls, type) or not issubclass(flow_cls, BaseFlow):
            raise FlowStepTypeError("flow_cls must be a BaseFlow subclass")
        flow_type = _type_name_for(flow_cls, "flow_type")
        if flow_type in self._types:
            raise FlowStepTypeError(f"flow type already registered: {flow_type}")

        params_cls = getattr(flow_cls, "Params", None)
        if not isinstance(params_cls, type) or not issubclass(params_cls, BaseModel):
            raise FlowStepTypeError(f"flow type {flow_type} must declare Params: type[BaseModel]")
        state_cls = getattr(flow_cls, "State", None)
        if not isinstance(state_cls, type) or not issubclass(state_cls, BaseFlowState):
            raise FlowStepTypeError(f"flow type {flow_type} must declare State: type[BaseFlowState]")
        build_from_request = getattr(flow_cls, "build_from_request", None)
        if not callable(build_from_request):
            raise FlowStepTypeError(f"flow type {flow_type} must implement build_from_request(ctx)")

        self._types[flow_type] = flow_cls
        self._state_types[flow_type] = _state_type_for(state_cls)
        _declared_model_map(flow_cls, "Results", expected_base=BaseFlowResult, type_field="result_type")

    def get(self, flow_type: str) -> type[BaseFlow]:
        try:
            return self._types[flow_type]
        except KeyError as exc:
            raise FlowStepTypeError(f"flow type is not registered: {flow_type}") from exc

    def list(self) -> list[str]:
        return list(self._types)

    def validate_request_params(self, request: FlowRequest) -> BaseModel:
        flow_cls = self.get(request.flow_type)
        params_cls = getattr(flow_cls, "Params")
        try:
            return params_cls.model_validate(request.params)
        except Exception as exc:
            raise FlowStepValidationError(
                f"params for flow type {request.flow_type} failed validation: {exc}"
            ) from exc

    def can_parse_state(self, flow_type: str, state_type: str) -> bool:
        self.get(flow_type)
        return self._state_types[flow_type] == state_type

    def parse_state(self, flow_type: str, payload: dict[str, Any]) -> BaseFlowState:
        flow_cls = self.get(flow_type)
        state_cls = _declared_model_cls(flow_cls, "State", BaseFlowState, expected_base=BaseFlowState)
        return _validate_payload(state_cls, payload, label=f"state for flow type {flow_type}")  # type: ignore[return-value]

    def parse_input(self, flow_type: str, payload: dict[str, Any] | None) -> BaseFlowInput | None:
        if payload is None:
            return None
        flow_cls = self.get(flow_type)
        input_cls = _declared_model_cls(flow_cls, "Input", BaseFlowInput, expected_base=BaseFlowInput)
        return _validate_payload(input_cls, payload, label=f"input for flow type {flow_type}")  # type: ignore[return-value]

    def parse_result(self, flow_type: str, payload: dict[str, Any] | None) -> BaseFlowResult | None:
        if payload is None:
            return None
        flow_cls = self.get(flow_type)
        result_type = _payload_type(payload, "result_type", label=f"result for flow type {flow_type}")
        result_cls = _declared_model_map(
            flow_cls,
            "Results",
            expected_base=BaseFlowResult,
            type_field="result_type",
        ).get(result_type)
        if result_cls is None:
            result_cls = _declared_model_cls(flow_cls, "Result", BaseFlowResult, expected_base=BaseFlowResult)
        return _validate_payload(result_cls, payload, label=f"result for flow type {flow_type}")  # type: ignore[return-value]

    def parse_error(self, flow_type: str, payload: dict[str, Any] | None) -> BaseFlowError | None:
        if payload is None:
            return None
        return _validate_payload(BaseFlowError, payload, label=f"error for flow type {flow_type}")  # type: ignore[return-value]


class StepTypeRegistry:
    def __init__(self) -> None:
        self._types: dict[str, type[BaseStep]] = {}
        self._state_types: dict[str, str] = {}

    @property
    def types(self) -> MappingProxyType[str, type[BaseStep]]:
        return MappingProxyType(self._types)

    def register(self, step_cls: type[BaseStep]) -> None:
        if not isinstance(step_cls, type) or not issubclass(step_cls, BaseStep):
            raise FlowStepTypeError("step_cls must be a BaseStep subclass")
        step_type = _type_name_for(step_cls, "step_type")
        if step_type in self._types:
            raise FlowStepTypeError(f"step type already registered: {step_type}")

        state_cls = getattr(step_cls, "State", None)
        if not isinstance(state_cls, type) or not issubclass(state_cls, BaseStepState):
            raise FlowStepTypeError(f"step type {step_type} must declare State: type[BaseStepState]")

        self._types[step_type] = step_cls
        self._state_types[step_type] = _state_type_for(state_cls)
        _declared_model_map(step_cls, "Results", expected_base=BaseStepResult, type_field="result_type")
        _declared_model_map(step_cls, "Submissions", expected_base=BaseSubmission, type_field="submission_type")

    def get(self, step_type: str) -> type[BaseStep]:
        try:
            return self._types[step_type]
        except KeyError as exc:
            raise FlowStepTypeError(f"step type is not registered: {step_type}") from exc

    def list(self) -> list[str]:
        return list(self._types)

    def can_parse_state(self, step_type: str, state_type: str) -> bool:
        self.get(step_type)
        return self._state_types[step_type] == state_type

    def parse_state(self, step_type: str, payload: dict[str, Any]) -> BaseStepState:
        step_cls = self.get(step_type)
        state_cls = _declared_model_cls(step_cls, "State", BaseStepState, expected_base=BaseStepState)
        return _validate_payload(state_cls, payload, label=f"state for step type {step_type}")  # type: ignore[return-value]

    def parse_result(self, step_type: str, payload: dict[str, Any] | None) -> BaseStepResult | None:
        if payload is None:
            return None
        step_cls = self.get(step_type)
        result_type = _payload_type(payload, "result_type", label=f"result for step type {step_type}")
        result_cls = _declared_model_map(
            step_cls,
            "Results",
            expected_base=BaseStepResult,
            type_field="result_type",
        ).get(result_type)
        if result_cls is None:
            result_cls = _declared_model_cls(step_cls, "Result", BaseStepResult, expected_base=BaseStepResult)
        return _validate_payload(result_cls, payload, label=f"result for step type {step_type}")  # type: ignore[return-value]

    def parse_error(self, step_type: str, payload: dict[str, Any] | None) -> BaseStepError | None:
        if payload is None:
            return None
        return _validate_payload(BaseStepError, payload, label=f"error for step type {step_type}")  # type: ignore[return-value]

    def parse_submission(self, step_type: str, payload: dict[str, Any] | None) -> BaseSubmission | None:
        if payload is None:
            return None
        step_cls = self.get(step_type)
        submission_type = _payload_type(payload, "submission_type", label=f"submission for step type {step_type}")
        submission_cls = _declared_model_map(
            step_cls,
            "Submissions",
            expected_base=BaseSubmission,
            type_field="submission_type",
        ).get(submission_type)
        if submission_cls is None:
            submission_cls = _declared_optional_model_cls(
                step_cls,
                "Submission",
                expected_base=BaseSubmission,
            )
        if submission_cls is None:
            raise FlowStepTypeError(
                f"step type {step_type} does not declare submission type {submission_type!r}"
            )
        return _validate_payload(submission_cls, payload, label=f"submission for step type {step_type}")  # type: ignore[return-value]
