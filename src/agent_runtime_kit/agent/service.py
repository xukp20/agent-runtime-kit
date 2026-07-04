from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Mapping

from agent_runtime_kit.runtime import (
    ARKServices,
    AppServices,
    RuntimeContext,
    RuntimePausedError,
    RuntimePauseController,
)

from .homes import HomeCreateSpec, HomeRecord, HomeService, build_provider_env
from .models import (
    Agent,
    AgentAlreadyRunningError,
    AgentClosedError,
    AgentCompletionCheckError,
    AgentCompletionRecord,
    AgentHasNoCompletedTurn,
    AgentIncompleteError,
    AgentPausedError,
    CompletionDecision,
    WaitAgentsResult,
)
from .providers.codex import CodexProvider
from .store import AgentStoreService
from .store_utils import utc_now_iso
from .templates import render_template


class AgentType:
    agent_type: str = ""
    developer_instructions_template: str | None = None
    start_prompt_template: str | None = None
    continue_prompt_template: str | None = None

    def render_developer_instructions(self, variables: dict[str, object]) -> str | None:
        return render_template(self.developer_instructions_template, variables)

    def render_start_prompt(self, variables: dict[str, object]) -> str:
        rendered = render_template(self.start_prompt_template, variables)
        if rendered is None:
            raise ValueError(f"AgentType {self.agent_type} has no start_prompt_template")
        return rendered

    def render_continue_prompt(
        self,
        variables: dict[str, object],
        ctx: "AgentCompletionContext",
        decision: CompletionDecision,
    ) -> str:
        merged = dict(variables)
        if decision.reason is not None:
            merged.setdefault("reason", decision.reason)
        rendered = render_template(self.continue_prompt_template, merged)
        if rendered is None:
            raise ValueError(f"AgentType {self.agent_type} has no continue_prompt_template")
        return rendered

    def check_completion(self, ctx: "AgentCompletionContext") -> CompletionDecision:
        return CompletionDecision(complete=True)

    def max_auto_continue_turns(self, ctx: "AgentCompletionContext | None") -> int:
        return 0


class AgentTypeRegistry:
    def __init__(self) -> None:
        self._types: dict[str, AgentType] = {}

    def register(self, agent_type: AgentType) -> None:
        key = agent_type.agent_type.strip()
        if not key:
            raise ValueError("agent_type must not be empty")
        if key in self._types:
            raise ValueError(f"duplicate agent_type: {key}")
        self._types[key] = agent_type

    def get(self, agent_type: str) -> AgentType:
        key = agent_type.strip()
        try:
            return self._types[key]
        except KeyError as exc:
            raise KeyError(f"unknown agent_type: {key}") from exc

    def list(self) -> list[AgentType]:
        return [self._types[key] for key in sorted(self._types)]


@dataclass(frozen=True)
class AgentCompletionContext(RuntimeContext):
    agent: Agent
    turn_result: object
    auto_continue_count: int
    variables: dict[str, object]


AgentRunPauseController = RuntimePauseController


@dataclass
class _ActiveAgentRun:
    agent_id: str
    worker: threading.Thread
    done_event: threading.Event
    latest_result: object | None = None
    latest_completion: AgentCompletionRecord | None = None
    error: BaseException | None = None


class AgentService:
    def __init__(
        self,
        runtime_root: Path,
        *,
        agent_types: AgentTypeRegistry | None = None,
        store: AgentStoreService | None = None,
        home_service: HomeService | None = None,
        providers: dict[str, object] | None = None,
        ark_services: ARKServices | None = None,
        app_services: AppServices | None = None,
        start_paused: bool = False,
    ) -> None:
        self.runtime_root = Path(runtime_root)
        self.agent_types = agent_types or AgentTypeRegistry()
        self.providers = providers or {"codex": CodexProvider(runtime_root=self.runtime_root)}
        self.store = store or AgentStoreService(self.runtime_root, providers=self.providers)
        self.home_service = home_service or HomeService(self.runtime_root)
        self.ark_services = ark_services or ARKServices()
        self.ark_services.agent_service = self
        self.app_services = app_services or AppServices()
        if self.ark_services.pause_controller is None:
            self.pause_controller = RuntimePauseController(global_paused=start_paused)
            self.ark_services.pause_controller = self.pause_controller
        else:
            self.pause_controller = self.ark_services.pause_controller
            if start_paused:
                self.pause_controller.pause(None)
        self._lock = threading.RLock()
        self._active: dict[str, _ActiveAgentRun] = {}
        self.trace_report_errors: list[dict[str, str]] = []

    def create_agent(
        self,
        scope_id: str,
        agent_type: str,
        cli_type: str = "codex",
        home_id: str | None = None,
    ) -> Agent:
        self.agent_types.get(agent_type)
        resolved_home_id = home_id or agent_type
        home = self.home_service.get_home(cli_type, resolved_home_id)
        if home.status != "active":
            raise RuntimeError(f"home is not active: {cli_type}/{resolved_home_id}")
        return self.store.create_agent_record(
            scope_id=scope_id,
            agent_type=agent_type,
            cli_type=cli_type,
            home_id=resolved_home_id,
        )

    def create_home(
        self,
        spec: HomeCreateSpec,
        *,
        initialize_provider_home: bool = True,
        env: dict[str, str] | None = None,
        workdir: str | None = None,
    ) -> HomeRecord:
        home = self.home_service.create_home(spec)
        if initialize_provider_home:
            self.ensure_provider_home_initialized(
                home.cli_type,
                home.home_id,
                env=env,
                workdir=workdir,
            )
        return home

    def ensure_provider_home_initialized(
        self,
        cli_type: str,
        home_id: str,
        *,
        env: dict[str, str] | None = None,
        workdir: str | None = None,
    ) -> object | None:
        provider = self.providers.get(cli_type)
        if provider is None:
            raise RuntimeError(f"no provider registered for {cli_type}")
        ensure = getattr(provider, "ensure_home_initialized", None)
        if not callable(ensure):
            return None
        home = self.home_service.get_home(cli_type, home_id)
        home_root = self.home_service.resolve_home_root(cli_type, home_id)
        provider_env = build_provider_env(home=home, home_root=home_root, run_env=env)
        return ensure(
            home_id=home_id,
            home_root=home_root,
            env=provider_env,
            workdir=workdir,
        )

    def get_agent(self, agent_id: str) -> Agent:
        return self.store.get_agent(agent_id)

    def list_agents(self, scope_id: str | None = None, status: str | None = None) -> list[Agent]:
        return self.store.list_agents(scope_id=scope_id, status=status)

    def close_agent(self, agent_id: str) -> Agent:
        with self._lock:
            agent = self.store.get_agent(agent_id)
            if agent.status == "running" or agent_id in self._active:
                raise AgentAlreadyRunningError(agent_id)
            return self.store.close_agent(agent_id)

    def start_agent(
        self,
        agent_id: str,
        *,
        variables: dict[str, object] | None = None,
        prompt: str | None = None,
        developer_instructions_template_override: str | None = None,
        start_prompt_template_override: str | None = None,
        continue_prompt_template_override: str | None = None,
        env: dict[str, str] | None = None,
        workdir: str | None = None,
    ) -> Agent:
        variables = dict(variables or {})
        with self._lock:
            agent = self.store.get_agent(agent_id)
            if agent.status == "closed":
                raise AgentClosedError(agent_id)
            if agent.status == "running" or agent_id in self._active:
                raise AgentAlreadyRunningError(agent_id)
            self._assert_agent_can_start(agent.scope_id)
            agent_type = self.agent_types.get(agent.agent_type)
            developer_instructions = _render_developer_instructions(
                agent_type,
                variables,
                developer_instructions_template_override,
            )
            overwrite_developer_instructions = developer_instructions_template_override is not None
            current_prompt = (
                prompt
                if prompt is not None
                else _render_start_prompt(agent_type, variables, start_prompt_template_override)
            )
            self.store.patch_agent(agent_id, status="running")
            done_event = threading.Event()
            active = _ActiveAgentRun(
                agent_id=agent_id,
                worker=threading.Thread(target=lambda: None),
                done_event=done_event,
            )
            worker = threading.Thread(
                target=self._run_agent_worker,
                kwargs={
                    "active": active,
                    "variables": variables,
                    "current_prompt": current_prompt,
                    "developer_instructions": developer_instructions,
                    "overwrite_developer_instructions": overwrite_developer_instructions,
                    "continue_prompt_template_override": continue_prompt_template_override,
                    "env": env,
                    "workdir": workdir,
                },
                daemon=True,
            )
            active.worker = worker
            self._active[agent_id] = active
            worker.start()
            return self.store.get_agent(agent_id)

    def _run_agent_worker(
        self,
        *,
        active: _ActiveAgentRun,
        variables: dict[str, object],
        current_prompt: str,
        developer_instructions: str | None,
        overwrite_developer_instructions: bool,
        continue_prompt_template_override: str | None,
        env: dict[str, str] | None,
        workdir: str | None,
    ) -> None:
        agent_id = active.agent_id
        auto_continue_count = 0
        try:
            while True:
                agent = self.store.get_agent(agent_id)
                agent_type = self.agent_types.get(agent.agent_type)
                home = self.home_service.get_home(agent.cli_type, agent.home_id)
                home_root = self.home_service.resolve_home_root(agent.cli_type, agent.home_id)
                provider_env = build_provider_env(home=home, home_root=home_root, run_env=env)
                provider = self.providers[agent.cli_type]
                if agent.thread_id is None:
                    result = provider.start_thread(
                        home_id=agent.home_id,
                        home_root=home_root,
                        env=provider_env,
                        workdir=workdir,
                        prompt=current_prompt,
                        developer_instructions=developer_instructions,
                        overwrite_developer_instructions=overwrite_developer_instructions,
                        agent_id=agent_id,
                    )
                else:
                    result = provider.resume_thread(
                        home_id=agent.home_id,
                        home_root=home_root,
                        env=provider_env,
                        thread_id=agent.thread_id,
                        workdir=workdir,
                        prompt=current_prompt,
                        developer_instructions=developer_instructions,
                        overwrite_developer_instructions=overwrite_developer_instructions,
                        agent_id=agent_id,
                    )
                active.latest_result = result.turn_result
                self.store.update_thread_locator(
                    agent_id,
                    thread_id=result.thread_id,
                    rollout_relpath=result.rollout_relpath,
                )
                agent = self.store.get_agent(agent_id)
                ctx = AgentCompletionContext(
                    ark=self.ark_services,
                    app=self.app_services,
                    agent=agent,
                    turn_result=result.turn_result,
                    auto_continue_count=auto_continue_count,
                    variables=variables,
                )
                try:
                    decision = agent_type.check_completion(ctx)
                    record = AgentCompletionRecord(
                        turn_id=_turn_id(result.turn_result),
                        decision=decision,
                        status="complete" if decision.complete else "incomplete",
                        auto_continue_count=auto_continue_count,
                        checked_at=utc_now_iso(),
                    )
                except BaseException as exc:
                    record = AgentCompletionRecord(
                        turn_id=_turn_id(result.turn_result),
                        decision=CompletionDecision(complete=False, reason=str(exc)),
                        status="checker_failed",
                        auto_continue_count=auto_continue_count,
                        checked_at=utc_now_iso(),
                        error_message=str(exc),
                    )
                    self.store.update_completion(agent_id, record)
                    self._export_trace_reports_best_effort(agent_id)
                    active.latest_completion = record
                    raise AgentCompletionCheckError(str(exc)) from exc
                self.store.update_completion(agent_id, record)
                self._export_trace_reports_best_effort(agent_id)
                active.latest_completion = record
                if decision.close_agent:
                    self.store.patch_agent(agent_id, status="closed")
                    return
                if decision.complete:
                    return
                max_turns = agent_type.max_auto_continue_turns(ctx)
                if auto_continue_count >= max_turns:
                    raise AgentIncompleteError(agent_id, record)
                auto_continue_count += 1
                current_prompt = decision.continue_prompt or _render_continue_prompt(
                    agent_type,
                    variables,
                    ctx,
                    decision,
                    continue_prompt_template_override,
                )
        except BaseException as exc:
            active.error = exc
        finally:
            with self._lock:
                try:
                    agent = self.store.get_agent(agent_id)
                    if agent.status != "closed":
                        self.store.patch_agent(agent_id, status="idle")
                finally:
                    self._active.pop(agent_id, None)
                    active.done_event.set()

    def wait_agent(self, agent_id: str, timeout_s: float | None = None) -> object:
        active = self._active.get(agent_id)
        if active is not None:
            if not active.done_event.wait(timeout_s):
                raise TimeoutError(agent_id)
            if active.error is not None:
                raise active.error
            if active.latest_result is not None:
                return active.latest_result
        agent = self.store.get_agent(agent_id)
        if agent.last_completion is not None:
            if agent.last_completion.status == "incomplete":
                raise AgentIncompleteError(agent_id, agent.last_completion)
            if agent.last_completion.status == "checker_failed":
                raise AgentCompletionCheckError(agent.last_completion.error_message or agent_id)
        return self.read_latest_turn_result(agent_id)

    def wait_agents(
        self,
        agent_ids: list[str],
        timeout_s: float | None = None,
        fail_fast: bool = False,
    ) -> WaitAgentsResult:
        completed: dict[str, object] = {}
        errors: dict[str, BaseException] = {}
        pending: list[str] = []
        timeout = False
        deadline = None if timeout_s is None else monotonic() + timeout_s
        for index, agent_id in enumerate(agent_ids):
            remaining = None
            if deadline is not None:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    timeout = True
                    pending.extend(agent_ids[index:])
                    break
            try:
                completed[agent_id] = self.wait_agent(agent_id, timeout_s=remaining)
            except TimeoutError:
                timeout = True
                pending.append(agent_id)
                if fail_fast:
                    break
            except BaseException as exc:
                errors[agent_id] = exc
                if fail_fast:
                    break
        return WaitAgentsResult(completed=completed, errors=errors, pending=tuple(pending), timeout=timeout)

    def reconcile_stale_running_agents(self, scope_id: str | None = None) -> list[str]:
        repaired: list[str] = []
        with self._lock:
            for agent in self.store.list_agents(scope_id=scope_id, status="running"):
                if agent.agent_id in self._active:
                    continue
                self.store.patch_agent(agent.agent_id, status="idle")
                repaired.append(agent.agent_id)
        return repaired

    def interrupt_agent(self, agent_id: str) -> bool:
        provider = self.providers.get(self.store.get_agent(agent_id).cli_type)
        if provider is None or not hasattr(provider, "interrupt_agent"):
            return False
        return bool(provider.interrupt_agent(agent_id))

    def pause_runs(self, scope_id: str | None = None) -> None:
        with self._lock:
            self.pause_controller.pause(scope_id)

    def resume_runs(self, scope_id: str | None = None) -> None:
        with self._lock:
            self.pause_controller.resume(scope_id)

    def is_paused(self, scope_id: str | None = None) -> bool:
        return self.pause_controller.is_paused(scope_id)

    def list_running_agents(self, scope_id: str | None = None) -> list[Agent]:
        return self.store.list_agents(scope_id=scope_id, status="running")

    def has_running_agents(self, scope_id: str | None = None) -> bool:
        return bool(self.list_running_agents(scope_id))

    def is_stable(self, scope_id: str | None = None) -> bool:
        return not self.has_running_agents(scope_id)

    def wait_scope_agents(
        self,
        scope_id: str,
        timeout_s: float | None = None,
        fail_fast: bool = False,
    ) -> WaitAgentsResult:
        return self.wait_agents(
            [agent.agent_id for agent in self.list_running_agents(scope_id)],
            timeout_s=timeout_s,
            fail_fast=fail_fast,
        )

    def wait_all_active_agents(
        self,
        timeout_s: float | None = None,
        fail_fast: bool = False,
    ) -> WaitAgentsResult:
        return self.wait_agents(
            [agent.agent_id for agent in self.list_running_agents()],
            timeout_s=timeout_s,
            fail_fast=fail_fast,
        )

    def fork_agent(self, source_agent_id: str, *, target_scope_id: str | None = None) -> Agent:
        with self._lock:
            source = self.store.get_agent(source_agent_id)
            if source.status != "idle":
                raise AgentAlreadyRunningError(source_agent_id)
            if not source.thread_id:
                raise AgentHasNoCompletedTurn(source_agent_id)
            target_scope = target_scope_id or source.scope_id
            self._assert_agent_can_start(source.scope_id)
            self._assert_agent_can_start(target_scope)
        home = self.home_service.get_home(source.cli_type, source.home_id)
        home_root = self.home_service.resolve_home_root(source.cli_type, source.home_id)
        provider_env = build_provider_env(home=home, home_root=home_root)
        forked = self.providers[source.cli_type].fork_thread(
            home_id=source.home_id,
            home_root=home_root,
            env=provider_env,
            thread_id=source.thread_id,
            agent_id=source.agent_id,
        )
        return self.store.create_agent_record(
            scope_id=target_scope,
            agent_type=source.agent_type,
            cli_type=source.cli_type,
            home_id=source.home_id,
            thread_id=forked.thread_id,
            rollout_relpath=forked.rollout_relpath,
            fork_source_agent_id=source.agent_id,
            fork_source_thread_id=source.thread_id,
        )

    def read_thread(self, agent_id: str, include_turns: bool = True) -> object:
        agent = self.store.get_agent(agent_id)
        if not agent.thread_id:
            raise AgentHasNoCompletedTurn(agent_id)
        provider = self.providers.get(agent.cli_type)
        if provider is None:
            raise RuntimeError(f"no provider registered for {agent.cli_type}")
        home = self.home_service.get_home(agent.cli_type, agent.home_id)
        home_root = self.home_service.resolve_home_root(agent.cli_type, agent.home_id)
        provider_env = build_provider_env(home=home, home_root=home_root)
        return provider.read_thread(
            agent,
            home_root=home_root,
            env=provider_env,
            include_turns=include_turns,
        )

    def list_turns(self, agent_id: str) -> list[object]:
        thread = self.read_thread(agent_id, include_turns=True)
        return list(getattr(thread, "turns", []) or [])

    def read_latest_turn_result(self, agent_id: str) -> object:
        agent = self.store.get_agent(agent_id)
        if not agent.thread_id:
            raise AgentHasNoCompletedTurn(agent_id)
        provider = self.providers.get(agent.cli_type)
        if provider is None:
            raise RuntimeError(f"no provider registered for {agent.cli_type}")
        home = self.home_service.get_home(agent.cli_type, agent.home_id)
        home_root = self.home_service.resolve_home_root(agent.cli_type, agent.home_id)
        provider_env = build_provider_env(home=home, home_root=home_root)
        return provider.read_latest_turn_result(agent, home_root=home_root, env=provider_env)

    def read_rollout_events(self, agent_id: str) -> list[dict]:
        return self.store.read_rollout_events(agent_id)

    def trace_reader(self, agent_id: str):
        return self.store.trace_reader(agent_id)

    def get_rollout_info(self, agent_id: str):
        return self.store.get_rollout_info(agent_id)

    def list_trace_turns(self, agent_id: str):
        return self.store.list_trace_turns(agent_id)

    def get_trace_turn(
        self,
        agent_id: str,
        *,
        turn_id: str | None = None,
        index: int | None = None,
        latest: bool = False,
    ):
        return self.store.get_trace_turn(agent_id, turn_id=turn_id, index=index, latest=latest)

    def get_trace_event(
        self,
        agent_id: str,
        *,
        index: int | None = None,
        last: bool = False,
    ):
        return self.store.get_trace_event(agent_id, index=index, last=last)

    def tail_trace_events(
        self,
        agent_id: str,
        *,
        limit: int = 20,
        event_type: str | None = None,
        payload_type: str | None = None,
    ):
        return self.store.tail_trace_events(
            agent_id,
            limit=limit,
            event_type=event_type,
            payload_type=payload_type,
        )

    def list_response_texts(
        self,
        agent_id: str,
        *,
        turn_id: str | None = None,
        latest: bool = False,
    ):
        return self.store.list_response_texts(agent_id, turn_id=turn_id, latest=latest)

    def get_latest_response_text(self, agent_id: str) -> str | None:
        return self.store.get_latest_response_text(agent_id)

    def list_tool_calls(
        self,
        agent_id: str,
        *,
        turn_id: str | None = None,
        latest: bool = False,
    ):
        return self.store.list_tool_calls(agent_id, turn_id=turn_id, latest=latest)

    def get_tool_call(
        self,
        agent_id: str,
        *,
        call_id: str | None = None,
        index: int | None = None,
        last: bool = False,
    ):
        return self.store.get_tool_call(agent_id, call_id=call_id, index=index, last=last)

    def build_trace_report(
        self,
        agent_id: str,
        *,
        artifact_path: str | Path | None = None,
        slow_call_limit: int = 20,
    ):
        return self.store.build_trace_report(
            agent_id,
            artifact_path=artifact_path,
            slow_call_limit=slow_call_limit,
        )

    def export_trace_report(
        self,
        agent_id: str,
        *,
        output_path: str | Path,
        format: str = "json",
        artifact_path: str | Path | None = None,
        slow_call_limit: int = 20,
    ):
        return self.store.export_trace_report(
            agent_id,
            output_path=output_path,
            format=format,
            artifact_path=artifact_path,
            slow_call_limit=slow_call_limit,
        )

    def get_default_trace_report_paths(self, agent_id: str):
        return self.store.get_default_trace_report_paths(agent_id)

    def export_default_trace_reports(
        self,
        agent_id: str,
        *,
        artifact_path: str | Path | None = None,
        slow_call_limit: int = 20,
    ):
        return self.store.export_default_trace_reports(
            agent_id,
            artifact_path=artifact_path,
            slow_call_limit=slow_call_limit,
        )

    def read_default_trace_report(self, agent_id: str, *, format: str = "json") -> object | None:
        return self.store.read_default_trace_report(agent_id, format=format)

    def close(self) -> None:
        for provider in self.providers.values():
            close = getattr(provider, "close", None)
            if callable(close):
                close()

    def _export_trace_reports_best_effort(self, agent_id: str) -> None:
        try:
            self.store.export_default_trace_reports(agent_id)
        except BaseException as exc:  # noqa: BLE001 - report generation must not fail Agent turns.
            self.trace_report_errors.append(
                {
                    "agent_id": agent_id,
                    "error_type": type(exc).__name__,
                    "message": str(exc) or type(exc).__name__,
                }
            )

    def _assert_agent_can_start(self, scope_id: str) -> None:
        try:
            self.pause_controller.assert_can_start(scope_id)
        except RuntimePausedError as exc:
            raise AgentPausedError(f"agent runs are paused for scope: {scope_id}") from exc


def _turn_id(turn_result: object) -> str:
    value = getattr(turn_result, "id", None)
    return str(value or f"turn_{uuid.uuid4().hex}")


def _render_developer_instructions(
    agent_type: AgentType,
    variables: dict[str, object],
    template_override: str | None,
) -> str | None:
    if template_override is not None:
        return render_template(template_override, variables)
    return agent_type.render_developer_instructions(variables)


def _render_start_prompt(
    agent_type: AgentType,
    variables: dict[str, object],
    template_override: str | None,
) -> str:
    if template_override is None:
        return agent_type.render_start_prompt(variables)
    rendered = render_template(template_override, variables)
    if rendered is None:
        raise ValueError(f"AgentType {agent_type.agent_type} has no start prompt template")
    return rendered


def _render_continue_prompt(
    agent_type: AgentType,
    variables: dict[str, object],
    ctx: AgentCompletionContext,
    decision: CompletionDecision,
    template_override: str | None,
) -> str:
    if template_override is None:
        return agent_type.render_continue_prompt(variables, ctx, decision)
    merged = dict(variables)
    if decision.reason is not None:
        merged.setdefault("reason", decision.reason)
    rendered = render_template(template_override, merged)
    if rendered is None:
        raise ValueError(f"AgentType {agent_type.agent_type} has no continue prompt template")
    return rendered
