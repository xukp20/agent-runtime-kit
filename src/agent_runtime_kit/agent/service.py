from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from time import monotonic

from agent_runtime_kit.runtime import (
    ARKServices,
    AppServices,
    RuntimeContext,
    RuntimePausedError,
    RuntimePauseController,
)

from .context import (
    AgentContextCompactionResult,
    AgentContextCompactionStatus,
    AgentContextMaintenanceJournal,
    AgentContextMaintenanceJournalStatus,
    AgentContextMaintenancePolicy,
    AgentContextUsage,
    ProviderContextCompactionResult,
    ProviderContextUsage,
)
from .homes import HomeCreateSpec, HomeRecord, HomeService, build_provider_env
from .models import (
    Agent,
    AgentAlreadyRunningError,
    AgentClosedError,
    AgentCompletionCheckError,
    AgentCompletionRecord,
    AgentForkInfo,
    AgentContextCompactionRequestUnknown,
    AgentContextMaintenanceBlocked,
    AgentContextMaintenanceUnsupported,
    AgentHasNoCompletedTurn,
    AgentIncompleteError,
    AgentPausedError,
    AgentStatusWaitResult,
    CompletionDecision,
    WaitAgentsResult,
)
from .provider_contracts import (
    AgentContextUsage as StandardAgentContextUsage,
    AgentEvent,
    AgentProviderBundle,
    AgentTurnResult,
    CapabilityKey,
    ModelBackendIdentity,
    Page,
    ProviderCapabilityUnavailable,
    ProviderContextCompactionRequest,
    ProviderContextCompactionResult as StandardProviderContextCompactionResult,
    ProviderContextQuery,
    ProviderContextReconcileRequest,
    ProviderContextUsage as StandardProviderContextUsage,
    ProviderEventQuery,
    ProviderForkRequest,
    ProviderCapabilities,
    ProviderRegistry,
    ProviderRunHandle,
    ProviderRunRequest,
    ProviderSessionListQuery,
    ProviderSessionLocator,
    ProviderToolQuery,
    ProviderTurnLocator,
    ProviderTurnQuery,
    ProviderTurnResult,
    ProviderUsageQuery,
)
from .providers.codex import CodexProvider
from .report_policy import AgentTraceReportPolicy, TraceReportPersistence
from .store import AgentStoreService
from .store_utils import utc_now_iso
from .templates import render_template


@dataclass(frozen=True)
class RunningAgentAuditRecord:
    agent_id: str
    scope_id: str
    classification: str
    thread_id: str | None
    rollout_relpath: str | None
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class RunningAgentRepairResult:
    agent_id: str
    classification: str
    action: str
    dry_run: bool
    repaired: bool


class AgentType:
    agent_type: str = ""
    provider_type: str = "codex"
    default_home_id: str | None = None
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
    standard_turn_result: AgentTurnResult | None = None


AgentRunPauseController = RuntimePauseController


@dataclass
class _ActiveAgentRun:
    agent_id: str
    worker: threading.Thread
    done_event: threading.Event
    handle_ready: threading.Event = field(default_factory=threading.Event)
    provider_handle: ProviderRunHandle | None = None
    latest_result: object | None = None
    latest_standard_result: AgentTurnResult | None = None
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
        provider_registry: ProviderRegistry | None = None,
        ark_services: ARKServices | None = None,
        app_services: AppServices | None = None,
        start_paused: bool = False,
        trace_report_policy: AgentTraceReportPolicy | None = None,
    ) -> None:
        self.runtime_root = Path(runtime_root)
        self.agent_types = agent_types or AgentTypeRegistry()
        self.providers = providers or {"codex": CodexProvider(runtime_root=self.runtime_root)}
        self.provider_registry = provider_registry or ProviderRegistry()
        self._provider_bundle_sources: dict[str, object] = {}
        for provider_type, provider in self.providers.items():
            if provider_type in self.provider_registry:
                continue
            bundle = _build_provider_bundle(provider, runtime_root=self.runtime_root)
            if bundle is not None:
                if bundle.provider_type != provider_type:
                    raise ValueError(
                        f"provider bundle key mismatch: {provider_type} != {bundle.provider_type}"
                    )
                self.provider_registry.register(bundle)
                self._provider_bundle_sources[provider_type] = provider
        self.store = store or AgentStoreService(self.runtime_root, providers=self.providers)
        if home_service is None:
            renderers = {
                bundle.provider_type: bundle.home_renderer for bundle in self.provider_registry.list()
            }
            self.home_service = HomeService(self.runtime_root, renderers=renderers or None)
        else:
            self.home_service = home_service
        self.ark_services = ark_services or ARKServices()
        self.ark_services.agent_service = self
        self.app_services = app_services or AppServices()
        self.trace_report_policy = trace_report_policy or AgentTraceReportPolicy()
        if self.ark_services.pause_controller is None:
            self.pause_controller = RuntimePauseController(global_paused=start_paused)
            self.ark_services.pause_controller = self.pause_controller
        else:
            self.pause_controller = self.ark_services.pause_controller
            if start_paused:
                self.pause_controller.pause(None)
        self._lock = threading.RLock()
        self._status_condition = threading.Condition(self._lock)
        self._active: dict[str, _ActiveAgentRun] = {}
        self._latest_standard_results: dict[str, AgentTurnResult] = {}
        self.trace_report_errors: list[dict[str, str]] = []

    def create_agent(
        self,
        scope_id: str,
        agent_type: str,
        cli_type: str | None = None,
        home_id: str | None = None,
    ) -> Agent:
        agent_type_spec = self.agent_types.get(agent_type)
        resolved_cli_type = cli_type or agent_type_spec.provider_type
        resolved_home_id = home_id or agent_type_spec.default_home_id or agent_type
        home = self.home_service.get_home(resolved_cli_type, resolved_home_id)
        if home.status != "active":
            raise RuntimeError(f"home is not active: {resolved_cli_type}/{resolved_home_id}")
        return self.store.create_agent_record(
            scope_id=scope_id,
            agent_type=agent_type,
            cli_type=resolved_cli_type,
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
        bundle = self._provider_bundle(cli_type)
        if bundle is not None:
            home = self.home_service.get_home(cli_type, home_id)
            context = self.home_service.build_execution_context(
                cli_type,
                home_id,
                run_env=env,
                workdir=workdir,
            )
            result = bundle.home_renderer.initialize(home, context)
            if result.materialization_changed:
                self.home_service.seal_home_materialization(cli_type, home_id)
            return result
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
        with self._status_condition:
            agent = self.store.get_agent(agent_id)
            if agent.status == "running" or agent_id in self._active:
                raise AgentAlreadyRunningError(agent_id)
            closed = self.store.close_agent(agent_id)
            self._status_condition.notify_all()
            return closed

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
        context_maintenance_policy: AgentContextMaintenancePolicy | None = None,
    ) -> Agent:
        variables = dict(variables or {})
        with self._status_condition:
            agent = self.store.get_agent(agent_id)
            if agent.status == "closed":
                raise AgentClosedError(agent_id)
            if agent.status == "running" or agent_id in self._active:
                raise AgentAlreadyRunningError(agent_id)
            self._assert_context_maintenance_resolved(agent_id)
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
                    "context_maintenance_policy": context_maintenance_policy,
                },
                daemon=True,
            )
            active.worker = worker
            self._active[agent_id] = active
            self._status_condition.notify_all()
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
        context_maintenance_policy: AgentContextMaintenancePolicy | None,
    ) -> None:
        agent_id = active.agent_id
        auto_continue_count = 0
        try:
            if context_maintenance_policy is not None and context_maintenance_policy.enabled:
                self._compact_agent_context(
                    agent_id,
                    threshold=context_maintenance_policy.threshold,
                    timeout_s=context_maintenance_policy.timeout_s,
                    trigger="threshold_preflight",
                    force=False,
                    env=env,
                    workdir=workdir,
                )
            while True:
                agent = self.store.get_agent(agent_id)
                agent_type = self.agent_types.get(agent.agent_type)
                turn_result, standard_result, thread_id, rollout_relpath = self._execute_provider_turn(
                    active=active,
                    agent=agent,
                    prompt=current_prompt,
                    developer_instructions=developer_instructions,
                    overwrite_developer_instructions=overwrite_developer_instructions,
                    env=env,
                    workdir=workdir,
                )
                active.latest_result = turn_result
                active.latest_standard_result = standard_result
                if standard_result is not None:
                    self._latest_standard_results[agent_id] = standard_result
                self.store.update_thread_locator(
                    agent_id,
                    thread_id=thread_id,
                    rollout_relpath=rollout_relpath,
                    session_locator=(
                        standard_result.session_locator if standard_result is not None else None
                    ),
                    latest_turn_locator=(
                        standard_result.turn_locator if standard_result is not None else None
                    ),
                    artifact_locator=(
                        standard_result.provider_result.artifact_locator
                        if standard_result is not None
                        else None
                    ),
                )
                agent = self.store.get_agent(agent_id)
                ctx = AgentCompletionContext(
                    ark=self.ark_services,
                    app=self.app_services,
                    agent=agent,
                    turn_result=turn_result,
                    auto_continue_count=auto_continue_count,
                    variables=variables,
                    standard_turn_result=standard_result,
                )
                try:
                    decision = agent_type.check_completion(ctx)
                    record = AgentCompletionRecord(
                        turn_id=_standard_or_legacy_turn_id(standard_result, turn_result),
                        decision=decision,
                        status="complete" if decision.complete else "incomplete",
                        auto_continue_count=auto_continue_count,
                        checked_at=utc_now_iso(),
                    )
                except BaseException as exc:
                    record = AgentCompletionRecord(
                        turn_id=_standard_or_legacy_turn_id(standard_result, turn_result),
                        decision=CompletionDecision(complete=False, reason=str(exc)),
                        status="checker_failed",
                        auto_continue_count=auto_continue_count,
                        checked_at=utc_now_iso(),
                        error_message=str(exc),
                    )
                    self.store.update_completion(agent_id, record)
                    if standard_result is not None:
                        standard_result = replace(standard_result, completion=record)
                        active.latest_standard_result = standard_result
                        self._latest_standard_results[agent_id] = standard_result
                    self._export_trace_reports_best_effort(agent_id)
                    active.latest_completion = record
                    raise AgentCompletionCheckError(str(exc)) from exc
                self.store.update_completion(agent_id, record)
                if standard_result is not None:
                    standard_result = replace(standard_result, completion=record)
                    active.latest_standard_result = standard_result
                    self._latest_standard_results[agent_id] = standard_result
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
            with self._status_condition:
                try:
                    agent = self.store.get_agent(agent_id)
                    if agent.status != "closed":
                        self.store.patch_agent(agent_id, status="idle")
                finally:
                    active.handle_ready.set()
                    self._active.pop(agent_id, None)
                    active.done_event.set()
                    self._status_condition.notify_all()

    def _execute_provider_turn(
        self,
        *,
        active: _ActiveAgentRun,
        agent: Agent,
        prompt: str,
        developer_instructions: str | None,
        overwrite_developer_instructions: bool,
        env: dict[str, str] | None,
        workdir: str | None,
    ) -> tuple[object, AgentTurnResult | None, str, str | None]:
        bundle = self._provider_bundle(agent.cli_type)
        if bundle is None:
            return self._execute_legacy_provider_turn(
                agent=agent,
                prompt=prompt,
                developer_instructions=developer_instructions,
                overwrite_developer_instructions=overwrite_developer_instructions,
                env=env,
                workdir=workdir,
            )
        if agent.cli_type == "codex":
            self.ensure_provider_home_initialized(
                agent.cli_type,
                agent.home_id,
                env=env,
                workdir=workdir,
            )
        execution_context = self.home_service.build_execution_context(
            agent.cli_type,
            agent.home_id,
            run_env=env,
            workdir=workdir,
        )
        session_locator = (
            self._provider_session_locator(agent) if agent.thread_id is not None else None
        )
        request = ProviderRunRequest(
            agent_id=agent.agent_id,
            scope_id=agent.scope_id,
            agent_type=agent.agent_type,
            provider_type=agent.cli_type,
            home_id=agent.home_id,
            session_locator=session_locator,
            prompt=prompt,
            developer_instructions=developer_instructions,
            replace_developer_instructions=overwrite_developer_instructions,
            workdir=workdir,
            environment=execution_context.process_environment,
            model_overrides=execution_context.resolved_defaults,
            metadata={"agent_created_at": agent.created_at},
            event_sink=lambda event: self._on_provider_event(agent.agent_id, event),
            execution_context=execution_context,
        )
        active.handle_ready.clear()
        handle = bundle.runtime.resume(request) if session_locator is not None else bundle.runtime.start(request)
        active.provider_handle = handle
        active.handle_ready.set()
        locator = handle.session_locator()
        if locator is not None:
            self.store.update_thread_locator(
                agent.agent_id,
                thread_id=locator.session_id,
                rollout_relpath=agent.rollout_relpath,
                session_locator=locator,
            )
        provider_result = handle.wait_terminal()
        standard_result = AgentTurnResult(
            agent_id=agent.agent_id,
            scope_id=agent.scope_id,
            agent_type=agent.agent_type,
            home_id=agent.home_id,
            provider_result=provider_result,
        )
        compatibility_result: object = standard_result
        if bundle.compatibility is not None:
            compatibility_result = bundle.compatibility.completion_turn_result(handle, provider_result)
        rollout_relpath = agent.rollout_relpath
        if provider_result.artifact_locator is not None:
            rollout_relpath = provider_result.artifact_locator.native_primary_ref or rollout_relpath
        return (
            compatibility_result,
            standard_result,
            provider_result.session_locator.session_id,
            rollout_relpath,
        )

    def _execute_legacy_provider_turn(
        self,
        *,
        agent: Agent,
        prompt: str,
        developer_instructions: str | None,
        overwrite_developer_instructions: bool,
        env: dict[str, str] | None,
        workdir: str | None,
    ) -> tuple[object, None, str, str | None]:
        # COMPAT(legacy-provider-runtime): preserves injected providers that
        # implement start_thread/resume_thread instead of ProviderRuntimeAdapter.
        # Remove after LC runtime-matrix fakes register AgentProviderBundle.
        home = self.home_service.get_home(agent.cli_type, agent.home_id)
        home_root = self.home_service.resolve_home_root(agent.cli_type, agent.home_id)
        provider_env = build_provider_env(home=home, home_root=home_root, run_env=env)
        provider = self.providers[agent.cli_type]
        common = {
            "home_id": agent.home_id,
            "home_root": home_root,
            "env": provider_env,
            "workdir": workdir,
            "prompt": prompt,
            "developer_instructions": developer_instructions,
            "overwrite_developer_instructions": overwrite_developer_instructions,
            "agent_id": agent.agent_id,
        }
        if agent.thread_id is None:
            result = provider.start_thread(**common)
        else:
            result = provider.resume_thread(**common, thread_id=agent.thread_id)
        return result.turn_result, None, result.thread_id, result.rollout_relpath

    def _provider_bundle(self, provider_type: str) -> AgentProviderBundle | None:
        current_provider = self.providers.get(provider_type)
        source = self._provider_bundle_sources.get(provider_type)
        if provider_type in self.provider_registry and (source is None or source is current_provider):
            return self.provider_registry.get(provider_type)
        if source is not None and source is not current_provider:
            # COMPAT(mutable-providers-dict): LC tests replace providers directly.
            # Rebuild a bundle when supported, otherwise deliberately fall back
            # to the legacy provider path for that replacement.
            replacement = _build_provider_bundle(current_provider, runtime_root=self.runtime_root)
            if replacement is None:
                self.provider_registry.unregister(provider_type)
                self._provider_bundle_sources.pop(provider_type, None)
                return None
            self.provider_registry.replace(replacement)
            self._provider_bundle_sources[provider_type] = current_provider
            return replacement
        if provider_type not in self.provider_registry:
            replacement = _build_provider_bundle(current_provider, runtime_root=self.runtime_root)
            if replacement is not None:
                self.provider_registry.register(replacement)
                self._provider_bundle_sources[provider_type] = current_provider
                return replacement
        return None

    def get_provider_bundle(self, provider_type: str) -> AgentProviderBundle | None:
        """Return the currently effective provider bundle for runtime services."""

        return self._provider_bundle(provider_type)

    def _on_provider_event(self, agent_id: str, event: AgentEvent) -> None:
        if event.session_id is None:
            return
        try:
            agent = self.store.get_agent(agent_id)
        except KeyError:
            return
        if agent.thread_id == event.session_id:
            return
        self.store.update_thread_locator(
            agent_id,
            thread_id=event.session_id,
            rollout_relpath=agent.rollout_relpath,
        )

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

    def wait_agent_status_change(
        self,
        agent_id: str,
        *,
        after_status: str,
        timeout_s: float | None = None,
    ) -> AgentStatusWaitResult:
        if timeout_s is not None and timeout_s < 0:
            raise ValueError("timeout_s must be non-negative")
        deadline = None if timeout_s is None else monotonic() + timeout_s
        with self._status_condition:
            while True:
                agent = self.store.get_agent(agent_id)
                if agent.status != after_status:
                    return AgentStatusWaitResult(
                        agent=agent,
                        changed=True,
                        timed_out=False,
                        observed_at=utc_now_iso(),
                    )
                remaining = None if deadline is None else deadline - monotonic()
                if remaining is not None and remaining <= 0:
                    return AgentStatusWaitResult(
                        agent=agent,
                        changed=False,
                        timed_out=True,
                        observed_at=utc_now_iso(),
                    )
                self._status_condition.wait(remaining)

    def wait_agent_result(
        self,
        agent_id: str,
        timeout_s: float | None = None,
    ) -> AgentTurnResult:
        """Wait for and return the Provider-neutral result for the latest turn."""

        active = self._active.get(agent_id)
        if active is not None:
            if not active.done_event.wait(timeout_s):
                raise TimeoutError(agent_id)
            if active.error is not None:
                raise active.error
        result = self._latest_standard_results.get(agent_id)
        if result is None:
            raise AgentHasNoCompletedTurn(agent_id)
        return result

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

    def inspect_agent_context(
        self,
        agent_id: str,
        *,
        env: dict[str, str] | None = None,
        workdir: str | None = None,
    ) -> AgentContextUsage:
        agent = self.store.get_agent(agent_id)
        if not agent.thread_id:
            return self._unavailable_context_usage(agent, "no_session")
        bundle = self._provider_bundle(agent.cli_type)
        if bundle is not None and bundle.context is not None:
            usage = self.inspect_agent_context_result(agent_id, env=env, workdir=workdir)
            return _legacy_agent_context_usage(usage)
        # COMPAT(legacy-provider-context-methods): injected providers may expose
        # inspect_thread_context without a bundle. Remove when they register a
        # ProviderContextAdapter; covered by fake-provider context tests.
        provider = self.providers.get(agent.cli_type)
        inspect = getattr(provider, "inspect_thread_context", None)
        if not callable(inspect):
            return self._unavailable_context_usage(agent, "provider_unsupported")
        home = self.home_service.get_home(agent.cli_type, agent.home_id)
        home_root = self.home_service.resolve_home_root(agent.cli_type, agent.home_id)
        provider_env = build_provider_env(home=home, home_root=home_root, run_env=env)
        usage = inspect(
            home_id=agent.home_id,
            home_root=home_root,
            env=provider_env,
            thread_id=agent.thread_id,
            workdir=workdir,
            agent_id=agent.agent_id,
        )
        if not isinstance(usage, ProviderContextUsage):
            raise TypeError(f"provider returned invalid context usage: {agent.cli_type}")
        return usage.for_agent(agent_id=agent.agent_id, provider_type=agent.cli_type)

    def inspect_agent_context_result(
        self,
        agent_id: str,
        *,
        env: dict[str, str] | None = None,
        workdir: str | None = None,
    ) -> StandardAgentContextUsage:
        """Inspect context through the provider-neutral Context adapter."""

        agent = self.store.get_agent(agent_id)
        if not agent.thread_id:
            return _standard_unavailable_context_usage(agent, "no_session")
        bundle = self._provider_bundle(agent.cli_type)
        if bundle is None or bundle.context is None:
            return _standard_unavailable_context_usage(agent, "provider_unsupported")
        usage = bundle.context.inspect(
            ProviderContextQuery(
                session=self._provider_session_locator(agent),
                agent_id=agent.agent_id,
                execution_context=self.home_service.build_execution_context(
                    agent.cli_type,
                    agent.home_id,
                    run_env=env,
                    workdir=workdir,
                ),
            )
        )
        if not isinstance(usage, StandardProviderContextUsage):
            raise TypeError(f"provider returned invalid standard context usage: {agent.cli_type}")
        return usage.for_agent(agent_id=agent.agent_id, provider_type=agent.cli_type)

    def compact_agent(
        self,
        agent_id: str,
        *,
        timeout_s: float = 120.0,
        env: dict[str, str] | None = None,
        workdir: str | None = None,
    ) -> AgentContextCompactionResult:
        return self._run_manual_context_maintenance(
            agent_id,
            threshold=None,
            timeout_s=timeout_s,
            trigger="manual",
            force=True,
            env=env,
            workdir=workdir,
        )

    def compact_agent_if_needed(
        self,
        agent_id: str,
        *,
        threshold: float = 0.80,
        timeout_s: float = 120.0,
        env: dict[str, str] | None = None,
        workdir: str | None = None,
    ) -> AgentContextCompactionResult:
        policy = AgentContextMaintenancePolicy(threshold=threshold, timeout_s=timeout_s)
        return self._run_manual_context_maintenance(
            agent_id,
            threshold=policy.threshold,
            timeout_s=policy.timeout_s,
            trigger="threshold_manual",
            force=False,
            env=env,
            workdir=workdir,
        )

    def _run_manual_context_maintenance(
        self,
        agent_id: str,
        *,
        threshold: float | None,
        timeout_s: float,
        trigger: str,
        force: bool,
        env: dict[str, str] | None,
        workdir: str | None,
    ) -> AgentContextCompactionResult:
        active = self._begin_synchronous_maintenance(agent_id)
        try:
            result = self._compact_agent_context(
                agent_id,
                threshold=threshold,
                timeout_s=timeout_s,
                trigger=trigger,
                force=force,
                env=env,
                workdir=workdir,
            )
            active.latest_result = result
            return result
        except BaseException as exc:
            active.error = exc
            raise
        finally:
            self._finish_synchronous_maintenance(active)

    def reconcile_agent_context_maintenance(
        self,
        agent_id: str,
        *,
        env: dict[str, str] | None = None,
        workdir: str | None = None,
    ) -> AgentContextMaintenanceJournal | None:
        journal = self.store.read_context_maintenance(agent_id)
        if journal is None or not journal.unresolved:
            return journal
        if not journal.session_id or not journal.baseline:
            raise AgentContextMaintenanceBlocked(
                f"context maintenance cannot be reconciled without session baseline: {agent_id}"
            )
        active = self._begin_synchronous_maintenance(
            agent_id,
            require_resolved=False,
            respect_pause=False,
        )
        try:
            agent = self.store.get_agent(agent_id)
            bundle = self._provider_bundle(agent.cli_type)
            context_adapter = bundle.context if bundle is not None else None
            # COMPAT(legacy-provider-context-reconcile): retain the old method
            # only for injected providers without a ProviderContextAdapter.
            provider = self.providers.get(agent.cli_type)
            reconcile = getattr(provider, "reconcile_thread_compaction", None)
            if context_adapter is None and not callable(reconcile):
                raise AgentContextMaintenanceBlocked(
                    f"provider cannot reconcile context maintenance: {agent.cli_type}"
                )
            if context_adapter is not None:
                standard_result = context_adapter.reconcile(
                    ProviderContextReconcileRequest(
                        session=self._provider_session_locator(agent),
                        operation_id=journal.provider_operation_id,
                        baseline=journal.baseline,
                        agent_id=agent.agent_id,
                        execution_context=self.home_service.build_execution_context(
                            agent.cli_type,
                            agent.home_id,
                            run_env=env,
                            workdir=workdir,
                        ),
                    )
                )
                provider_result = (
                    _legacy_provider_compaction_result(standard_result)
                    if standard_result is not None
                    else None
                )
            else:
                home = self.home_service.get_home(agent.cli_type, agent.home_id)
                home_root = self.home_service.resolve_home_root(agent.cli_type, agent.home_id)
                provider_env = build_provider_env(home=home, home_root=home_root, run_env=env)
                provider_result = reconcile(
                    home_id=agent.home_id,
                    home_root=home_root,
                    env=provider_env,
                    thread_id=journal.session_id,
                    workdir=workdir,
                    agent_id=agent.agent_id,
                    baseline=journal.baseline,
                    provider_operation_id=journal.provider_operation_id,
                )
            if provider_result is None:
                raise AgentContextMaintenanceBlocked(
                    f"provider has not confirmed context maintenance terminal state: {agent_id}"
                )
            if not isinstance(provider_result, ProviderContextCompactionResult):
                raise TypeError(f"provider returned invalid reconciliation result: {agent.cli_type}")
            confirmed = AgentContextMaintenanceJournal(
                agent_id=agent.agent_id,
                provider_type=agent.cli_type,
                session_id=journal.session_id,
                status=AgentContextMaintenanceJournalStatus.CONFIRMED,
                trigger=journal.trigger,
                prepared_at=journal.prepared_at,
                started_at=journal.started_at or provider_result.started_at,
                completed_at=provider_result.completed_at,
                provider_operation_id=journal.provider_operation_id,
                baseline=journal.baseline,
            )
            self.store.write_context_maintenance(agent_id, confirmed)
            active.latest_result = confirmed
            return confirmed
        except BaseException as exc:
            active.error = exc
            raise
        finally:
            self._finish_synchronous_maintenance(active)

    def _begin_synchronous_maintenance(
        self,
        agent_id: str,
        *,
        require_resolved: bool = True,
        respect_pause: bool = True,
    ) -> _ActiveAgentRun:
        with self._status_condition:
            agent = self.store.get_agent(agent_id)
            if agent.status == "closed":
                raise AgentClosedError(agent_id)
            if agent.status == "running" or agent_id in self._active:
                raise AgentAlreadyRunningError(agent_id)
            if require_resolved:
                self._assert_context_maintenance_resolved(agent_id)
            if respect_pause:
                self._assert_agent_can_start(agent.scope_id)
            self.store.patch_agent(agent_id, status="running")
            active = _ActiveAgentRun(
                agent_id=agent_id,
                worker=threading.current_thread(),
                done_event=threading.Event(),
            )
            self._active[agent_id] = active
            self._status_condition.notify_all()
            return active

    def _finish_synchronous_maintenance(self, active: _ActiveAgentRun) -> None:
        with self._status_condition:
            try:
                agent = self.store.get_agent(active.agent_id)
                if agent.status != "closed":
                    self.store.patch_agent(active.agent_id, status="idle")
            finally:
                self._active.pop(active.agent_id, None)
                active.done_event.set()
                self._status_condition.notify_all()

    def _compact_agent_context(
        self,
        agent_id: str,
        *,
        threshold: float | None,
        timeout_s: float,
        trigger: str,
        force: bool,
        env: dict[str, str] | None,
        workdir: str | None,
    ) -> AgentContextCompactionResult:
        self._assert_context_maintenance_resolved(agent_id)
        agent = self.store.get_agent(agent_id)
        usage_before = self.inspect_agent_context(agent_id, env=env, workdir=workdir)
        now = utc_now_iso()
        if not agent.thread_id:
            return self._skipped_context_compaction(agent, usage_before, now, "no_session")
        bundle = self._provider_bundle(agent.cli_type)
        context_adapter = bundle.context if bundle is not None else None
        # COMPAT(legacy-provider-context-compact): retain compact_thread only
        # for injected providers without a ProviderContextAdapter.
        provider = self.providers.get(agent.cli_type)
        compact = getattr(provider, "compact_thread", None)
        if context_adapter is None and not callable(compact):
            if force:
                raise AgentContextMaintenanceUnsupported(agent.cli_type)
            return self._unsupported_context_compaction(agent, usage_before, now)
        if context_adapter is not None:
            home = self.home_service.get_home(agent.cli_type, agent.home_id)
            model_backend = self._provider_session_locator(agent).backend_identity
            try:
                compact_support = bundle.resolve_capabilities(home, model_backend).get(
                    CapabilityKey.CONTROL_COMPACT
                )
            except ProviderCapabilityUnavailable:
                compact_support = None
            if compact_support is None or not compact_support.available:
                if force:
                    reason = compact_support.reason if compact_support is not None else None
                    detail = f": {reason}" if reason else ""
                    raise AgentContextMaintenanceUnsupported(
                        f"{agent.cli_type} does not support {CapabilityKey.CONTROL_COMPACT.value}{detail}"
                    )
                return self._unsupported_context_compaction(agent, usage_before, now)
        if not force:
            if not usage_before.available:
                return self._skipped_context_compaction(
                    agent,
                    usage_before,
                    now,
                    usage_before.reason or "usage_unavailable",
                )
            if threshold is None:
                raise ValueError("threshold is required for conditional compaction")
            assert usage_before.usage_ratio is not None
            if usage_before.usage_ratio < threshold:
                return self._skipped_context_compaction(agent, usage_before, now, "below_threshold")

        if context_adapter is not None:
            execution_context = self.home_service.build_execution_context(
                agent.cli_type,
                agent.home_id,
                run_env=env,
                workdir=workdir,
            )
            home_root = provider_env = None
        else:
            home = self.home_service.get_home(agent.cli_type, agent.home_id)
            home_root = self.home_service.resolve_home_root(agent.cli_type, agent.home_id)
            provider_env = build_provider_env(home=home, home_root=home_root, run_env=env)
            execution_context = None
        prepared_at = utc_now_iso()
        journal = AgentContextMaintenanceJournal(
            agent_id=agent.agent_id,
            provider_type=agent.cli_type,
            session_id=agent.thread_id,
            status=AgentContextMaintenanceJournalStatus.PREPARED,
            trigger=trigger,
            prepared_at=prepared_at,
        )
        self.store.write_context_maintenance(agent_id, journal)
        request_started = False

        def on_compaction_started(baseline: dict[str, object], operation_id: str | None) -> None:
            nonlocal request_started
            request_started = True
            self.store.write_context_maintenance(
                agent_id,
                AgentContextMaintenanceJournal(
                    agent_id=agent.agent_id,
                    provider_type=agent.cli_type,
                    session_id=agent.thread_id,
                    status=AgentContextMaintenanceJournalStatus.STARTED,
                    trigger=trigger,
                    prepared_at=prepared_at,
                    started_at=utc_now_iso(),
                    provider_operation_id=operation_id,
                    baseline=baseline,
                ),
            )

        try:
            if context_adapter is not None:
                standard_result = context_adapter.compact(
                    ProviderContextCompactionRequest(
                        session=self._provider_session_locator(agent),
                        trigger=trigger,
                        timeout_s=timeout_s,
                        agent_id=agent.agent_id,
                        execution_context=execution_context,
                        on_started=on_compaction_started,
                    )
                )
                provider_result = _legacy_provider_compaction_result(standard_result)
            else:
                provider_result = compact(
                    home_id=agent.home_id,
                    home_root=home_root,
                    env=provider_env,
                    thread_id=agent.thread_id,
                    workdir=workdir,
                    agent_id=agent.agent_id,
                    timeout_s=timeout_s,
                    on_compaction_started=on_compaction_started,
                )
        except BaseException as exc:
            if request_started or isinstance(exc, AgentContextCompactionRequestUnknown):
                started_journal = self.store.read_context_maintenance(agent_id)
                self.store.write_context_maintenance(
                    agent_id,
                    AgentContextMaintenanceJournal(
                        agent_id=agent.agent_id,
                        provider_type=agent.cli_type,
                        session_id=agent.thread_id,
                        status=AgentContextMaintenanceJournalStatus.UNKNOWN_TERMINAL,
                        trigger=trigger,
                        prepared_at=prepared_at,
                        started_at=started_journal.started_at if started_journal is not None else None,
                        provider_operation_id=(
                            started_journal.provider_operation_id if started_journal is not None else None
                        ),
                        baseline=started_journal.baseline if started_journal is not None else {},
                        error_type=type(exc).__name__,
                    ),
                )
            else:
                self.store.clear_context_maintenance(agent_id)
            raise
        if not isinstance(provider_result, ProviderContextCompactionResult):
            if request_started:
                started_journal = self.store.read_context_maintenance(agent_id)
                self.store.write_context_maintenance(
                    agent_id,
                    AgentContextMaintenanceJournal(
                        agent_id=agent.agent_id,
                        provider_type=agent.cli_type,
                        session_id=agent.thread_id,
                        status=AgentContextMaintenanceJournalStatus.UNKNOWN_TERMINAL,
                        trigger=trigger,
                        prepared_at=prepared_at,
                        started_at=started_journal.started_at if started_journal is not None else None,
                        provider_operation_id=(
                            started_journal.provider_operation_id if started_journal is not None else None
                        ),
                        baseline=started_journal.baseline if started_journal is not None else {},
                        error_type="InvalidProviderCompactionResult",
                    ),
                )
            else:
                self.store.clear_context_maintenance(agent_id)
            raise TypeError(f"provider returned invalid compaction result: {agent.cli_type}")
        if provider_result.session_id != agent.thread_id:
            started_journal = self.store.read_context_maintenance(agent_id)
            self.store.write_context_maintenance(
                agent_id,
                AgentContextMaintenanceJournal(
                    agent_id=agent.agent_id,
                    provider_type=agent.cli_type,
                    session_id=agent.thread_id,
                    status=AgentContextMaintenanceJournalStatus.UNKNOWN_TERMINAL,
                    trigger=trigger,
                    prepared_at=prepared_at,
                    started_at=started_journal.started_at if started_journal is not None else None,
                    provider_operation_id=provider_result.provider_operation_id,
                    baseline=started_journal.baseline if started_journal is not None else {},
                    error_type="ProviderSessionMismatch",
                ),
            )
            raise TypeError(
                f"provider compaction session mismatch: expected {agent.thread_id}, "
                f"got {provider_result.session_id}"
            )
        usage_after = (
            provider_result.usage_after.for_agent(agent_id=agent.agent_id, provider_type=agent.cli_type)
            if provider_result.usage_after is not None
            else None
        )
        self.store.write_context_maintenance(
            agent_id,
            AgentContextMaintenanceJournal(
                agent_id=agent.agent_id,
                provider_type=agent.cli_type,
                session_id=agent.thread_id,
                status=AgentContextMaintenanceJournalStatus.CONFIRMED,
                trigger=trigger,
                prepared_at=prepared_at,
                started_at=provider_result.started_at,
                completed_at=provider_result.completed_at,
                provider_operation_id=provider_result.provider_operation_id,
            ),
        )
        return AgentContextCompactionResult(
            agent_id=agent.agent_id,
            provider_type=agent.cli_type,
            session_id=agent.thread_id,
            status=AgentContextCompactionStatus.COMPACTED,
            reason="forced" if force else "threshold_reached",
            usage_before=usage_before,
            usage_after=usage_after,
            started_at=provider_result.started_at,
            completed_at=provider_result.completed_at,
            provider_operation_id=provider_result.provider_operation_id,
        )

    def _assert_context_maintenance_resolved(self, agent_id: str) -> None:
        journal = self.store.read_context_maintenance(agent_id)
        if journal is not None and journal.unresolved:
            raise AgentContextMaintenanceBlocked(
                f"agent has unresolved context maintenance: {agent_id} ({journal.status.value})"
            )

    def _unavailable_context_usage(self, agent: Agent, reason: str) -> AgentContextUsage:
        return AgentContextUsage(
            agent_id=agent.agent_id,
            provider_type=agent.cli_type,
            session_id=agent.thread_id,
            total_tokens=None,
            context_window=None,
            observed_at=utc_now_iso(),
            source="provider",
            available=False,
            reason=reason,
        )

    def _skipped_context_compaction(
        self,
        agent: Agent,
        usage: AgentContextUsage,
        now: str,
        reason: str,
    ) -> AgentContextCompactionResult:
        return AgentContextCompactionResult(
            agent_id=agent.agent_id,
            provider_type=agent.cli_type,
            session_id=agent.thread_id,
            status=AgentContextCompactionStatus.SKIPPED,
            reason=reason,
            usage_before=usage,
            usage_after=None,
            started_at=now,
            completed_at=now,
        )

    def _unsupported_context_compaction(
        self,
        agent: Agent,
        usage: AgentContextUsage,
        now: str,
    ) -> AgentContextCompactionResult:
        return AgentContextCompactionResult(
            agent_id=agent.agent_id,
            provider_type=agent.cli_type,
            session_id=agent.thread_id,
            status=AgentContextCompactionStatus.UNSUPPORTED,
            reason="provider_unsupported",
            usage_before=usage,
            usage_after=None,
            started_at=now,
            completed_at=now,
        )

    def reconcile_stale_running_agents(self, scope_id: str | None = None) -> list[str]:
        """Compatibility helper for locator-free stale records only."""

        repaired: list[str] = []
        for audit in self.audit_running_agents(scope_id=scope_id):
            if audit.classification != "safe_to_mark_idle":
                continue
            result = self.repair_running_agent(
                audit.agent_id,
                expected_scope_id=audit.scope_id,
                expected_thread_id=audit.thread_id,
                expected_rollout_relpath=audit.rollout_relpath,
                action="mark_idle",
                dry_run=False,
            )
            if result.repaired:
                repaired.append(audit.agent_id)
        return repaired

    def audit_running_agents(self, scope_id: str | None = None) -> list[RunningAgentAuditRecord]:
        records: list[RunningAgentAuditRecord] = []
        with self._lock:
            active_ids = set(self._active)
            for agent in self.store.list_agents(scope_id=scope_id, status="running"):
                provider = self.providers.get(agent.cli_type)
                provider_active = False
                if provider is not None and hasattr(provider, "list_active_agents"):
                    provider_active = agent.agent_id in provider.list_active_agents(agent.home_id)
                if agent.agent_id in active_ids or provider_active:
                    classification = "healthy_running"
                    evidence = ("active_worker",) if agent.agent_id in active_ids else ("provider_active",)
                elif agent.thread_id or agent.rollout_relpath:
                    classification = "requires_review"
                    evidence = tuple(
                        item
                        for item in (
                            "thread_locator_present" if agent.thread_id else None,
                            "rollout_locator_present" if agent.rollout_relpath else None,
                        )
                        if item is not None
                    )
                else:
                    classification = "safe_to_mark_idle"
                    evidence = ("no_active_worker_or_locator",)
                records.append(
                    RunningAgentAuditRecord(
                        agent_id=agent.agent_id,
                        scope_id=agent.scope_id,
                        classification=classification,
                        thread_id=agent.thread_id,
                        rollout_relpath=agent.rollout_relpath,
                        evidence=evidence,
                    )
                )
        return records

    def repair_running_agent(
        self,
        agent_id: str,
        *,
        expected_scope_id: str,
        expected_thread_id: str | None,
        expected_rollout_relpath: str | None,
        action: str,
        dry_run: bool = True,
    ) -> RunningAgentRepairResult:
        if action != "mark_idle":
            raise ValueError("agent repair action must be 'mark_idle'")
        audits = {item.agent_id: item for item in self.audit_running_agents()}
        audit = audits.get(agent_id)
        if audit is None:
            raise RuntimeError(f"agent is not a running repair candidate: {agent_id}")
        if (
            audit.scope_id != expected_scope_id
            or audit.thread_id != expected_thread_id
            or audit.rollout_relpath != expected_rollout_relpath
        ):
            raise RuntimeError(f"agent repair identity changed: {agent_id}")
        if audit.classification == "healthy_running":
            raise RuntimeError(f"healthy running agent cannot be repaired: {agent_id}")
        repaired = False
        if not dry_run:
            with self._status_condition:
                current = self.store.get_agent(agent_id)
                if current.status != "running":
                    raise RuntimeError(f"agent status changed before repair: {agent_id}")
                self.store.patch_agent(agent_id, status="idle")
                repaired = True
                self._status_condition.notify_all()
        return RunningAgentRepairResult(
            agent_id=agent_id,
            classification=audit.classification,
            action=action,
            dry_run=dry_run,
            repaired=repaired,
        )

    def interrupt_agent(self, agent_id: str, timeout_s: float | None = None) -> bool:
        agent = self.store.get_agent(agent_id)
        bundle = self._provider_bundle(agent.cli_type)
        active = self._active.get(agent_id)
        if bundle is not None and active is not None:
            deadline = None if timeout_s is None else monotonic() + timeout_s
            if active.provider_handle is None:
                remaining = None if deadline is None else max(0.0, deadline - monotonic())
                if not active.handle_ready.wait(remaining):
                    return False
            handle = active.provider_handle
            if handle is None:
                return False
            remaining = None if deadline is None else max(0.0, deadline - monotonic())
            result = handle.interrupt(remaining)
            if not (result.accepted and result.terminal_confirmed):
                return False
            remaining = None if deadline is None else max(0.0, deadline - monotonic())
            return active.done_event.wait(remaining)
        provider = self.providers.get(agent.cli_type)
        # COMPAT(legacy-provider-interrupt): providers registered through the
        # old dictionary expose interrupt_agent(agent_id). Provider bundles use
        # the active ProviderRunHandle directly. Remove with the providers dict.
        if provider is None or not hasattr(provider, "interrupt_agent"):
            return False
        accepted = bool(provider.interrupt_agent(agent_id))
        if not accepted:
            return False
        active = self._active.get(agent_id)
        if active is None:
            return True
        if not active.done_event.wait(timeout_s):
            return False
        return True

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
        bundle = self._provider_bundle(source.cli_type)
        if bundle is not None:
            target_agent_id = f"a_{uuid.uuid4().hex}"
            source_session = source.session_locator or self._provider_session_locator(source)
            forked_standard = bundle.runtime.fork(
                ProviderForkRequest(
                    source_agent_id=source.agent_id,
                    source_session=source_session,
                    source_turn=source.latest_turn_locator,
                    target_agent_id=target_agent_id,
                    target_scope_id=target_scope,
                    target_home_id=source.home_id,
                    execution_context=self.home_service.build_execution_context(
                        source.cli_type,
                        source.home_id,
                    ),
                )
            )
            rollout_relpath = (
                forked_standard.artifact_locator.native_primary_ref
                if forked_standard.artifact_locator is not None
                else _native_rollout_relpath(forked_standard.target_session.native_locator)
            )
            return self.store.create_agent_record(
                agent_id=target_agent_id,
                scope_id=target_scope,
                agent_type=source.agent_type,
                cli_type=source.cli_type,
                provider_type=source.provider_type,
                home_id=source.home_id,
                session_locator=forked_standard.target_session,
                artifact_locator=forked_standard.artifact_locator,
                thread_id=forked_standard.target_session.session_id,
                rollout_relpath=rollout_relpath,
                fork_info=AgentForkInfo(
                    source_agent_id=source.agent_id,
                    source_session_id=source_session.session_id,
                    source_turn_id=(
                        forked_standard.source_turn.turn_id
                        if forked_standard.source_turn is not None
                        else None
                    ),
                    fork_mode=forked_standard.fork_mode,
                    workspace_isolated=forked_standard.workspace_isolated,
                    created_at=utc_now_iso(),
                ),
                fork_source_agent_id=source.agent_id,
                fork_source_thread_id=source.thread_id,
            )

        # COMPAT(legacy-provider-fork): injected providers without a bundle use
        # fork_thread and the legacy locator fields. Remove when LC fakes and
        # external takeover providers implement ProviderRuntimeAdapter.fork().
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

    def resolve_provider_capabilities(
        self,
        *,
        provider_type: str,
        home_id: str,
        model_backend: ModelBackendIdentity | None = None,
    ) -> ProviderCapabilities:
        home = self.home_service.get_home(provider_type, home_id)
        bundle = self._provider_bundle(provider_type)
        if bundle is None:
            raise RuntimeError(f"provider does not expose standard capabilities: {provider_type}")
        return bundle.resolve_capabilities(home, model_backend)

    def query_sessions(
        self,
        *,
        provider_type: str,
        home_id: str,
        cursor: str | None = None,
        limit: int = 100,
    ) -> Page:
        self.home_service.get_home(provider_type, home_id)
        bundle = self._provider_bundle(provider_type)
        if bundle is None or bundle.query is None:
            raise RuntimeError(f"provider does not support standard query: {provider_type}")
        return bundle.query.list_sessions(
            ProviderSessionListQuery(home_id=home_id, cursor=cursor, limit=limit)
        )

    def query_turns(
        self,
        agent_id: str,
        *,
        cursor: str | None = None,
        limit: int = 100,
    ) -> Page:
        bundle, session = self._query_bundle_and_session(agent_id)
        return bundle.query.list_turns(
            ProviderTurnQuery(session=session, cursor=cursor, limit=limit)
        )

    def query_turn(
        self,
        agent_id: str,
        *,
        turn_id: str | None = None,
        latest: bool = False,
    ) -> object | None:
        bundle, session = self._query_bundle_and_session(agent_id)
        turn = ProviderTurnLocator(session=session, turn_id=turn_id) if turn_id is not None else None
        return bundle.query.read_turn(
            ProviderTurnQuery(session=session, turn=turn, latest=latest)
        )

    def query_events(
        self,
        agent_id: str,
        *,
        turn_id: str | None = None,
        kind: str | None = None,
        cursor: str | None = None,
        limit: int = 100,
    ) -> Page:
        bundle, session = self._query_bundle_and_session(agent_id)
        turn = ProviderTurnLocator(session=session, turn_id=turn_id) if turn_id is not None else None
        return bundle.query.list_events(
            ProviderEventQuery(
                session=session,
                turn=turn,
                kind=kind,
                cursor=cursor,
                limit=limit,
            )
        )

    def query_tool_calls(
        self,
        agent_id: str,
        *,
        turn_id: str | None = None,
        call_id: str | None = None,
        cursor: str | None = None,
        limit: int = 100,
    ) -> Page:
        bundle, session = self._query_bundle_and_session(agent_id)
        turn = ProviderTurnLocator(session=session, turn_id=turn_id) if turn_id is not None else None
        return bundle.query.list_tool_calls(
            ProviderToolQuery(
                session=session,
                turn=turn,
                call_id=call_id,
                cursor=cursor,
                limit=limit,
            )
        )

    def query_usage(
        self,
        agent_id: str,
        *,
        turn_id: str | None = None,
        latest: bool = False,
        include_session_aggregate: bool = False,
    ) -> object:
        bundle, session = self._query_bundle_and_session(agent_id)
        turn = ProviderTurnLocator(session=session, turn_id=turn_id) if turn_id is not None else None
        return bundle.query.read_usage(
            ProviderUsageQuery(
                session=session,
                turn=turn,
                latest=latest,
                include_session_aggregate=include_session_aggregate,
            )
        )

    def _query_bundle_and_session(
        self,
        agent_id: str,
    ) -> tuple[AgentProviderBundle, ProviderSessionLocator]:
        agent = self.store.get_agent(agent_id)
        if not agent.thread_id:
            raise AgentHasNoCompletedTurn(agent_id)
        bundle = self._provider_bundle(agent.cli_type)
        if bundle is None or bundle.query is None:
            raise RuntimeError(f"provider does not support standard query: {agent.cli_type}")
        return bundle, self._provider_session_locator(agent)

    def _provider_session_locator(self, agent: Agent) -> ProviderSessionLocator:
        if not agent.thread_id:
            raise AgentHasNoCompletedTurn(agent.agent_id)
        if agent.session_locator is not None:
            return agent.session_locator
        return ProviderSessionLocator(
            provider_type=agent.cli_type,
            session_id=agent.thread_id,
            home_id=agent.home_id,
            created_at=agent.created_at or utc_now_iso(),
            native_locator={"rollout_relpath": agent.rollout_relpath},
        )

    def _refresh_codex_rollout_locator(self, agent_id: str) -> Agent:
        agent = self.store.get_agent(agent_id)
        if agent.cli_type != "codex" or not agent.thread_id or agent.rollout_relpath:
            return agent
        provider = self.providers.get(agent.cli_type)
        find = getattr(provider, "find_rollout_relpath", None)
        if not callable(find):
            return agent
        home_root = self.home_service.resolve_home_root(agent.cli_type, agent.home_id)
        rollout_relpath = find(home_root=home_root, thread_id=agent.thread_id)
        if not rollout_relpath:
            return agent
        return self.store.update_thread_locator(
            agent_id,
            thread_id=agent.thread_id,
            rollout_relpath=rollout_relpath,
        )

    def read_rollout_events(self, agent_id: str) -> list[dict]:
        self._refresh_codex_rollout_locator(agent_id)
        return self.store.read_rollout_events(agent_id)

    def trace_reader(self, agent_id: str):
        self._refresh_codex_rollout_locator(agent_id)
        return self.store.trace_reader(agent_id)

    def get_rollout_info(self, agent_id: str):
        self._refresh_codex_rollout_locator(agent_id)
        return self.store.get_rollout_info(agent_id)

    def list_trace_turns(self, agent_id: str):
        self._refresh_codex_rollout_locator(agent_id)
        return self.store.list_trace_turns(agent_id)

    def get_trace_turn(
        self,
        agent_id: str,
        *,
        turn_id: str | None = None,
        index: int | None = None,
        latest: bool = False,
    ):
        self._refresh_codex_rollout_locator(agent_id)
        return self.store.get_trace_turn(agent_id, turn_id=turn_id, index=index, latest=latest)

    def get_trace_event(
        self,
        agent_id: str,
        *,
        index: int | None = None,
        last: bool = False,
    ):
        self._refresh_codex_rollout_locator(agent_id)
        return self.store.get_trace_event(agent_id, index=index, last=last)

    def tail_trace_events(
        self,
        agent_id: str,
        *,
        limit: int = 20,
        event_type: str | None = None,
        payload_type: str | None = None,
    ):
        self._refresh_codex_rollout_locator(agent_id)
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
        self._refresh_codex_rollout_locator(agent_id)
        return self.store.list_response_texts(agent_id, turn_id=turn_id, latest=latest)

    def get_latest_response_text(self, agent_id: str) -> str | None:
        self._refresh_codex_rollout_locator(agent_id)
        return self.store.get_latest_response_text(agent_id)

    def list_tool_calls(
        self,
        agent_id: str,
        *,
        turn_id: str | None = None,
        latest: bool = False,
    ):
        self._refresh_codex_rollout_locator(agent_id)
        return self.store.list_tool_calls(agent_id, turn_id=turn_id, latest=latest)

    def get_tool_call(
        self,
        agent_id: str,
        *,
        call_id: str | None = None,
        index: int | None = None,
        last: bool = False,
    ):
        self._refresh_codex_rollout_locator(agent_id)
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
        closed_provider_ids: set[int] = set()
        for bundle in self.provider_registry.list():
            bundle.runtime.close()
            source = self._provider_bundle_sources.get(bundle.provider_type)
            if source is not None:
                closed_provider_ids.add(id(source))
        for provider in self.providers.values():
            if id(provider) in closed_provider_ids:
                continue
            close = getattr(provider, "close", None)
            if callable(close):
                close()

    def _export_trace_reports_best_effort(self, agent_id: str) -> None:
        if self.trace_report_policy.persistence == TraceReportPersistence.DISABLED:
            return
        try:
            self.store.export_default_trace_reports(
                agent_id,
                include_turn_history=(
                    self.trace_report_policy.persistence == TraceReportPersistence.LATEST_AND_TURNS
                ),
            )
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


def _standard_or_legacy_turn_id(
    standard_result: AgentTurnResult | None,
    legacy_result: object,
) -> str:
    if standard_result is not None and standard_result.turn_locator is not None:
        return standard_result.turn_locator.turn_id
    return _turn_id(legacy_result)


def _native_rollout_relpath(native_locator: object | None) -> str | None:
    if not isinstance(native_locator, dict):
        return None
    value = native_locator.get("rollout_relpath")
    return str(value) if value is not None else None


def _build_provider_bundle(
    provider: object | None,
    *,
    runtime_root: Path,
) -> AgentProviderBundle | None:
    if provider is None:
        return None
    # COMPAT(provider-self-bundle-bootstrap): lets existing constructor callers
    # pass a CodexProvider while AgentService migrates to ProviderRegistry.
    # Future providers should pass ProviderRegistry explicitly.
    builder = getattr(provider, "build_provider_bundle", None)
    if not callable(builder):
        return None
    bundle = builder(runtime_root=runtime_root)
    if not isinstance(bundle, AgentProviderBundle):
        raise TypeError("build_provider_bundle() must return AgentProviderBundle")
    return bundle


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


def _standard_unavailable_context_usage(agent: Agent, reason: str) -> StandardAgentContextUsage:
    return StandardAgentContextUsage(
        agent_id=agent.agent_id,
        provider_type=agent.cli_type,
        session_id=agent.thread_id,
        observed_at=utc_now_iso(),
        source="unavailable",
        available=False,
        measurement="unavailable",
        reason=reason,
    )


def _legacy_agent_context_usage(usage: StandardAgentContextUsage) -> AgentContextUsage:
    window = usage.context_window
    available = usage.available and usage.used_tokens is not None and window is not None
    return AgentContextUsage(
        agent_id=usage.agent_id,
        provider_type=usage.provider_type,
        session_id=usage.session_id,
        total_tokens=usage.used_tokens,
        context_window=window,
        observed_at=usage.observed_at,
        source=usage.source,
        available=available,
        reason=usage.reason if available else usage.reason or "context_window_unavailable",
    )


def _legacy_provider_context_usage(
    usage: StandardProviderContextUsage,
) -> ProviderContextUsage:
    window = usage.context_window
    available = usage.available and usage.used_tokens is not None and window is not None
    return ProviderContextUsage(
        session_id=usage.session_id,
        total_tokens=usage.used_tokens,
        context_window=window,
        observed_at=usage.observed_at,
        source=usage.source,
        available=available,
        reason=usage.reason if available else usage.reason or "context_window_unavailable",
    )


def _legacy_provider_compaction_result(
    result: StandardProviderContextCompactionResult,
) -> ProviderContextCompactionResult:
    if not isinstance(result, StandardProviderContextCompactionResult):
        raise TypeError("provider returned invalid standard compaction result")
    return ProviderContextCompactionResult(
        session_id=result.session_id,
        usage_after=(
            _legacy_provider_context_usage(result.usage_after)
            if result.usage_after is not None
            else None
        ),
        started_at=result.started_at,
        completed_at=result.completed_at,
        provider_operation_id=result.provider_operation_id,
    )
