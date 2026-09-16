"""Safe failure locations: no exception text, source lines, locals or credentials."""
import hashlib
import json
from pathlib import Path


def exception_diagnostics(exc: BaseException) -> dict[str, object]:
    frames = []
    tb = exc.__traceback__
    while tb is not None:
        code = tb.tb_frame.f_code
        frames.append({"file": Path(code.co_filename).name, "function": code.co_name, "line": tb.tb_lineno})
        tb = tb.tb_next
    result: dict[str, object] = {"exception_type": type(exc).__name__, "frames": frames[-12:]}
    stage = getattr(exc, "ark_provider_stage", None)
    if isinstance(stage, str):
        result["stage"] = stage
    # Match fixed adapter messages only; never export arbitrary exception text.
    codes = {
        "Grok Home auth reference changed": "grok_home_auth_reference_changed",
        "Grok Home materialization manifest hash mismatch": "grok_home_manifest_mismatch",
        "Grok Home managed file set changed": "grok_home_file_set_changed",
        "Grok session locator has no native identity": "grok_session_native_identity_missing",
        "Grok changed config.toml outside the verified marketplace initialization metadata": "grok_home_config_changed",
    }
    code = codes.get(str(exc))
    if code is not None:
        result["code"] = code
    context = getattr(exc, "ark_context_diagnostics", None)
    if isinstance(context, dict):
        safe = {}
        for key in ("rpc_code", "rpc_elapsed_s", "rpc_attempt", "wait_elapsed_s", "wait_limit_s",
                    "has_new_compacted", "has_new_completion_marker", "has_new_token_count",
                    "evidence_complete", "provider_idle", "rpc_mcp_failure_count",
                    "rpc_mcp_lc_app", "rpc_mcp_lc_submit", "rpc_message_length"):
            if type(context.get(key)) in (int, float, bool):
                safe[key] = context[key]
        for key in ("rpc_message_thread_wrapper", "rpc_message_required_mcp",
                    "rpc_message_startup_timeout", "rpc_message_event_stream_timeout",
                    "rpc_message_handshake", "rpc_message_timeout",
                    "rpc_message_lc_app", "rpc_message_lc_submit"):
            if type(context.get(key)) is bool:
                safe[key] = context[key]
        if context.get("rpc_method") in {"thread/start", "thread/resume", "thread/read", "thread/compact/start"}:
            safe["rpc_method"] = context["rpc_method"]
        if context.get("rpc_category") in {"transport_closed", "server_busy", "unclassified",
                                           "required_mcp_startup_timeout", "required_mcp_startup_failed"}:
            safe["rpc_category"] = context["rpc_category"]
        result["context"] = safe
    signature = {
        "exception_type": result["exception_type"],
        "stage": result.get("stage"),
        "code": result.get("code"),
        "origin": frames[-1] if frames else None,
    }
    result["fingerprint"] = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:16]
    return result
