from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

LOGGER = logging.getLogger(__name__)

UPSTREAM_API_ROOT = "https://api.github.com/repos/nikopueringer/CorridorKey"
SYNCED_UPSTREAM_HEAD_SHA = "38248c6dd1785b8d7e51ef257fd0889143ca58af"
SYNCED_UPSTREAM_HEAD_DATE = "2026-03-01T22:23:57Z"
SYNCED_UPSTREAM_HEAD_MESSAGE = (
    "Merge pull request #34 from taylorOntologize/tgw/coverage-review"
)
SYNCED_UPSTREAM_HEAD_CHECK_CONCLUSIONS = ("failure", "success", "cancelled")

_BLOCKING_CHECK_CONCLUSIONS = {
    "action_required",
    "failure",
    "startup_failure",
    "timed_out",
}
_CHECK_THREAD_STARTED = False
_CHECK_THREAD_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class UpstreamCommitRecord:
    sha: str
    date: str
    message: str
    conclusions: tuple[str, ...]


def _parse_bool_env(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() not in {"0", "false", "no", "off"}


def _parse_float_env(name: str, default: float, minimum: float, maximum: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        parsed = float(raw_value)
    except ValueError:
        return default
    return max(minimum, min(maximum, parsed))


def _parse_int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        parsed = int(raw_value)
    except ValueError:
        return default
    return max(minimum, min(maximum, parsed))


def is_verified_check_conclusions(conclusions: Sequence[str]) -> bool:
    normalized = tuple(
        str(conclusion).strip().lower()
        for conclusion in conclusions
        if str(conclusion).strip()
    )
    if not normalized:
        return False
    has_success = any(conclusion == "success" for conclusion in normalized)
    has_blocker = any(conclusion in _BLOCKING_CHECK_CONCLUSIONS for conclusion in normalized)
    return has_success and not has_blocker


def _extract_commit_record(commit_payload: dict[str, Any], check_payload: dict[str, Any]) -> UpstreamCommitRecord:
    message = str(
        commit_payload.get("commit", {})
        .get("message", "")
    ).splitlines()[0]
    conclusions = tuple(
        str(check_run.get("conclusion", "")).strip().lower()
        for check_run in check_payload.get("check_runs", [])
    )
    return UpstreamCommitRecord(
        sha=str(commit_payload.get("sha", "")).strip(),
        date=str(commit_payload.get("commit", {}).get("author", {}).get("date", "")).strip(),
        message=message,
        conclusions=conclusions,
    )


def select_latest_verified_commit(
    commit_payloads: Sequence[dict[str, Any]],
    check_payloads_by_sha: dict[str, dict[str, Any]],
) -> UpstreamCommitRecord | None:
    for commit_payload in commit_payloads:
        sha = str(commit_payload.get("sha", "")).strip()
        if not sha:
            continue
        check_payload = check_payloads_by_sha.get(sha)
        if check_payload is None:
            continue
        record = _extract_commit_record(commit_payload, check_payload)
        if is_verified_check_conclusions(record.conclusions):
            return record
    return None


def _fetch_json(url: str, timeout_seconds: float) -> Any:
    request = Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "ComfyUI-CorridorKey",
        },
        method="GET",
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_latest_verified_commit(timeout_seconds: float, depth: int) -> UpstreamCommitRecord | None:
    commits_url = f"{UPSTREAM_API_ROOT}/commits?sha=main&per_page={depth}"
    commit_payloads = _fetch_json(commits_url, timeout_seconds)
    if not isinstance(commit_payloads, list):
        raise ValueError("GitHub commits API returned an unexpected payload.")

    check_payloads_by_sha: dict[str, dict[str, Any]] = {}
    for commit_payload in commit_payloads:
        sha = str(commit_payload.get("sha", "")).strip()
        if not sha:
            continue
        check_url = f"{UPSTREAM_API_ROOT}/commits/{sha}/check-runs"
        check_payload = _fetch_json(check_url, timeout_seconds)
        if isinstance(check_payload, dict):
            check_payloads_by_sha[sha] = check_payload

    return select_latest_verified_commit(commit_payloads, check_payloads_by_sha)


def _run_upstream_check() -> None:
    timeout_seconds = _parse_float_env("CORRIDORKEY_UPSTREAM_TIMEOUT_SECONDS", 3.0, 0.5, 10.0)
    depth = _parse_int_env("CORRIDORKEY_UPSTREAM_CHECK_DEPTH", 15, 1, 30)

    try:
        latest_verified = fetch_latest_verified_commit(
            timeout_seconds=timeout_seconds,
            depth=depth,
        )
    except (HTTPError, URLError, OSError, TimeoutError, ValueError) as exc:
        LOGGER.debug("CorridorKey upstream check skipped: %s", exc)
        return

    if latest_verified is None:
        LOGGER.debug(
            "CorridorKey upstream check found no recent commit without failing checks."
        )
        return

    if latest_verified.sha == SYNCED_UPSTREAM_HEAD_SHA:
        LOGGER.debug(
            "CorridorKey upstream head %s is already the current reviewed baseline.",
            latest_verified.sha[:12],
        )
        return

    LOGGER.info(
        "CorridorKey upstream update available: %s (%s). "
        "This node does not auto-overwrite itself; review and port changes manually.",
        latest_verified.sha[:12],
        latest_verified.date,
    )


def schedule_upstream_check() -> None:
    if not _parse_bool_env("CORRIDORKEY_AUTO_CHECK_UPSTREAM", True):
        return

    global _CHECK_THREAD_STARTED
    with _CHECK_THREAD_LOCK:
        if _CHECK_THREAD_STARTED:
            return
        _CHECK_THREAD_STARTED = True

    thread = threading.Thread(
        target=_run_upstream_check,
        name="CorridorKeyUpstreamCheck",
        daemon=True,
    )
    thread.start()
