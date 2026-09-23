"""Isaac side: start (or connect to) the lerobot server and call it, failing loudly.

Failure model. The eval loop (scripts/eval_policy.py) catches any exception
from ``policy.eval``, logs it with a traceback, marks the seed ``error`` and
moves on to the next seed. So a silent or late failure here turns into a run
of "error" seeds; this client therefore

* retries only connection-level failures (refused, reset, timeout) a few
  times with a short backoff, never an answer the server did give;
* checks the server process between retries and raises with its exit code
  the moment it is gone;
* raises :class:`BridgeError` with the server's own traceback when the
  server answered ``{"type": "error"}``;
* validates the action (shape, dtype, finite) before handing it to Isaac;
* stays ``dead`` after a fatal failure so every later call fails at once
  instead of waiting through another timeout.
"""

from __future__ import annotations

import http.client
import json
import os
import secrets
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from ..convert.schema import STATE_DIM
from .wire import from_wire, to_wire

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVER_SCRIPT = REPO_ROOT / "policy" / "lerobot" / "server.py"
DEFAULT_LEROBOT_PYTHON = "~/miniforge3/envs/lerobot/bin/python"
CONNECTION_ERRORS = (
    urllib.error.URLError,
    socket.timeout,
    TimeoutError,
    ConnectionError,
    http.client.RemoteDisconnected,
    http.client.IncompleteRead,
)


class BridgeError(RuntimeError):
    """The lerobot server is unreachable, died, or answered with an error."""


def _log(msg: str) -> None:
    print(f"[lerobot-bridge] {msg}", file=sys.stderr, flush=True)


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class ServerClient:
    """HTTP client for ``server.py``; ``process`` is the server subprocess when this client started it."""

    def __init__(
        self,
        host: str,
        port: int,
        authkey: str,
        request_timeout: float = 300.0,
        retries: int = 3,
        backoff: float = 0.5,
        process: subprocess.Popen | None = None,
        action_dim: int = STATE_DIM,
    ):
        self.base_url = f"http://{host}:{port}"
        self.authkey = authkey
        self.request_timeout = request_timeout
        self.retries = retries
        self.backoff = backoff
        self.process = process
        self.action_dim = action_dim
        self.dead: str | None = None  # why the bridge is unusable, once it is
        self._closed = False

    # --- transport ------------------------------------------------------------------

    def _request(self, method: str, path: str, data: bytes | None = None) -> Any:
        """One HTTP round trip. Connection failures raise BridgeError('connection: ...')."""
        headers = {"X-Lerobot-Auth": self.authkey}
        if data is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self.base_url + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.request_timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise BridgeError(f"server HTTP {exc.code} on {path}: {exc.read().decode('utf-8', 'replace')}") from exc
        except CONNECTION_ERRORS as exc:
            raise BridgeError(f"connection: {path}: {exc!r}") from exc

    def server_exit_code(self) -> int | None:
        return self.process.poll() if self.process is not None else None

    def _fail(self, reason: str) -> BridgeError:
        self.dead = reason
        _log(reason)
        return BridgeError(reason)

    def post(self, path: str, payload: Any) -> Any:
        """POST JSON; retry connection failures; raise BridgeError on any server-side error."""
        if self.dead:
            raise BridgeError(f"bridge is down: {self.dead}")
        data = json.dumps(payload).encode("utf-8")
        last: BridgeError | None = None
        for attempt in range(1, self.retries + 1):
            code = self.server_exit_code()
            if code is not None:
                raise self._fail(f"server process exited with code {code} before {path}")
            try:
                response = self._request("POST", path, data)
            except BridgeError as exc:
                if not str(exc).startswith("connection:"):
                    raise self._fail(str(exc))
                last = exc
                _log(f"{path} attempt {attempt}/{self.retries} failed: {exc}")
                time.sleep(self.backoff * attempt)
                continue
            if isinstance(response, dict) and response.get("type") == "error":
                raise self._fail(f"server error on {path}:\n{response.get('error')}\n{response.get('traceback')}")
            return response
        raise self._fail(f"{path} unreachable after {self.retries} attempts: {last}")

    # --- API ------------------------------------------------------------------------

    def wait_until_ready(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        last: BaseException | None = None
        while time.monotonic() < deadline:
            code = self.server_exit_code()
            if code is not None:
                raise self._fail(f"server process exited with code {code} during startup")
            try:
                if self._request("GET", "/health").get("status") == "ok":
                    return
            except BridgeError as exc:
                last = exc
            time.sleep(0.25)
        raise self._fail(f"server not ready after {timeout:.0f} s: {last}")

    def init(self, args: dict) -> None:
        self.post("/init", {"authkey": self.authkey, "args": to_wire(args)})

    def act(self, observation: dict, instruction: str) -> np.ndarray:
        response = self.post(
            "/act", {"authkey": self.authkey, "instruction": instruction, "observation": to_wire(observation)}
        )
        return validate_action(from_wire(response.get("action")), self.action_dim)

    def reset(self) -> None:
        self.post("/reset", {"authkey": self.authkey})

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if not self.dead:
            try:
                self.post("/shutdown" if self.process is not None else "/reset", {"authkey": self.authkey})
            except BridgeError:
                pass
        if self.process is not None and self.process.poll() is None:
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
        if self.process is not None:
            _log(f"server process finished with code {self.process.returncode}")


def validate_action(action: Any, action_dim: int) -> np.ndarray:
    """The server's answer as a finite float32 ``(action_dim,)`` vector, else BridgeError."""
    if action is None:
        raise BridgeError("server answered /act without an action")
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape != (action_dim,):
        raise BridgeError(f"action has shape {action.shape}, expected ({action_dim},)")
    if not np.isfinite(action).all():
        raise BridgeError(f"action contains non-finite values: {action}")
    return action


def start_server(python: Path, host: str, port: int, authkey: str) -> subprocess.Popen:
    if not python.exists():
        raise BridgeError(f"lerobot python not found: {python} (set lerobot_python or $LEROBOT_PYTHON)")
    return subprocess.Popen(
        [str(python), str(SERVER_SCRIPT), "--host", host, "--port", str(port), "--authkey", authkey],
        cwd=str(REPO_ROOT),
    )


def connect(args: dict) -> ServerClient:
    """Start a private server (``lerobot_port`` 0) or join a running one, wait for it, send /init."""
    host = str(args.get("lerobot_host", os.environ.get("LEROBOT_HOST", "127.0.0.1")))
    port = int(args.get("lerobot_port", os.environ.get("LEROBOT_PORT", 0)) or 0)
    authkey = str(args.get("lerobot_authkey", os.environ.get("LEROBOT_AUTHKEY", "")))
    python = Path(str(args.get("lerobot_python") or os.environ.get("LEROBOT_PYTHON") or DEFAULT_LEROBOT_PYTHON))
    process = None
    if port == 0:
        port = find_free_port()
        authkey = secrets.token_hex(16)
        process = start_server(python.expanduser(), host, port, authkey)
    client = ServerClient(
        host,
        port,
        authkey,
        request_timeout=float(args.get("lerobot_request_timeout", 300)),
        retries=int(args.get("lerobot_retries", 3)),
        process=process,
    )
    client.wait_until_ready(float(args.get("lerobot_startup_timeout", 600)))
    client.init(args)
    return client
