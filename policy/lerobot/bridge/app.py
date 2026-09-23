"""FastAPI app of the inference server: /health, /init, /act, /reset, /shutdown.

Errors raised while serving a request come back as ``{"type": "error",
"error", "traceback"}`` with HTTP 200 so the client can show the server's
traceback; transport-level problems are HTTP status codes (401 auth, 409 not
initialised, 422 malformed body).
"""

from __future__ import annotations

import asyncio
import traceback

# FastAPI resolves the (postponed) annotations of the route functions against
# this module's globals, so Request/Header must be imported at module level.
from fastapi import FastAPI, Header, HTTPException, Request

from .local_policy import LocalPolicy
from .wire import from_wire, to_wire


def create_app(authkey: str) -> FastAPI:
    app = FastAPI(title="UniVTAC lerobot inference server")
    app.state.authkey = authkey
    app.state.model = None
    app.state.server = None

    def check_auth(header_auth: str | None, body_auth: str | None = None) -> None:
        expected = app.state.authkey
        if expected and header_auth != expected and body_auth != expected:
            raise HTTPException(status_code=401, detail="authentication failed")

    def error_payload(exc: BaseException) -> dict:
        return {"type": "error", "error": repr(exc), "traceback": traceback.format_exc()}

    @app.get("/health")
    def health(x_lerobot_auth: str | None = Header(default=None)):
        check_auth(x_lerobot_auth)
        return {"status": "ok", "model_loaded": app.state.model is not None}

    @app.post("/init")
    async def init(request: Request, x_lerobot_auth: str | None = Header(default=None)):
        payload = await request.json()
        check_auth(x_lerobot_auth, payload.get("authkey"))
        try:
            if app.state.model is None:
                app.state.model = LocalPolicy(from_wire(payload["args"]))
            return {"type": "ok"}
        except BaseException as exc:
            return error_payload(exc)

    @app.post("/act")
    async def act(request: Request, x_lerobot_auth: str | None = Header(default=None)):
        payload = await request.json()
        check_auth(x_lerobot_auth, payload.get("authkey"))
        if app.state.model is None:
            raise HTTPException(status_code=409, detail="model is not initialized")
        try:
            action = app.state.model.act(from_wire(payload["observation"]), payload.get("instruction") or "")
            return {"type": "ok", "action": to_wire(action)}
        except BaseException as exc:
            return error_payload(exc)

    @app.post("/reset")
    async def reset(request: Request, x_lerobot_auth: str | None = Header(default=None)):
        payload = await request.json()
        check_auth(x_lerobot_auth, payload.get("authkey"))
        if app.state.model is not None:
            app.state.model.reset()
        return {"type": "ok"}

    @app.post("/shutdown")
    async def shutdown(request: Request, x_lerobot_auth: str | None = Header(default=None)):
        payload = await request.json()
        check_auth(x_lerobot_auth, payload.get("authkey"))
        asyncio.create_task(_shutdown_soon())
        return {"type": "ok"}

    async def _shutdown_soon() -> None:
        await asyncio.sleep(0.1)
        if app.state.server is not None:
            app.state.server.should_exit = True

    return app
