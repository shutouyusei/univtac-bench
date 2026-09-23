"""Entry point of the lerobot inference server; run with the ``lerobot`` env's python.

``policy/lerobot/deploy_policy.py`` starts this from the Isaac process (or
connects to one started by hand: ``python policy/lerobot/server.py --port
10800``). One ``/act`` request is one ``policy.select_action``: SmolVLA
executes its chunk from an action queue, tacforcing streams the chunk block
by block, each block conditioned on the tactile embedding of the frame that
asked for it. The app lives in ``policy/lerobot/bridge/app.py``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from policy.lerobot.bridge.app import create_app  # noqa: E402


def main() -> int:
    import uvicorn

    parser = argparse.ArgumentParser(description="lerobot policy inference server for UniVTAC")
    parser.add_argument("--host", default=os.environ.get("LEROBOT_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=os.environ.get("LEROBOT_PORT"))
    parser.add_argument("--authkey", default=os.environ.get("LEROBOT_AUTHKEY", ""))
    args = parser.parse_args()
    if args.port is None:
        raise SystemExit("--port or LEROBOT_PORT is required")

    app = create_app(args.authkey)
    server = uvicorn.Server(
        uvicorn.Config(app, host=args.host, port=int(args.port), log_level="info", access_log=False)
    )
    app.state.server = server
    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
