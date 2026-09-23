"""ServerClient failure model: retries only connection failures, dies loudly, validates actions."""

import numpy as np
import pytest

from policy.lerobot.bridge import client as cl
from policy.lerobot.bridge.wire import to_wire


class FakeProcess:
    def __init__(self, exit_code=None):
        self.exit_code = exit_code
        self.returncode = exit_code

    def poll(self):
        return self.exit_code


def make_client(responses, process=None, retries=3):
    """A client whose _request pops scripted results: a callable raises, anything else is returned."""
    c = cl.ServerClient("127.0.0.1", 1, "k", request_timeout=1, retries=retries, backoff=0.0, process=process)
    script = list(responses)
    calls = []

    def fake_request(method, path, data=None):
        calls.append(path)
        item = script.pop(0)
        if callable(item):
            raise item()
        return item

    c._request = fake_request
    c.calls = calls
    return c


def conn_err():
    return cl.BridgeError("connection: /act: ConnectionRefusedError")


def test_act_returns_validated_action():
    c = make_client([{"type": "ok", "action": to_wire(np.arange(8, dtype=np.float32))}])
    out = c.act({"images": {}, "tactile": {}, "joint": np.zeros(8)}, "go")
    np.testing.assert_array_equal(out, np.arange(8, dtype=np.float32))
    assert out.dtype == np.float32 and c.dead is None


def test_connection_failures_are_retried_then_fatal():
    c = make_client([conn_err, conn_err, {"type": "ok"}])
    c.reset()
    assert c.calls == ["/reset"] * 3 and c.dead is None

    c = make_client([conn_err, conn_err, conn_err], retries=3)
    with pytest.raises(cl.BridgeError, match="unreachable after 3 attempts"):
        c.reset()
    assert c.dead
    with pytest.raises(cl.BridgeError, match="bridge is down"):
        c.reset()  # no further network calls once dead
    assert len(c.calls) == 3


def test_server_error_response_is_not_retried():
    c = make_client([{"type": "error", "error": "ValueError('x')", "traceback": "tb"}, {"type": "ok"}])
    with pytest.raises(cl.BridgeError, match="server error on /reset"):
        c.reset()
    assert c.calls == ["/reset"] and c.dead


def test_http_error_is_not_retried():
    def http_err():
        return cl.BridgeError("server HTTP 409 on /act: not initialized")

    c = make_client([http_err, {"type": "ok"}])
    with pytest.raises(cl.BridgeError, match="HTTP 409"):
        c.reset()
    assert len(c.calls) == 1


def test_dead_server_process_short_circuits_retries():
    c = make_client([conn_err, conn_err, conn_err], process=FakeProcess(exit_code=None))
    c.process.exit_code = 137  # dies after construction
    with pytest.raises(cl.BridgeError, match="exited with code 137"):
        c.reset()
    assert c.calls == []


def test_wait_until_ready_reports_server_death():
    c = make_client([conn_err] * 10, process=FakeProcess(exit_code=1))
    with pytest.raises(cl.BridgeError, match="exited with code 1 during startup"):
        c.wait_until_ready(timeout=5)


@pytest.mark.parametrize(
    "action,msg",
    [
        (None, "without an action"),
        (np.zeros(7, np.float32), "shape"),
        (np.array([np.nan] * 8, np.float32), "non-finite"),
    ],
)
def test_validate_action_rejects_bad_answers(action, msg):
    with pytest.raises(cl.BridgeError, match=msg):
        cl.validate_action(action, 8)
