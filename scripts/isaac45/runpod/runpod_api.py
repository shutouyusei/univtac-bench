"""Minimal stdlib client for RunPod's REST API (pods: create, get, list, stop, delete)."""

import json
import urllib.error
import urllib.parse
import urllib.request

BASE_URL = "https://rest.runpod.io/v1"
USER_AGENT = "univtac-bench-runpod/1.0"  # RunPod's Cloudflare front rejects urllib's default (HTTP 403, code 1010)


class RunPodError(RuntimeError):
    pass


class RunPod:
    def __init__(self, api_key: str, opener=urllib.request.urlopen):
        if not api_key:
            raise RunPodError("no RunPod API key (set RUNPOD_API_KEY)")
        self.api_key = api_key
        self.opener = opener

    def _call(self, method: str, path: str, body=None, query=None):
        url = BASE_URL + path + ("?" + urllib.parse.urlencode(query) if query else "")
        data = None if body is None else json.dumps(body).encode()
        headers = {"Authorization": f"Bearer {self.api_key}", "User-Agent": USER_AGENT}
        if data is not None:
            headers["Content-Type"] = "application/json"
        try:
            with self.opener(urllib.request.Request(url, data=data, headers=headers, method=method),
                             timeout=60) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            raise RunPodError(f"{method} {path}: HTTP {error.code} {error.read().decode(errors='replace')[:500]}") \
                from error
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            # the request may still have reached RunPod: a create may have made a pod (launch.find_created checks)
            raise RunPodError(f"{method} {path}: no response ({error})") from error
        try:
            return json.loads(raw) if raw.strip() else None
        except json.JSONDecodeError as error:
            raise RunPodError(f"{method} {path}: unreadable response {raw[:200]!r}") from error

    def create_pod(self, spec: dict) -> dict:
        return self._call("POST", "/pods", body=spec)

    def get_pod(self, pod_id: str):
        """The pod, or None once it is deleted."""
        try:
            return self._call("GET", f"/pods/{pod_id}")
        except RunPodError as error:
            if "HTTP 404" in str(error):
                return None
            raise

    def list_pods(self, name: str = None) -> list:
        return self._call("GET", "/pods", query={"name": name} if name else None) or []

    def stop_pod(self, pod_id: str):
        return self._call("POST", f"/pods/{pod_id}/stop")

    def delete_pod(self, pod_id: str):
        return self._call("DELETE", f"/pods/{pod_id}")


def ssh_endpoint(pod: dict):
    """(public ip, public port of port 22), or None while the pod starts."""
    ip, port = pod.get("publicIp"), (pod.get("portMappings") or {}).get("22")
    return (ip, int(port)) if ip and port else None
