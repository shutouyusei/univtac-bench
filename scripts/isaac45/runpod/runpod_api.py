"""Minimal client for RunPod's REST API (https://rest.runpod.io/v1): create, get, list, stop and delete pods.

Standard library only. The API key comes from the caller (launch.py / collect.py read RUNPOD_API_KEY).
"""

import json
import urllib.error
import urllib.parse
import urllib.request

BASE_URL = "https://rest.runpod.io/v1"


class RunPodError(RuntimeError):
    pass


class RunPod:
    def __init__(self, api_key: str, base_url: str = BASE_URL, opener=urllib.request.urlopen, timeout: float = 60):
        if not api_key:
            raise RunPodError("no RunPod API key (set RUNPOD_API_KEY)")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.opener = opener
        self.timeout = timeout

    def _call(self, method: str, path: str, body=None, query=None):
        url = f"{self.base_url}{path}"
        if query:
            url += "?" + urllib.parse.urlencode(query)
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(url, data=data, method=method)
        request.add_header("Authorization", f"Bearer {self.api_key}")
        if data is not None:
            request.add_header("Content-Type", "application/json")
        try:
            with self.opener(request, timeout=self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:500]
            raise RunPodError(f"{method} {path}: HTTP {error.code} {detail}") from error
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            # the request may still have reached RunPod (a pod may exist); callers check before retrying a create
            raise RunPodError(f"{method} {path}: no response ({error})") from error
        try:
            return json.loads(raw) if raw.strip() else None
        except json.JSONDecodeError as error:
            raise RunPodError(f"{method} {path}: unreadable response {raw[:200]!r}") from error

    def create_pod(self, spec: dict) -> dict:
        return self._call("POST", "/pods", body=spec)

    def get_pod(self, pod_id: str):
        """The pod, or None once RunPod no longer knows it (deleted)."""
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
    """(public ip, public port of the pod's port 22), or None while the pod is still starting."""
    ip = pod.get("publicIp")
    port = (pod.get("portMappings") or {}).get("22")
    if not ip or not port:
        return None
    return ip, int(port)
