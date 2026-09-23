"""lerobot bridge: SmolVLA / tacforcing trained in the lerobot env, evaluated in UniVTAC via server.py.

``Policy`` is imported lazily so the converter, server and tests can import
this package without pulling in the Isaac-side adapter.
"""


def __getattr__(name):
    if name == "Policy":
        from .deploy_policy import Policy

        return Policy
    raise AttributeError(name)
