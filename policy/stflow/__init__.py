"""stflow: a FlowPolicy checkpoint from the streaming-tactile-flow repo, run directly in the Isaac process."""


def __getattr__(name):
    if name == "Policy":
        from .deploy_policy import Policy

        return Policy
    raise AttributeError(name)
