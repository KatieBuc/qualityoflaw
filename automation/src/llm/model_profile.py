from dataclasses import dataclass


@dataclass
class ModelProfile:
    name: str
    deployment: str
    temperature: float
    max_tokens: int | None = None
    max_retries: int = 3
    # Reasoning-family deployments reject `logprobs`; set false in
    # model_config.yaml to skip requesting it instead of relying on the
    # wrapper's runtime auto-degrade.
    supports_logprobs: bool = True


@dataclass
class RerankerProfile:
    name: str
    deployment: str
