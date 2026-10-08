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
    # Reasoning-family deployments also reject `temperature`; set false in
    # model_config.yaml to stop sending it. Without that, the wrapper drops it
    # the first time a 400 says it is unsupported.
    supports_temperature: bool = True


@dataclass
class RerankerProfile:
    name: str
    deployment: str
