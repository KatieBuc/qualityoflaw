from dataclasses import dataclass


@dataclass
class ModelProfile:
    name: str
    deployment: str
    temperature: float
    max_tokens: int | None = None
    max_retries: int = 3


@dataclass
class RerankerProfile:
    name: str
    deployment: str
