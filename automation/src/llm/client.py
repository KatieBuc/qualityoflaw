import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI

ApiStyle = Literal["chat", "responses"]

DEFAULT_API_TIMEOUT_S = 300.0


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_env() -> None:
    load_dotenv(get_project_root() / ".env")


def clean_env_value(value: str | None) -> str:
    if not value:
        return ""
    return value.split("#", 1)[0].strip()


def get_api_style() -> ApiStyle:
    _load_env()
    endpoint = clean_env_value(os.getenv("AZURE_OPENAI_ENDPOINT", ""))
    if "/openai/v1" in endpoint:
        return "responses"
    return "chat"


def get_azure_client() -> OpenAI | AzureOpenAI:
    _load_env()
    api_key = clean_env_value(os.environ.get("AZURE_OPENAI_API_KEY"))
    endpoint = clean_env_value(os.environ.get("AZURE_OPENAI_ENDPOINT", ""))

    if not api_key or not endpoint:
        raise RuntimeError("AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT must be set.")

    if get_api_style() == "responses":
        return OpenAI(
            api_key=api_key,
            base_url=endpoint.rstrip("/") + "/",
            max_retries=0,
            timeout=DEFAULT_API_TIMEOUT_S,
        )

    return AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=clean_env_value(os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")),
        max_retries=0,
        timeout=DEFAULT_API_TIMEOUT_S,
    )


def get_embedding_deployment() -> str:
    _load_env()
    deployment = clean_env_value(os.environ.get("AZURE_OPENAI_EMBEDDING_MODEL"))
    if not deployment:
        raise RuntimeError(
            "AZURE_OPENAI_EMBEDDING_MODEL must be set in .env for the storage/retrieval RAG step."
        )
    return deployment
