import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI

ApiStyle = Literal["chat", "responses"]


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_env() -> None:
    load_dotenv(get_project_root() / ".env")


def _clean_env_value(value: str | None) -> str:
    if not value:
        return ""
    return value.split("#", 1)[0].strip()


def get_azure_model() -> str:
    _load_env()
    model = _clean_env_value(os.getenv("AZURE_OPENAI_MODEL"))
    if not model:
        raise RuntimeError("AZURE_OPENAI_MODEL is not set.")
    return model


def get_api_style() -> ApiStyle:
    _load_env()
    endpoint = _clean_env_value(os.getenv("AZURE_OPENAI_ENDPOINT", ""))
    if "/openai/v1" in endpoint:
        return "responses"
    return "chat"


def get_azure_client() -> OpenAI | AzureOpenAI:
    _load_env()
    api_key = _clean_env_value(os.environ.get("AZURE_OPENAI_API_KEY"))
    endpoint = _clean_env_value(os.environ.get("AZURE_OPENAI_ENDPOINT", ""))

    if not api_key or not endpoint:
        raise RuntimeError("AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT must be set.")

    if get_api_style() == "responses":
        return OpenAI(
            api_key=api_key,
            base_url=endpoint.rstrip("/") + "/",
        )

    return AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=_clean_env_value(
            os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
        ),
    )
