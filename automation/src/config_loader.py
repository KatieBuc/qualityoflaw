import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

from automation.src.chunking import SAFE_LIMIT_DEFAULT
from automation.src.constants import (
    AUTOMATION_ROOT,
    CHUNKING_FALLBACK_PROMPT,
    DEFAULT_DATA_ROOT,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_PIPELINE_CONFIG,
    PROJECT_ROOT,
    PROMPTS_ROOT,
)
from automation.src.llm.model_profile import ModelProfile


@dataclass
class PipelinePaths:
    input_dir: Path
    golden_csv: Path
    index_schema: Path


@dataclass
class ConcurrencyConfig:
    enabled: bool
    max_workers: int


@dataclass
class ChunkingConfig:
    enabled: bool
    safe_limit: int
    fallback_prompt_path: Path


@dataclass
class StorageConfig:
    enabled: bool
    batch_size: int


@dataclass
class HybridBM25Config:
    enabled: bool
    rrf_k: int


@dataclass
class RerankerConfig:
    enabled: bool  # stub — see automation/src/rag/retriever.py:rerank()


@dataclass
class RetrievalConfig:
    top_k: int
    hybrid_bm25: HybridBM25Config
    reranker: RerankerConfig
    evidence_verification_enabled: bool


@dataclass
class ResolvedPipelineConfig:
    experiment_name: str
    translation_model: ModelProfile
    evaluation_model: ModelProfile
    translation_prompt_path: Path
    evaluation_criteria_dir: Path
    evaluation_template_path: Path
    paths: PipelinePaths
    concurrency: ConcurrencyConfig
    chunking: ChunkingConfig
    storage: StorageConfig
    retrieval: RetrievalConfig
    pipeline_config_path: Path
    model_config_path: Path


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_model_profile(name: str, raw: dict) -> ModelProfile:
    if not raw:
        raise ValueError(f"Model profile '{name}' is empty.")
    deployment = raw.get("deployment")
    if not deployment:
        raise ValueError(f"Model profile '{name}' missing deployment.")
    return ModelProfile(
        name=name,
        deployment=str(deployment),
        temperature=float(raw.get("temperature", 0.1)),
        max_tokens=raw.get("max_tokens"),
        max_retries=int(raw.get("max_retries", 3)),
    )


def load_model_profiles(model_config_path: Path) -> dict[str, ModelProfile]:
    data = load_yaml(model_config_path)
    models = data.get("models", {})
    if not models:
        raise ValueError(f"No models defined in {model_config_path}")
    return {name: parse_model_profile(name, raw) for name, raw in models.items()}


def get_model_profile(profiles: dict[str, ModelProfile], key: str) -> ModelProfile:
    if key not in profiles:
        available = ", ".join(sorted(profiles))
        raise ValueError(f"Unknown model key '{key}'. Available: {available}")
    return profiles[key]


def parse_concurrency_config(raw: dict | None) -> ConcurrencyConfig:
    if not raw:
        return ConcurrencyConfig(enabled=True, max_workers=5)

    max_workers = int(raw.get("max_workers", 5))
    if max_workers < 1:
        raise ValueError("concurrency.max_workers must be >= 1")

    return ConcurrencyConfig(
        enabled=bool(raw.get("enabled", True)),
        max_workers=max_workers,
    )


def parse_chunking_config(raw: dict | None) -> ChunkingConfig:
    if not raw:
        return ChunkingConfig(
            enabled=False,
            safe_limit=SAFE_LIMIT_DEFAULT,
            fallback_prompt_path=CHUNKING_FALLBACK_PROMPT,
        )

    enabled = bool(raw.get("enabled", False))
    safe_limit = int(raw.get("safe_limit", SAFE_LIMIT_DEFAULT))
    if safe_limit < 1:
        raise ValueError("translation.chunking.safe_limit must be >= 1")

    fallback_prompt_path = CHUNKING_FALLBACK_PROMPT
    if enabled and not fallback_prompt_path.exists():
        raise FileNotFoundError(f"Chunking fallback prompt not found: {fallback_prompt_path}")

    return ChunkingConfig(
        enabled=enabled,
        safe_limit=safe_limit,
        fallback_prompt_path=fallback_prompt_path,
    )


def parse_storage_config(raw: dict | None) -> StorageConfig:
    if not raw:
        return StorageConfig(enabled=True, batch_size=16)

    batch_size = int(raw.get("batch_size", 16))
    if batch_size < 1:
        raise ValueError("storage.batch_size must be >= 1")

    return StorageConfig(enabled=bool(raw.get("enabled", True)), batch_size=batch_size)


def parse_retrieval_config(raw: dict | None) -> RetrievalConfig:
    raw = raw or {}

    top_k = int(raw.get("top_k", 10))
    if top_k < 1:
        raise ValueError("retrieval.top_k must be >= 1")

    bm25_raw = raw.get("hybrid_bm25") or {}
    reranker_raw = raw.get("reranker") or {}
    verification_raw = raw.get("evidence_verification") or {}

    return RetrievalConfig(
        top_k=top_k,
        hybrid_bm25=HybridBM25Config(
            enabled=bool(bm25_raw.get("enabled", False)),
            rrf_k=int(bm25_raw.get("rrf_k", 60)),
        ),
        reranker=RerankerConfig(enabled=bool(reranker_raw.get("enabled", False))),
        evidence_verification_enabled=bool(verification_raw.get("enabled", True)),
    )


def resolve_prompt_paths(
    translation_version: str,
    evaluation_version: str,
) -> tuple[Path, Path, Path]:
    translation_prompt = PROMPTS_ROOT / "translation" / translation_version / "prompt.txt"
    evaluation_dir = PROMPTS_ROOT / "quality_eval" / evaluation_version
    evaluation_template = evaluation_dir / "prompt_template.txt"

    if not translation_prompt.exists():
        raise FileNotFoundError(f"Translation prompt not found: {translation_prompt}")
    if not evaluation_dir.is_dir():
        raise FileNotFoundError(f"Evaluation prompts dir not found: {evaluation_dir}")
    if not evaluation_template.exists():
        raise FileNotFoundError(f"Evaluation template not found: {evaluation_template}")

    return translation_prompt, evaluation_dir, evaluation_template


def load_pipeline_config(
    pipeline_config_path: Path | None = None,
    model_config_path: Path | None = None,
) -> ResolvedPipelineConfig:
    pipeline_path = pipeline_config_path or DEFAULT_PIPELINE_CONFIG
    model_path = model_config_path or DEFAULT_MODEL_CONFIG

    pipeline_data = load_yaml(pipeline_path)
    profiles = load_model_profiles(model_path)

    translation_cfg = pipeline_data.get("translation", {})
    evaluation_cfg = pipeline_data.get("evaluation", {})
    paths_cfg = pipeline_data.get("paths", {})

    translation_model_key = translation_cfg.get("model")
    evaluation_model_key = evaluation_cfg.get("model")
    if not translation_model_key or not evaluation_model_key:
        raise ValueError("pipeline_config must define translation.model and evaluation.model")

    translation_version = translation_cfg.get("prompt_version", "v1")
    evaluation_version = evaluation_cfg.get("prompt_version", "v1")

    translation_prompt, evaluation_dir, evaluation_template = resolve_prompt_paths(
        translation_version,
        evaluation_version,
    )

    return ResolvedPipelineConfig(
        experiment_name=pipeline_data.get("experiment_name", "unnamed"),
        translation_model=get_model_profile(profiles, translation_model_key),
        evaluation_model=get_model_profile(profiles, evaluation_model_key),
        translation_prompt_path=translation_prompt,
        evaluation_criteria_dir=evaluation_dir,
        evaluation_template_path=evaluation_template,
        paths=PipelinePaths(
            input_dir=_resolve_path(paths_cfg.get("input_dir", "data/raw/localpolicies")),
            golden_csv=_resolve_path(
                paths_cfg.get("golden_csv", "data/processed/long_policy_encoding.csv")
            ),
            index_schema=_resolve_path(
                paths_cfg.get("index_schema", "data/mapping/index_schema.yaml")
            ),
        ),
        concurrency=parse_concurrency_config(pipeline_data.get("concurrency")),
        chunking=parse_chunking_config(translation_cfg.get("chunking")),
        storage=parse_storage_config(pipeline_data.get("storage")),
        retrieval=parse_retrieval_config(pipeline_data.get("retrieval")),
        pipeline_config_path=pipeline_path.resolve(),
        model_config_path=model_path.resolve(),
    )


def get_run_dir(run_id: str) -> Path:
    return DEFAULT_DATA_ROOT / run_id


def snapshot_configs(
    run_dir: Path,
    pipeline_config_path: Path,
    model_config_path: Path,
) -> Path:
    config_dir = run_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pipeline_config_path, config_dir / "pipeline_config.yaml")
    shutil.copy2(model_config_path, config_dir / "model_config.yaml")
    return config_dir
