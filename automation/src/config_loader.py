import shutil
from dataclasses import dataclass, field
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
from automation.src.llm.model_profile import ModelProfile, RerankerProfile


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
    enabled: bool
    model: str = ""  # resolved deployment name (e.g. Cohere rerank model), not a config key
    candidate_pool_size: int = 30
    top_k: int = 5


@dataclass
class RetrievalConfig:
    top_k: int
    hybrid_bm25: HybridBM25Config
    reranker: RerankerConfig
    evidence_verification_enabled: bool
    enabled: bool = True


@dataclass
class SlidingWindowConfig:
    window_sentences: int = 40
    overlap_sentences: int = 10
    prompt_version: str = "sliding_window_v1"


EVALUATION_METHODS = ("rag", "sliding_window")


@dataclass
class ResolvedPipelineConfig:
    experiment_name: str
    translation_model: ModelProfile
    evaluation_model: ModelProfile
    discrepancy_diagnosis_model: ModelProfile
    translation_prompt_path: Path
    evaluation_criteria_dir: Path
    evaluation_template_path: Path
    discrepancy_diagnosis_template_path: Path
    paths: PipelinePaths
    concurrency: ConcurrencyConfig
    chunking: ChunkingConfig
    storage: StorageConfig
    retrieval: RetrievalConfig
    pipeline_config_path: Path
    model_config_path: Path
    evaluation_method: str = "rag"
    sliding_window: SlidingWindowConfig = field(default_factory=SlidingWindowConfig)
    sliding_window_template_path: Path = field(
        default_factory=lambda: PROMPTS_ROOT / "quality_eval" / "sliding_window_v1" / "prompt_template.txt"
    )


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


def parse_reranker_profile(name: str, raw: dict) -> RerankerProfile:
    if not raw:
        raise ValueError(f"Reranker profile '{name}' is empty.")
    deployment = raw.get("deployment")
    if not deployment:
        raise ValueError(f"Reranker profile '{name}' missing deployment.")
    return RerankerProfile(name=name, deployment=str(deployment))


def load_reranker_profiles(model_config_path: Path) -> dict[str, RerankerProfile]:
    data = load_yaml(model_config_path)
    rerankers = data.get("rerankers", {})
    return {name: parse_reranker_profile(name, raw) for name, raw in rerankers.items()}


def get_reranker_profile(profiles: dict[str, RerankerProfile], key: str) -> RerankerProfile:
    if key not in profiles:
        available = ", ".join(sorted(profiles))
        raise ValueError(f"Unknown reranker model key '{key}'. Available: {available}")
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


def parse_storage_config(raw: dict | None, *, rag_enabled: bool) -> StorageConfig:
    raw = raw or {}

    batch_size = int(raw.get("batch_size", 16))
    if batch_size < 1:
        raise ValueError("evaluation.rag.storage.batch_size must be >= 1")

    return StorageConfig(enabled=rag_enabled, batch_size=batch_size)


def parse_retrieval_config(
    raw: dict | None,
    *,
    rag_enabled: bool,
    reranker_profiles: dict[str, RerankerProfile] | None = None,
) -> RetrievalConfig:
    raw = raw or {}

    top_k = int(raw.get("top_k", 10))
    if top_k < 1:
        raise ValueError("evaluation.rag.retrieval.top_k must be >= 1")

    bm25_raw = raw.get("hybrid_bm25") or {}
    reranker_raw = raw.get("reranker") or {}
    verification_raw = raw.get("evidence_verification") or {}

    reranker_enabled = bool(reranker_raw.get("enabled", False))

    reranker_top_k = int(reranker_raw.get("top_k", top_k))
    if reranker_top_k < 1:
        raise ValueError("evaluation.rag.retrieval.reranker.top_k must be >= 1")

    candidate_pool_size = int(reranker_raw.get("candidate_pool_size", top_k * 3))
    if candidate_pool_size < reranker_top_k:
        raise ValueError(
            "evaluation.rag.retrieval.reranker.candidate_pool_size must be >= reranker.top_k"
        )

    reranker_deployment = ""
    if reranker_enabled:
        reranker_model_key = reranker_raw.get("model")
        if not reranker_model_key:
            raise ValueError(
                "evaluation.rag.retrieval.reranker.model must be set (a key in model_config.yaml's "
                "rerankers:) when reranker.enabled is true"
            )
        reranker_deployment = get_reranker_profile(reranker_profiles or {}, reranker_model_key).deployment

    return RetrievalConfig(
        top_k=top_k,
        hybrid_bm25=HybridBM25Config(
            enabled=bool(bm25_raw.get("enabled", False)),
            rrf_k=int(bm25_raw.get("rrf_k", 60)),
        ),
        reranker=RerankerConfig(
            enabled=reranker_enabled,
            model=reranker_deployment,
            candidate_pool_size=candidate_pool_size,
            top_k=reranker_top_k,
        ),
        evidence_verification_enabled=bool(verification_raw.get("enabled", True)),
        enabled=rag_enabled,
    )


def parse_sliding_window_config(raw: dict | None) -> SlidingWindowConfig:
    raw = raw or {}

    window_sentences = int(raw.get("window_sentences", 40))
    if window_sentences < 1:
        raise ValueError("evaluation.sliding_window.window_sentences must be >= 1")

    overlap_sentences = int(raw.get("overlap_sentences", 10))
    if overlap_sentences < 0:
        raise ValueError("evaluation.sliding_window.overlap_sentences must be >= 0")
    if overlap_sentences >= window_sentences:
        raise ValueError(
            "evaluation.sliding_window.overlap_sentences must be < window_sentences"
        )

    return SlidingWindowConfig(
        window_sentences=window_sentences,
        overlap_sentences=overlap_sentences,
        prompt_version=str(raw.get("prompt_version", "sliding_window_v1")),
    )


def evaluation_output_names(method: str) -> tuple[str, str]:
    """Map the active evaluation method to its (evaluation_dir, comparison_dir)
    names, so both methods can be run against the same run_id without one
    overwriting the other's results."""
    if method == "sliding_window":
        return "evaluation_sliding_window", "comparison_sliding_window"
    return "evaluation", "comparison"


def resolve_prompt_paths(
    translation_version: str,
    evaluation_version: str,
    discrepancy_diagnosis_version: str,
) -> tuple[Path, Path, Path, Path]:
    translation_prompt = PROMPTS_ROOT / "translation" / translation_version / "prompt.txt"
    evaluation_dir = PROMPTS_ROOT / "quality_eval" / evaluation_version
    evaluation_template = evaluation_dir / "prompt_template.txt"
    discrepancy_diagnosis_template = (
        PROMPTS_ROOT / "discrepancy_diagnosis" / discrepancy_diagnosis_version / "prompt_template.txt"
    )

    if not translation_prompt.exists():
        raise FileNotFoundError(f"Translation prompt not found: {translation_prompt}")
    if not evaluation_dir.is_dir():
        raise FileNotFoundError(f"Evaluation prompts dir not found: {evaluation_dir}")
    if not evaluation_template.exists():
        raise FileNotFoundError(f"Evaluation template not found: {evaluation_template}")
    if not discrepancy_diagnosis_template.exists():
        raise FileNotFoundError(
            f"Discrepancy diagnosis template not found: {discrepancy_diagnosis_template}"
        )

    return translation_prompt, evaluation_dir, evaluation_template, discrepancy_diagnosis_template


def resolve_sliding_window_prompt_path(version: str) -> Path:
    template_path = PROMPTS_ROOT / "quality_eval" / version / "prompt_template.txt"
    if not template_path.exists():
        raise FileNotFoundError(f"Sliding window prompt template not found: {template_path}")
    return template_path


def load_pipeline_config(
    pipeline_config_path: Path | None = None,
    model_config_path: Path | None = None,
) -> ResolvedPipelineConfig:
    pipeline_path = pipeline_config_path or DEFAULT_PIPELINE_CONFIG
    model_path = model_config_path or DEFAULT_MODEL_CONFIG

    pipeline_data = load_yaml(pipeline_path)
    profiles = load_model_profiles(model_path)
    reranker_profiles = load_reranker_profiles(model_path)

    translation_cfg = pipeline_data.get("translation", {})
    evaluation_cfg = pipeline_data.get("evaluation", {})
    diagnosis_cfg = pipeline_data.get("discrepancy_diagnosis", {})
    paths_cfg = pipeline_data.get("paths", {})

    translation_model_key = translation_cfg.get("model")
    evaluation_model_key = evaluation_cfg.get("model")
    diagnosis_model_key = diagnosis_cfg.get("model")
    if not translation_model_key or not evaluation_model_key:
        raise ValueError("pipeline_config must define translation.model and evaluation.model")
    if not diagnosis_model_key:
        raise ValueError("pipeline_config must define discrepancy_diagnosis.model")

    translation_version = translation_cfg.get("prompt_version", "v1")
    evaluation_version = evaluation_cfg.get("prompt_version", "v1")
    diagnosis_version = diagnosis_cfg.get("prompt_version", "v1")

    translation_prompt, evaluation_dir, evaluation_template, diagnosis_template = resolve_prompt_paths(
        translation_version,
        evaluation_version,
        diagnosis_version,
    )

    rag_cfg = evaluation_cfg.get("rag") or {}
    rag_enabled = bool(rag_cfg.get("enabled", True))

    evaluation_method = str(evaluation_cfg.get("method", "rag"))
    if evaluation_method not in EVALUATION_METHODS:
        raise ValueError(
            f"evaluation.method must be one of {EVALUATION_METHODS}, got '{evaluation_method}'"
        )

    sliding_window = parse_sliding_window_config(evaluation_cfg.get("sliding_window"))
    sliding_window_template_path = resolve_sliding_window_prompt_path(sliding_window.prompt_version)

    return ResolvedPipelineConfig(
        experiment_name=pipeline_data.get("experiment_name", "unnamed"),
        translation_model=get_model_profile(profiles, translation_model_key),
        evaluation_model=get_model_profile(profiles, evaluation_model_key),
        discrepancy_diagnosis_model=get_model_profile(profiles, diagnosis_model_key),
        translation_prompt_path=translation_prompt,
        evaluation_criteria_dir=evaluation_dir,
        evaluation_template_path=evaluation_template,
        discrepancy_diagnosis_template_path=diagnosis_template,
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
        storage=parse_storage_config(
            rag_cfg.get("storage"), rag_enabled=rag_enabled and evaluation_method == "rag"
        ),
        retrieval=parse_retrieval_config(
            rag_cfg.get("retrieval"), rag_enabled=rag_enabled, reranker_profiles=reranker_profiles
        ),
        pipeline_config_path=pipeline_path.resolve(),
        model_config_path=model_path.resolve(),
        evaluation_method=evaluation_method,
        sliding_window=sliding_window,
        sliding_window_template_path=sliding_window_template_path,
    )


def get_run_dir(run_id: str) -> Path:
    return DEFAULT_DATA_ROOT / run_id


def results_dir(run_dir: Path, name: str) -> Path:
    """Path to a main-result subfolder (cleaned_text, translation, evaluation,
    comparison, diagnosis) — final deliverables of the pipeline."""
    return run_dir / "results" / name


def mid_product_dir(run_dir: Path, name: str) -> Path:
    """Path to a mid-product subfolder (chunks, rag_store, rag_candidates) —
    intermediate artifacts consumed by later pipeline stages."""
    return run_dir / "mid_product" / name


def _resolve_existing_or_new(run_dir: Path, name: str, new_path: Path) -> Path:
    if new_path.is_dir():
        return new_path
    old_path = run_dir / name
    if old_path.is_dir():
        return old_path
    return new_path


def resolve_results_dir(run_dir: Path, name: str) -> Path:
    """Resolve a main-result subfolder, for either a new or a pre-existing run.

    Prefers results/<name> (the current layout); falls back to the
    pre-refactor flat run_dir/<name> layout when that's what a given run
    already has on disk, so runs created before results/mid_product/ was
    introduced keep working without being physically migrated. When neither
    exists yet (a brand new run), returns the results/<name> path.
    """
    return _resolve_existing_or_new(run_dir, name, results_dir(run_dir, name))


def resolve_mid_product_dir(run_dir: Path, name: str) -> Path:
    """Same as resolve_results_dir, but for mid_product/<name> subfolders."""
    return _resolve_existing_or_new(run_dir, name, mid_product_dir(run_dir, name))


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
