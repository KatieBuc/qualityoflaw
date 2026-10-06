import shutil
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from automation.src.chunking import SAFE_LIMIT_DEFAULT
from automation.src.markdown.chunking import TARGET_CHARS_DEFAULT
from automation.src.markdown.policy_files import CLEANED_MD_SUFFIX
from automation.src.constants import (
    AUTOMATION_ROOT,
    CHUNKING_FALLBACK_PROMPT,
    MARKDOWN_FALLBACK_PROMPT,
    DEFAULT_PROJECT,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_PIPELINE_CONFIG,
    DEFAULT_SMALL_SCALE_STEMS,
    PROJECT_ROOT,
    PROMPTS_ROOT,
)
from automation.src.llm.confidence import (
    AVAILABLE_METHODS,
    METHOD_LOGPROBS,
    METHOD_MARGIN,
    ConfidenceSpec,
)
from automation.src.llm.model_profile import ModelProfile, RerankerProfile
from automation.src.paths import project_dirs


@dataclass
class PipelinePaths:
    input_dir: Path
    golden_csv: Path
    index_schema: Path
    # The curated Markdown corpus the `translation_md` step reads. Separate
    # from `input_dir`, which stays pointed at the raw OCR text: `diagnose`
    # still needs the Indonesian original as plain text, and the raw-text
    # translation path stays runnable side by side with the Markdown one.
    markdown_input_dir: Path = field(
        default_factory=lambda: project_dirs(DEFAULT_PROJECT).processed_cleaned_markdown
    )
    # File suffix identifying a source document in `markdown_input_dir`
    # (glob is `*<suffix>`). The curated Indonesian corpus names files
    # `<POLICY>.cleaned.md`; a differently-named corpus (e.g. one with plain
    # `<name>.md` files) overrides this via `paths.markdown_input_suffix`.
    markdown_input_suffix: str = CLEANED_MD_SUFFIX
    # Bare policy names (no suffix) that `--small-scale` selects, for every
    # step. Lives here rather than in a constant because the benchmark
    # documents belong to the corpus: localpolicies and globallaws share no
    # filenames. Each step derives its own filename from the stem (`<stem>.txt`
    # for the plain-text artifacts, `<stem>.md` for the Markdown ones).
    small_scale_stems: tuple[str, ...] = DEFAULT_SMALL_SCALE_STEMS
    project: str = DEFAULT_PROJECT
    # Manual indicator corrections applied by the comparison step.
    manual_overwrites: Path = field(
        default_factory=lambda: project_dirs(DEFAULT_PROJECT).manual_overwrites
    )


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
class MarkdownTranslationConfig:
    """Settings for the Markdown translation path.

    Deliberately has no `model` of its own -- it reuses
    `translation.model` / `translation_qa.model`, so switching models stays a
    one-place change. `target_chars` is what packs heading sections into
    translation units; `safe_limit` still guards the model's context and
    triggers sentence-level fallback splitting.
    """

    prompt_path: Path
    fallback_prompt_path: Path
    qa_template_path: Path
    target_chars: int = TARGET_CHARS_DEFAULT
    safe_limit: int = SAFE_LIMIT_DEFAULT
    # How many times translation_qa_md may re-audit one chunk before giving
    # up and keeping its last correction.
    qa_max_passes: int = 5


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


#: Evaluation-side confidence capture. Which methods to record and which one
#: leads; see `llm/confidence.py` for what each one measures.
ConfidenceConfig = ConfidenceSpec


@dataclass
class ConfidenceReportConfig:
    """Comparison-side cutoffs for the confidence report.

    Both absolute and quantile cutoffs are reported: logprob confidence on a
    binary answer tends to sit near 1.0, where absolute thresholds can flag
    nothing at all, while a quantile always yields a non-empty flagged set.
    """

    thresholds: tuple[float, ...] = (0.5, 0.7, 0.9, 0.95, 0.99)
    quantiles: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30)
    primary_threshold: float = 0.9
    flagged_sample_size: int = 100
    calibration_bins: int = 10


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
    translation_qa_model: ModelProfile
    evaluation_model: ModelProfile
    discrepancy_diagnosis_model: ModelProfile
    translation_prompt_path: Path
    translation_qa_template_path: Path
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
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    comparison_confidence: ConfidenceReportConfig = field(
        default_factory=ConfidenceReportConfig
    )
    sliding_window: SlidingWindowConfig = field(default_factory=SlidingWindowConfig)
    sliding_window_template_path: Path = field(
        default_factory=lambda: PROMPTS_ROOT / "quality_eval" / "sliding_window_v1" / "prompt_template.txt"
    )
    markdown: MarkdownTranslationConfig = field(
        default_factory=lambda: MarkdownTranslationConfig(
            prompt_path=PROMPTS_ROOT / "translation" / "v3" / "prompt.txt",
            fallback_prompt_path=MARKDOWN_FALLBACK_PROMPT,
            qa_template_path=PROMPTS_ROOT / "translation_qa" / "v2" / "prompt_template.txt",
        )
    )


def _require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _require_dir(path: Path, label: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


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
        supports_logprobs=bool(raw.get("supports_logprobs", True)),
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


def parse_small_scale_stems(raw: list | None) -> tuple[str, ...]:
    """`paths.small_scale_stems`: bare policy names, defaulting to the
    localpolicies benchmark set when omitted."""
    if raw is None:
        return DEFAULT_SMALL_SCALE_STEMS
    if not isinstance(raw, list) or not raw or not all(isinstance(s, str) and s for s in raw):
        raise ValueError("paths.small_scale_stems must be a non-empty list of policy names")
    return tuple(raw)


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


def parse_confidence_config(raw: dict | None) -> ConfidenceConfig:
    raw = raw or {}

    top_logprobs = int(raw.get("top_logprobs", 5))
    if not 1 <= top_logprobs <= 20:
        raise ValueError("evaluation.confidence.top_logprobs must be between 1 and 20")

    raw_methods = raw.get("methods")
    if raw_methods is None:
        methods = (METHOD_LOGPROBS, METHOD_MARGIN)
    else:
        if not isinstance(raw_methods, (list, tuple)):
            raise ValueError("evaluation.confidence.methods must be a list")
        methods = tuple(dict.fromkeys(str(method) for method in raw_methods))
        unknown = [method for method in methods if method not in AVAILABLE_METHODS]
        if unknown:
            raise ValueError(
                f"Unknown evaluation.confidence.methods: {', '.join(unknown)}. "
                f"Available: {', '.join(AVAILABLE_METHODS)}"
            )

    primary = str(raw.get("primary", methods[0] if methods else METHOD_LOGPROBS))
    if methods and primary not in methods:
        raise ValueError(
            f"evaluation.confidence.primary '{primary}' must be one of the enabled "
            f"methods: {', '.join(methods)}"
        )

    return ConfidenceConfig(
        enabled=bool(raw.get("enabled", True)) and bool(methods),
        methods=methods,
        primary=primary,
        top_logprobs=top_logprobs,
    )


def _parse_cutoffs(values: object, key: str, *, exclusive_upper: bool) -> tuple[float, ...]:
    if values is None:
        return ()
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{key} must be a list of numbers")
    cutoffs = []
    for value in values:
        cutoff = float(value)
        upper_ok = cutoff < 1.0 if exclusive_upper else cutoff <= 1.0
        if not (0.0 < cutoff and upper_ok):
            bound = "0 < x < 1" if exclusive_upper else "0 < x <= 1"
            raise ValueError(f"{key} entries must satisfy {bound}, got {cutoff}")
        cutoffs.append(cutoff)
    return tuple(sorted(set(cutoffs)))


def parse_confidence_report_config(raw: dict | None) -> ConfidenceReportConfig:
    raw = raw or {}
    defaults = ConfidenceReportConfig()

    thresholds = (
        _parse_cutoffs(raw.get("thresholds"), "comparison.confidence.thresholds", exclusive_upper=False)
        or defaults.thresholds
    )
    quantiles = (
        _parse_cutoffs(raw.get("quantiles"), "comparison.confidence.quantiles", exclusive_upper=True)
        or defaults.quantiles
    )

    primary_threshold = float(raw.get("primary_threshold", defaults.primary_threshold))
    if not 0.0 < primary_threshold <= 1.0:
        raise ValueError("comparison.confidence.primary_threshold must satisfy 0 < x <= 1")

    flagged_sample_size = int(raw.get("flagged_sample_size", defaults.flagged_sample_size))
    if flagged_sample_size < 0:
        raise ValueError("comparison.confidence.flagged_sample_size must be >= 0")

    calibration_bins = int(raw.get("calibration_bins", defaults.calibration_bins))
    if calibration_bins < 2:
        raise ValueError("comparison.confidence.calibration_bins must be >= 2")

    return ConfidenceReportConfig(
        thresholds=thresholds,
        quantiles=quantiles,
        primary_threshold=primary_threshold,
        flagged_sample_size=flagged_sample_size,
        calibration_bins=calibration_bins,
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


def parse_markdown_config(raw: dict | None) -> MarkdownTranslationConfig:
    """Resolve the `translation_markdown:` block, defaults included.

    Prompt versions resolve the same way the raw-text path's do, so a missing
    prompt file fails loudly at config-load time rather than mid-run.
    """
    raw = raw or {}
    chunking = raw.get("chunking") or {}

    prompt_version = str(raw.get("prompt_version", "v3"))
    qa_version = str(raw.get("qa_prompt_version", "v2"))
    prompt_path = _require_file(
        PROMPTS_ROOT / "translation" / prompt_version / "prompt.txt",
        "Markdown translation prompt",
    )
    _require_file(MARKDOWN_FALLBACK_PROMPT, "Markdown fallback prompt")
    qa_template_path = _require_file(
        PROMPTS_ROOT / "translation_qa" / qa_version / "prompt_template.txt",
        "Markdown translation QA prompt",
    )

    return MarkdownTranslationConfig(
        prompt_path=prompt_path,
        fallback_prompt_path=MARKDOWN_FALLBACK_PROMPT,
        qa_template_path=qa_template_path,
        target_chars=int(chunking.get("target_chars", TARGET_CHARS_DEFAULT)),
        safe_limit=int(chunking.get("safe_limit", SAFE_LIMIT_DEFAULT)),
        qa_max_passes=max(1, int(raw.get("qa_max_passes", 5))),
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
    translation_qa_version: str,
) -> tuple[Path, Path, Path, Path, Path]:
    translation_prompt = _require_file(
        PROMPTS_ROOT / "translation" / translation_version / "prompt.txt",
        "Translation prompt",
    )
    evaluation_dir = _require_dir(
        PROMPTS_ROOT / "quality_eval" / evaluation_version, "Evaluation prompts dir"
    )
    evaluation_template = _require_file(
        evaluation_dir / "prompt_template.txt", "Evaluation template"
    )
    discrepancy_diagnosis_template = _require_file(
        PROMPTS_ROOT / "discrepancy_diagnosis" / discrepancy_diagnosis_version / "prompt_template.txt",
        "Discrepancy diagnosis template",
    )
    translation_qa_template = _require_file(
        PROMPTS_ROOT / "translation_qa" / translation_qa_version / "prompt_template.txt",
        "Translation QA template",
    )

    return (
        translation_prompt,
        evaluation_dir,
        evaluation_template,
        discrepancy_diagnosis_template,
        translation_qa_template,
    )


def resolve_sliding_window_prompt_path(version: str) -> Path:
    return _require_file(
        PROMPTS_ROOT / "quality_eval" / version / "prompt_template.txt",
        "Sliding window prompt template",
    )


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
    translation_qa_cfg = pipeline_data.get("translation_qa", {})
    evaluation_cfg = pipeline_data.get("evaluation", {})
    diagnosis_cfg = pipeline_data.get("discrepancy_diagnosis", {})
    paths_cfg = pipeline_data.get("paths") or {}
    project = str(pipeline_data.get("project") or DEFAULT_PROJECT)
    set_active_project(project)
    dirs = project_dirs(project)

    translation_model_key = translation_cfg.get("model")
    translation_qa_model_key = translation_qa_cfg.get("model")
    evaluation_model_key = evaluation_cfg.get("model")
    diagnosis_model_key = diagnosis_cfg.get("model")
    if not translation_model_key or not evaluation_model_key:
        raise ValueError("pipeline_config must define translation.model and evaluation.model")
    if not diagnosis_model_key:
        raise ValueError("pipeline_config must define discrepancy_diagnosis.model")
    if not translation_qa_model_key:
        raise ValueError("pipeline_config must define translation_qa.model")

    translation_version = translation_cfg.get("prompt_version", "v1")
    translation_qa_version = translation_qa_cfg.get("prompt_version", "v1")
    evaluation_version = evaluation_cfg.get("prompt_version", "v1")
    diagnosis_version = diagnosis_cfg.get("prompt_version", "v1")

    (
        translation_prompt,
        evaluation_dir,
        evaluation_template,
        diagnosis_template,
        translation_qa_template,
    ) = resolve_prompt_paths(
        translation_version,
        evaluation_version,
        diagnosis_version,
        translation_qa_version,
    )

    markdown_cfg = pipeline_data.get("translation_markdown") or {}
    markdown = parse_markdown_config(markdown_cfg)

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
        translation_qa_model=get_model_profile(profiles, translation_qa_model_key),
        evaluation_model=get_model_profile(profiles, evaluation_model_key),
        discrepancy_diagnosis_model=get_model_profile(profiles, diagnosis_model_key),
        translation_prompt_path=translation_prompt,
        translation_qa_template_path=translation_qa_template,
        evaluation_criteria_dir=evaluation_dir,
        evaluation_template_path=evaluation_template,
        discrepancy_diagnosis_template_path=diagnosis_template,
        paths=PipelinePaths(
            input_dir=_resolve_path(paths_cfg.get("input_dir", dirs.raw)),
            golden_csv=_resolve_path(paths_cfg.get("golden_csv", dirs.golden_csv)),
            index_schema=_resolve_path(paths_cfg.get("index_schema", dirs.index_schema)),
            markdown_input_dir=_resolve_path(
                paths_cfg.get("markdown_input_dir", dirs.processed_cleaned_markdown)
            ),
            markdown_input_suffix=str(paths_cfg.get("markdown_input_suffix", CLEANED_MD_SUFFIX)),
            small_scale_stems=parse_small_scale_stems(paths_cfg.get("small_scale_stems")),
            project=project,
            manual_overwrites=_resolve_path(
                paths_cfg.get("manual_overwrites", dirs.manual_overwrites)
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
        confidence=parse_confidence_config(evaluation_cfg.get("confidence")),
        comparison_confidence=parse_confidence_report_config(
            (pipeline_data.get("comparison") or {}).get("confidence")
        ),
        sliding_window=sliding_window,
        sliding_window_template_path=sliding_window_template_path,
        markdown=markdown,
    )


# Runs root override. None means "derive from the active project"
# (data/<project>/automation); tests point it at a tmp dir.
DEFAULT_DATA_ROOT: Path | None = None
_active_project: str = DEFAULT_PROJECT


def set_active_project(project: str) -> None:
    """Called by `load_pipeline_config`, so every `get_run_dir` caller follows
    the `project:` config key without threading it through each step."""
    global _active_project
    _active_project = project


def get_run_dir(run_id: str) -> Path:
    root = DEFAULT_DATA_ROOT or project_dirs(_active_project).automation
    return root / run_id


def results_dir(run_dir: Path, name: str) -> Path:
    """Path to a main-result subfolder (cleaned_text, cleaned_markdown,
    translation, translation_markdown, evaluation, comparison, diagnosis) —
    final deliverables of the pipeline."""
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
