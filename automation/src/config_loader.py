import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

from automation.src.constants import (
    AUTOMATION_ROOT,
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
class ResolvedPipelineConfig:
    experiment_name: str
    translation_model: ModelProfile
    evaluation_model: ModelProfile
    translation_prompt_path: Path
    evaluation_criteria_dir: Path
    evaluation_template_path: Path
    paths: PipelinePaths
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
