"""The pinned YAML must resolve to exactly the same config after any refactor."""

from automation.src.config_loader import load_pipeline_config, load_model_profiles, load_reranker_profiles
from automation.src.constants import (
    DEFAULT_MODEL_CONFIG,
    DEFAULT_STEPS,
    OPTIONAL_STEPS,
    VALID_STEPS,
)

from ._serialize import GOLDEN_PIPELINE_CONFIG, assert_matches_golden, to_jsonable


def test_resolved_default_config_is_unchanged():
    config = load_pipeline_config(GOLDEN_PIPELINE_CONFIG, DEFAULT_MODEL_CONFIG)
    data = to_jsonable(config)
    data.pop("pipeline_config_path", None)
    data.pop("model_config_path", None)
    assert_matches_golden("resolved_config", data)


def test_the_live_default_configs_still_load():
    # not a snapshot: the live YAMLs change with the project being run, but they must always load
    from automation.src.constants import DEFAULT_PIPELINE_CONFIG

    config = load_pipeline_config(DEFAULT_PIPELINE_CONFIG, DEFAULT_MODEL_CONFIG)
    assert config.paths.markdown_input_dir.name == "cleaned_markdown"


def test_model_and_reranker_profiles_are_unchanged():
    data = {
        "models": to_jsonable(load_model_profiles(DEFAULT_MODEL_CONFIG)),
        "rerankers": to_jsonable(load_reranker_profiles(DEFAULT_MODEL_CONFIG)),
    }
    assert_matches_golden("model_profiles", data)


def test_step_names_are_unchanged():
    assert_matches_golden(
        "step_names",
        {
            "default": list(DEFAULT_STEPS),
            "optional": list(OPTIONAL_STEPS),
            "valid": list(VALID_STEPS),
        },
    )
