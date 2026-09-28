"""The default YAMLs must resolve to exactly the same config after any refactor."""

from automation.src.config_loader import load_pipeline_config, load_model_profiles, load_reranker_profiles
from automation.src.constants import (
    DEFAULT_MODEL_CONFIG,
    DEFAULT_PIPELINE_CONFIG,
    DEFAULT_STEPS,
    OPTIONAL_STEPS,
    VALID_STEPS,
)

from ._serialize import assert_matches_golden, to_jsonable


def test_resolved_default_config_is_unchanged():
    config = load_pipeline_config(DEFAULT_PIPELINE_CONFIG, DEFAULT_MODEL_CONFIG)
    data = to_jsonable(config)
    data.pop("pipeline_config_path", None)
    data.pop("model_config_path", None)
    assert_matches_golden("resolved_config", data)


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
