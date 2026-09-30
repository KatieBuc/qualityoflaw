import pytest
import yaml

from automation.src.config_loader import (
    get_reranker_profile,
    load_reranker_profiles,
    parse_reranker_profile,
    parse_retrieval_config,
)
from automation.src.llm.model_profile import RerankerProfile

_PROFILES = {"cohere-rerank-v3.5": RerankerProfile(name="cohere-rerank-v3.5", deployment="rerank-v3.5")}


def test_reranker_defaults_when_absent():
    config = parse_retrieval_config({"top_k": 10}, rag_enabled=True)
    assert config.reranker.enabled is False
    assert config.reranker.model == ""
    assert config.reranker.candidate_pool_size == 30
    assert config.reranker.top_k == 10


def test_reranker_resolves_model_key_to_deployment():
    config = parse_retrieval_config(
        {
            "top_k": 10,
            "reranker": {
                "enabled": True,
                "model": "cohere-rerank-v3.5",
                "candidate_pool_size": 40,
                "top_k": 5,
            },
        },
        rag_enabled=True,
        reranker_profiles=_PROFILES,
    )
    assert config.reranker.enabled is True
    assert config.reranker.model == "rerank-v3.5"
    assert config.reranker.candidate_pool_size == 40
    assert config.reranker.top_k == 5


def test_reranker_enabled_without_model_key_raises():
    with pytest.raises(ValueError, match="reranker.model must be set"):
        parse_retrieval_config(
            {"top_k": 10, "reranker": {"enabled": True}},
            rag_enabled=True,
            reranker_profiles=_PROFILES,
        )


def test_reranker_enabled_with_unknown_model_key_raises():
    with pytest.raises(ValueError, match="Unknown reranker model key"):
        parse_retrieval_config(
            {"top_k": 10, "reranker": {"enabled": True, "model": "does-not-exist"}},
            rag_enabled=True,
            reranker_profiles=_PROFILES,
        )


def test_reranker_disabled_does_not_require_model_key():
    config = parse_retrieval_config(
        {"top_k": 10, "reranker": {"enabled": False}},
        rag_enabled=True,
        reranker_profiles=_PROFILES,
    )
    assert config.reranker.enabled is False
    assert config.reranker.model == ""


def test_reranker_candidate_pool_size_must_be_at_least_top_k():
    with pytest.raises(ValueError, match="candidate_pool_size"):
        parse_retrieval_config(
            {"top_k": 10, "reranker": {"candidate_pool_size": 3, "top_k": 5}},
            rag_enabled=True,
        )


def test_reranker_top_k_must_be_positive():
    with pytest.raises(ValueError, match="reranker.top_k"):
        parse_retrieval_config(
            {"top_k": 10, "reranker": {"top_k": 0}},
            rag_enabled=True,
        )


def test_parse_reranker_profile_requires_deployment():
    with pytest.raises(ValueError, match="missing deployment"):
        parse_reranker_profile("cohere-rerank-v3.5", {"some_other_field": "x"})


def test_load_reranker_profiles_reads_rerankers_section(tmp_path):
    path = tmp_path / "model_config.yaml"
    path.write_text(
        yaml.dump({"rerankers": {"cohere-rerank-v3.5": {"deployment": "rerank-v3.5"}}}),
        encoding="utf-8",
    )
    profiles = load_reranker_profiles(path)
    assert profiles["cohere-rerank-v3.5"].deployment == "rerank-v3.5"


def test_load_reranker_profiles_empty_when_section_missing(tmp_path):
    path = tmp_path / "model_config.yaml"
    path.write_text(yaml.dump({"models": {}}), encoding="utf-8")
    assert load_reranker_profiles(path) == {}


def test_get_reranker_profile_unknown_key_raises():
    with pytest.raises(ValueError, match="Unknown reranker model key"):
        get_reranker_profile(_PROFILES, "bogus")
