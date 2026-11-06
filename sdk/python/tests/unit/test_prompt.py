import json
from datetime import datetime, timedelta

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from feast import (
    Entity,
    FeatureStore,
    FeatureView,
    Field,
    FileSource,
    RepoConfig,
    RequestSource,
)
from feast.context import TokenBudgetExceededError
from feast.diff.property_diff import TransitionType
from feast.diff.registry_diff import apply_diff_to_registry, diff_between
from feast.errors import PromptNotFoundException
from feast.feature_server import get_app
from feast.infra.registry.registry import FeastObjectType, Registry
from feast.infra.registry.sql import SqlRegistry, SqlRegistryConfig
from feast.prompt import Prompt, prompt
from feast.repo_config import RegistryConfig
from feast.repo_contents import RepoContents
from feast.types import String

PROJECT = "prompt_test"
USER = Entity(name="user", join_keys=["user_id"])
LOCATION = RequestSource(
    name="location_request", schema=[Field(name="location", dtype=String)]
)


def _profile_view(path: str = "unused.parquet") -> FeatureView:
    return FeatureView(
        name="user_profile",
        entities=[USER],
        ttl=timedelta(days=1),
        schema=[Field(name="name", dtype=String), Field(name="visits", dtype=String)],
        source=FileSource(path=path, timestamp_field="event_timestamp"),
    )


def _concierge(profile, request):
    return f"You are serving {profile['name']} in {request['location']}."


def _friendlier_concierge(profile, request):
    return f"Hello! You are serving {profile['name']} in {request['location']}."


def _concierge_variables(profile, request):
    return {
        "name": profile["name"],
        "location": request["location"],
        "visits": profile["visits"],
    }


def _uses_a_module_import(profile, request):
    return json.dumps(profile)


def _build(function=_concierge, profile=None, **settings) -> Prompt:
    settings = {"tokenizer": "approximate", "name": "concierge", **settings}
    return prompt(sources=[profile or _profile_view(), LOCATION], **settings)(function)


def _apply(registry, concierge: Prompt):
    """Apply a prompt the way `feast apply` does: diff, then apply the diff."""
    repo = RepoContents(
        projects=[],
        data_sources=[],
        feature_views=[],
        on_demand_feature_views=[],
        stream_feature_views=[],
        label_views=[],
        entities=[],
        feature_services=[],
        permissions=[],
        prompts=[concierge],
    )
    diff = diff_between(registry, PROJECT, repo)
    apply_diff_to_registry(registry, diff, PROJECT)
    return diff


@pytest.fixture(params=["file", "sql"])
def registry(request, tmp_path):
    if request.param == "file":
        return Registry(
            PROJECT, RegistryConfig(path=str(tmp_path / "registry.db")), None
        )
    return SqlRegistry(
        SqlRegistryConfig(
            registry_type="sql",
            path=f"sqlite:///{tmp_path / 'registry.db'}",
            purge_feast_metadata=False,
        ),
        PROJECT,
        None,
    )


def test_decorated_function_renders_the_same_after_a_registry_round_trip():
    concierge = Prompt.from_proto(_build(max_tokens=20).to_proto())
    templated = Prompt.from_proto(
        _build(
            _concierge_variables,
            template="You are serving {{ name }} in {{ location }}. Visited: {{ visits }}",
            max_tokens=20,
            variables={"name": {"priority": "critical"}},
        ).to_proto()
    )
    row = {
        "user_profile__name": "Jane",
        "user_profile__visits": "Mama Ricotta's, " * 20,
        "location": "Chicago",
    }

    assert concierge == _build(max_tokens=20)
    assert concierge.features == ["user_profile:name", "user_profile:visits"]
    assert concierge.render(row).text == "You are serving Jane in Chicago."
    assert templated.render(row).text.startswith("You are serving Jane in Chicago.")
    assert templated.render(row).truncated_variables == ("visits",)
    with pytest.raises(TokenBudgetExceededError):
        concierge.render({**row, "location": "Chicago " * 20})


@pytest.mark.parametrize(
    "function, settings",
    [
        (_uses_a_module_import, {}),
        (lambda profile, request: "", {}),
        (lambda profile: "", {}),
        (_concierge, {"max_tokens": 0}),
        (_concierge, {"tokenizer": "cl100k"}),
        (_concierge, {"template": "{{ name }}"}),
        (_concierge, {"variables": {"name": {"priority": "critical"}}}),
        (_concierge, {"template": "prompts:/concierge", "max_tokens": 20}),
        (_concierge, {"template": "prompts:/concierge@production"}),
    ],
    ids=[
        "module import",
        "lambda",
        "one argument for two sources",
        "no budget",
        "unknown tokenizer",
        "template without max_tokens",
        "variables without template",
        "MLflow URI without version or alias",
        "MLflow URI without max_tokens",
    ],
)
def test_prompts_that_cannot_work_fail_when_defined(function, settings):
    with pytest.raises(ValueError):
        _build(function, **settings)


def test_feast_apply_stores_each_changed_prompt_as_a_new_version(registry):
    _apply(registry, _build())
    unchanged = _apply(registry, _build())
    _apply(registry, _build(_friendlier_concierge))

    assert [
        diff.transition_type
        for diff in unchanged.feast_object_diffs
        if diff.feast_object_type == FeastObjectType.PROMPT
    ] == [TransitionType.UNCHANGED]
    assert [
        version.reference
        for version in registry.list_prompt_versions("concierge", PROJECT)
    ] == ["concierge@v1", "concierge@v2"]
    assert registry.get_prompt("concierge", PROJECT) == _build(_friendlier_concierge)
    assert registry.get_prompt("concierge", PROJECT, version=1) == _build()
    assert [p.reference for p in registry.list_prompts(PROJECT)] == ["concierge@v2"]
    with pytest.raises(PromptNotFoundException):
        registry.get_prompt("concierge", PROJECT, version=3)

    registry.delete_prompt("concierge", PROJECT)
    assert registry.list_prompt_versions("concierge", PROJECT) == []


def _profile_store(tmp_path, now: datetime, **config):
    """A local store holding one user's profile, and the profile view."""
    data_path = tmp_path / "profiles.parquet"
    pd.DataFrame(
        {
            "user_id": ["user1"],
            "name": ["Jane"],
            "visits": ["Mama Ricotta's"],
            "event_timestamp": [now - timedelta(hours=1)],
        }
    ).to_parquet(data_path)
    store = FeatureStore(
        config=RepoConfig(
            project=PROJECT,
            registry=str(tmp_path / "registry.db"),
            provider="local",
            online_store={"type": "sqlite", "path": str(tmp_path / "online.db")},
            entity_key_serialization_version=3,
            **config,
        )
    )
    return store, _profile_view(str(data_path))


def test_render_prompt_reads_features_online_without_storing_the_prompt(tmp_path):
    now = datetime.now().replace(microsecond=0)
    store, profile = _profile_store(tmp_path, now)
    store.apply([USER, profile, _build(profile=profile)])
    store.materialize(now - timedelta(days=1), now)
    store.apply([_build(_friendlier_concierge, profile=profile)])

    [latest] = store.render_prompt(
        "concierge", [{"user_id": "user1", "location": "Chicago"}]
    )
    [pinned] = store.render_prompt(
        "concierge@v1", {"user_id": ["user1"], "location": ["Chicago"]}
    )
    training = store.get_historical_features(
        entity_df=pd.DataFrame(
            {"user_id": ["user1"], "location": ["Chicago"], "event_timestamp": [now]}
        ),
        features=store.get_prompt("concierge").features,
        full_feature_names=True,
    ).to_df()
    offline = store.get_prompt("concierge").render(training.iloc[0].to_dict())
    client = TestClient(get_app(store))
    served = client.post(
        "/render-prompt",
        json={
            "prompt": "concierge@v1",
            "entities": {"user_id": ["user1"], "location": ["Chicago"]},
        },
    ).json()

    assert latest.reference == "concierge@v2"
    assert latest.text == "Hello! You are serving Jane in Chicago."
    assert pinned.reference == "concierge@v1"
    assert pinned.text == "You are serving Jane in Chicago."
    assert offline.text == latest.text
    assert served["prompts"][0]["text"] == pinned.text
    assert client.get("/prompts/concierge").json()["meta"]["versionNumber"] == 2


@pytest.fixture
def mlflow_prompts(tmp_path) -> str:
    """An MLflow tracking URI whose registry holds two concierge versions."""
    mlflow = pytest.importorskip("mlflow")
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    client.register_prompt(
        name="concierge",
        template="You are serving {{ name }} in {{ location }}. Visited: {{ visits }}",
        model_config={"model_name": "gpt-4o", "temperature": 0.2},
    )
    client.register_prompt(
        name="concierge",
        template="Hello {{ name }}! In {{ location }}, after {{ visits }}",
    )
    client.register_prompt(
        name="chat", template=[{"role": "system", "content": "Serve {{ name }}"}]
    )
    client.set_prompt_alias("concierge", "production", 1)
    return tracking_uri


def test_template_from_mlflow_is_loaded_when_the_prompt_renders(mlflow_prompts):
    settings = {"max_tokens": 20, "variables": {"name": {"priority": "critical"}}}
    production = Prompt.from_proto(
        _build(
            _concierge_variables,
            template="prompts:/concierge@production",
            **settings,
        ).to_proto()
    )
    pinned = _build(_concierge_variables, template="prompts:/concierge/2", **settings)
    chat = _build(_concierge_variables, template="prompts:/chat/1", **settings)
    row = {
        "user_profile__name": "Jane",
        "user_profile__visits": "Mama Ricotta's, " * 20,
        "location": "Chicago",
    }

    rendered = production.render(row, mlflow_tracking_uri=mlflow_prompts)

    assert rendered.text.startswith("You are serving Jane in Chicago. Visited: Mama")
    assert rendered.truncated_variables == ("visits",)
    assert rendered.template_uri == "prompts:/concierge/1"
    assert rendered.model_config == {"model_name": "gpt-4o", "temperature": 0.2}
    assert pinned.render(row, mlflow_tracking_uri=mlflow_prompts).text.startswith(
        "Hello Jane! In Chicago, after Mama"
    )
    with pytest.raises(ValueError, match="chat prompt"):
        chat.render(row, mlflow_tracking_uri=mlflow_prompts)


def test_render_prompt_loads_mlflow_templates_from_feature_store_config(
    tmp_path, mlflow_prompts
):
    now = datetime.now().replace(microsecond=0)
    store, profile = _profile_store(
        tmp_path, now, mlflow={"tracking_uri": mlflow_prompts}
    )
    concierge = _build(
        _concierge_variables,
        profile=profile,
        template="prompts:/concierge@production",
        max_tokens=50,
    )
    store.apply([USER, profile, concierge])
    store.materialize(now - timedelta(days=1), now)

    [rendered] = store.render_prompt(
        "concierge", [{"user_id": "user1", "location": "Chicago"}]
    )

    assert rendered.text == ("You are serving Jane in Chicago. Visited: Mama Ricotta's")
    assert rendered.reference == "concierge@v1"
    assert rendered.template_uri == "prompts:/concierge/1"
