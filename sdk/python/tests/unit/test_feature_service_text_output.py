from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from feast import (
    Entity,
    FeatureService,
    FeatureStore,
    FeatureView,
    Field,
    FileSource,
    RepoConfig,
    RequestSource,
)
from feast.errors import FeatureServiceTextOutputError
from feast.feature_server import get_app
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Int64, String

USER = Entity(name="user", join_keys=["user_id"])
LOCATION = RequestSource(
    name="location_request", schema=[Field(name="location", dtype=String)]
)


def _profile_view(path: str = "unused.parquet") -> FeatureView:
    return FeatureView(
        name="user_profile",
        entities=[USER],
        ttl=timedelta(days=1),
        schema=[Field(name="name", dtype=String)],
        source=FileSource(path=path, timestamp_field="event_timestamp"),
    )


@on_demand_feature_view(
    sources=[_profile_view(), LOCATION],
    schema=[
        Field(name="recommendation_prompt", dtype=String),
        Field(name="prompt_length", dtype=Int64),
    ],
    mode="python",
)
def recommendation_prompt_view(inputs: dict[str, Any]) -> dict[str, Any]:
    prompts = [
        f"Recommend films for {name} in {location}."
        for name, location in zip(inputs["name"], inputs["location"])
    ]
    return {
        "recommendation_prompt": prompts,
        "prompt_length": [len(prompt) for prompt in prompts],
    }


def _store(tmp_path) -> FeatureStore:
    now = datetime.now().replace(microsecond=0)
    data_path = tmp_path / "profiles.parquet"
    pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "name": ["Alice", "Bob"],
            "event_timestamp": [now - timedelta(hours=1)] * 2,
        }
    ).to_parquet(data_path)
    store = FeatureStore(
        config=RepoConfig(
            project="text_output",
            registry=str(tmp_path / "registry.db"),
            provider="local",
            online_store={"type": "sqlite", "path": str(tmp_path / "online.db")},
            entity_key_serialization_version=3,
        )
    )
    store.apply([USER, _profile_view(str(data_path)), recommendation_prompt_view])
    store.materialize(now - timedelta(days=1), now)
    return store


def test_text_mode_feature_service_returns_its_prompt_as_text(tmp_path):
    store = _store(tmp_path)
    store.apply(
        [
            FeatureService(
                name="recommendation_prompt_service",
                features=[recommendation_prompt_view],
                output_format="text",
                output_field="recommendation_prompt",
            )
        ]
    )
    service = store.get_feature_service("recommendation_prompt_service")

    one = store.get_online_features(
        features=service, entity_rows=[{"user_id": "u1", "location": "Paris"}]
    )
    both = store.get_online_features(
        features=service,
        entity_rows={"user_id": ["u1", "u2"], "location": ["Paris", "Rome"]},
    )
    client = TestClient(get_app(store))
    body = {
        "feature_service": "recommendation_prompt_service",
        "entities": {"user_id": ["u1"], "location": ["Paris"]},
    }
    as_text = client.post(
        "/get-online-features", json=body, headers={"Accept": "text/plain"}
    )
    as_json = client.post("/get-online-features", json=body)

    assert service.output_format == "text"
    assert one.to_text() == "Recommend films for Alice in Paris."
    # The feature response is unchanged; text_response is what gRPC returns.
    assert one.to_dict()["recommendation_prompt"] == [one.to_text()]
    assert list(both.proto.text_response) == [
        "Recommend films for Alice in Paris.",
        "Recommend films for Bob in Rome.",
    ]
    with pytest.raises(FeatureServiceTextOutputError, match="holds 2"):
        both.to_text()
    assert as_text.headers["content-type"].startswith("text/plain")
    assert as_text.text == "Recommend films for Alice in Paris."
    assert "textResponse" not in as_json.json()
    assert "text_response" not in as_json.json()
    assert as_json.json()["metadata"]["feature_names"][-2:] == [
        "recommendation_prompt",
        "prompt_length",
    ]


def test_text_mode_needs_one_string_output_field(tmp_path):
    with pytest.raises(ValueError, match="needs output_field"):
        FeatureService(
            name="s", features=[recommendation_prompt_view], output_format="text"
        )
    with pytest.raises(ValueError, match="the only output format is 'text'"):
        FeatureService(
            name="s",
            features=[recommendation_prompt_view],
            output_format="json",
            output_field="recommendation_prompt",
        )
    with pytest.raises(ValueError, match="must be a String feature"):
        _store(tmp_path).apply(
            [
                FeatureService(
                    name="length_service",
                    features=[recommendation_prompt_view],
                    output_format="text",
                    output_field="prompt_length",
                )
            ]
        )
