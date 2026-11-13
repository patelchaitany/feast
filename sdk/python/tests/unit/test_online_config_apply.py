import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, patch

import pandas as pd

from feast import Entity, FeatureStore, FeatureView, Field, OnlineConfig, ValueType
from feast.data_source import PushMode, PushSource
from feast.feature_server_utils import convert_response_to_dict
from feast.infra.offline_stores.file_source import FileSource
from feast.infra.online_stores.online_store import SequenceRead
from feast.infra.online_stores.sqlite import SqliteOnlineStore, SqliteOnlineStoreConfig
from feast.protos.feast.types.Value_pb2 import Value as ValueProto
from feast.repo_config import RepoConfig
from feast.types import Float32, Int64
from feast.utils import make_tzaware


def _feature_store(tmp_path: Path) -> FeatureStore:
    return FeatureStore(
        config=RepoConfig(
            registry=str(tmp_path / "registry.db"),
            project="online_config_test",
            provider="local",
            online_store=SqliteOnlineStoreConfig(path=str(tmp_path / "online.db")),
            entity_key_serialization_version=3,
        )
    )


def test_apply_persists_updates_and_clears_online_config(tmp_path: Path) -> None:
    store = _feature_store(tmp_path)
    source = FileSource(
        path="events.parquet",
        timestamp_field="event_timestamp",
        created_timestamp_column="created_timestamp",
    )
    entity = Entity(name="event", join_keys=["event_id"], value_type=ValueType.INT64)
    schema = [
        Field(name="event_id", dtype=Int64),
        Field(name="event_value", dtype=Float32),
    ]

    def feature_view(online_config: Optional[OnlineConfig]) -> FeatureView:
        return FeatureView(
            name="events",
            entities=[entity],
            schema=schema,
            source=source,
            online_config=online_config,
        )

    try:
        initial_config = OnlineConfig(
            mode="sequence", max_length=10, write_mode="append"
        )
        store.apply([entity, feature_view(initial_config)])
        assert store.get_feature_view("events").online_config == initial_config
        versions_before_config_update = store.list_feature_view_versions("events")

        updated_config = OnlineConfig(
            mode="sequence",
            max_length=50,
            max_age=timedelta(days=90),
            write_mode="append",
        )
        store.apply([entity, feature_view(updated_config)])
        assert store.get_feature_view("events").online_config == updated_config
        assert (
            store.list_feature_view_versions("events") == versions_before_config_update
        )

        store.apply([entity, feature_view(None)])
        stored = store.get_feature_view("events")
        assert stored.online_config is None
        assert not stored.to_proto().spec.HasField("online_config")
        assert (
            store.list_feature_view_versions("events") == versions_before_config_update
        )
    finally:
        store.teardown()


def test_write_to_online_store_routes_by_write_mode(tmp_path: Path) -> None:
    store = _feature_store(tmp_path)
    source = FileSource(
        path="events.parquet",
        timestamp_field="event_timestamp",
        created_timestamp_column="created_timestamp",
    )
    entity = Entity(name="event", join_keys=["event_id"], value_type=ValueType.INT64)
    schema = [
        Field(name="event_id", dtype=Int64),
        Field(name="event_value", dtype=Float32),
    ]
    append_view = FeatureView(
        name="event_sequence",
        entities=[entity],
        schema=schema,
        source=source,
        online_config=OnlineConfig(mode="sequence", max_length=10, write_mode="append"),
    )
    overwrite_view = FeatureView(
        name="event_latest", entities=[entity], schema=schema, source=source
    )
    now = datetime.now(tz=timezone.utc)
    df = pd.DataFrame(
        {
            "event_id": [1],
            "event_value": [1.0],
            "event_timestamp": [now],
            "created_timestamp": [now],
        }
    )

    try:
        store.apply([entity, append_view, overwrite_view])
        with (
            patch.object(SqliteOnlineStore, "online_append") as online_append,
            patch.object(
                SqliteOnlineStore, "online_append_async", new_callable=AsyncMock
            ) as online_append_async,
            patch.object(SqliteOnlineStore, "online_write_batch") as online_write_batch,
        ):
            store.write_to_online_store("event_sequence", df)
            asyncio.run(store.write_to_online_store_async("event_sequence", df))
            store.write_to_online_store("event_latest", df)

        assert [call.args[1].name for call in online_append.call_args_list] == [
            "event_sequence"
        ]
        assert [call.args[1].name for call in online_append_async.await_args_list] == [
            "event_sequence"
        ]
        assert [call.args[1].name for call in online_write_batch.call_args_list] == [
            "event_latest"
        ]
    finally:
        store.teardown()


def test_push_appends_each_event_for_sequence_views(tmp_path: Path) -> None:
    store = _feature_store(tmp_path)
    push_source = PushSource(
        name="event_push",
        batch_source=FileSource(
            path="events.parquet", timestamp_field="event_timestamp"
        ),
    )
    entity = Entity(name="event", join_keys=["event_id"], value_type=ValueType.INT64)
    sequence_view = FeatureView(
        name="event_sequence",
        entities=[entity],
        schema=[
            Field(name="event_id", dtype=Int64),
            Field(name="event_value", dtype=Float32),
        ],
        source=push_source,
        online_config=OnlineConfig(mode="sequence", max_length=10, write_mode="append"),
    )
    t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t2 = t1 + timedelta(minutes=1)

    def event(value: float, ts: datetime) -> pd.DataFrame:
        return pd.DataFrame(
            {"event_id": [1], "event_value": [value], "event_timestamp": [ts]}
        )

    try:
        store.apply([entity, sequence_view])
        with (
            patch.object(SqliteOnlineStore, "online_append") as online_append,
            patch.object(SqliteOnlineStore, "online_write_batch") as online_write_batch,
            patch.object(FeatureStore, "write_to_offline_store") as offline_write,
        ):
            store.push("event_push", event(1.0, t1))
            store.push("event_push", event(2.0, t2), to=PushMode.ONLINE_AND_OFFLINE)

        # Each push appends its event; nothing overwrites the first one.
        appended = [
            (make_tzaware(ts), values["event_value"].float_val)
            for call in online_append.call_args_list
            for _, values, ts, _ in call.args[2]
        ]
        assert appended == [(t1, 1.0), (t2, 2.0)]
        online_write_batch.assert_not_called()
        offline_write.assert_called_once()
    finally:
        store.teardown()


def test_get_online_features_returns_sequences(tmp_path: Path) -> None:
    store = _feature_store(tmp_path)
    source = FileSource(path="events.parquet", timestamp_field="event_timestamp")
    entity = Entity(name="event", join_keys=["event_id"], value_type=ValueType.INT64)
    schema = [
        Field(name="event_id", dtype=Int64),
        Field(name="event_value", dtype=Float32),
    ]
    sequence_view = FeatureView(
        name="event_sequence",
        entities=[entity],
        schema=schema,
        source=source,
        online_config=OnlineConfig(mode="sequence", max_length=10, write_mode="append"),
    )
    latest_view = FeatureView(
        name="event_latest", entities=[entity], schema=schema, source=source
    )
    t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t2 = t1 + timedelta(minutes=1, milliseconds=250)
    events = [
        (t2, {"event_value": ValueProto(float_val=2.0)}),
        (t1, {"event_value": ValueProto(float_val=1.0)}),
    ]

    def read_sequence(config, table, entity_keys, requested_features, limit, **_):
        return [
            SequenceRead(events[:limit], 5)
            if key.entity_values[0].int64_val == 1
            else SequenceRead([], 0)
            for key in entity_keys
        ]

    try:
        store.apply([entity, sequence_view, latest_view])
        with patch.object(
            SqliteOnlineStore, "online_read_sequence", side_effect=read_sequence
        ) as online_read_sequence:
            response = store.get_online_features(
                features=["event_sequence:event_value", "event_latest:event_value"],
                entity_rows=[{"event_id": 1}, {"event_id": 2}],
                full_feature_names=True,
                sequence_length=2,
                sequence_order="asc",
            )
            latest = store.get_online_features(
                features=["event_sequence:event_value"],
                entity_rows=[{"event_id": 1}],
            )

        # Latest-mode views stay in "results"; sequence-mode views move to "sequences".
        name = "event_sequence__event_value"
        assert response.to_dict() == {
            "event_id": [1, 2],
            "event_latest__event_value": [None, None],
        }
        assert response.to_sequences() == {
            "event_sequence": [
                [
                    {"event_timestamp": t2, name: 2.0},
                    {"event_timestamp": t1, name: 1.0},
                ],
                [],
            ]
        }
        assert response.to_sequence_df()["event_id"].tolist() == [1, 1]
        assert convert_response_to_dict(response.proto)["sequences"][0] == {
            "feature_view": "event_sequence",
            "entity_key": {"event_id": 1},
            "events": [
                {"event_timestamp_ms": 1767225660250, "features": {name: 2.0}},
                {"event_timestamp_ms": 1767225600000, "features": {name: 1.0}},
            ],
            "total_count": 5,
        }
        assert online_read_sequence.call_args_list[0].kwargs["order"] == "asc"

        # sequence_length 0 or 1 keeps one value per feature in "results".
        assert latest.to_dict() == {"event_id": [1], "event_value": [2.0]}
        assert not latest.proto.sequences
        assert online_read_sequence.call_args_list[1].kwargs["limit"] == 1
    finally:
        store.teardown()
