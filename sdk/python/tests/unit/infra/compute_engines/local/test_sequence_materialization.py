from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from unittest.mock import MagicMock

import pandas as pd
import pytest

from feast import Entity, FeatureView, Field, FileSource, OnlineConfig, RepoConfig
from feast.infra.common.materialization_job import (
    MaterializationJobStatus,
    MaterializationTask,
)
from feast.infra.compute_engines.local.compute import LocalComputeEngine
from feast.infra.offline_stores.dask import DaskOfflineStore
from feast.infra.online_stores.online_store import OnlineStore
from feast.protos.feast.types.EntityKey_pb2 import EntityKey as EntityKeyProto
from feast.protos.feast.types.Value_pb2 import Value as ValueProto
from feast.repo_config import MaterializationConfig
from feast.types import Int32, Int64
from feast.value_type import ValueType

Row = Tuple[EntityKeyProto, Dict[str, ValueProto], datetime, Optional[datetime]]

START = datetime(2026, 7, 1, tzinfo=timezone.utc)
END = datetime(2026, 8, 2, tzinfo=timezone.utc)


class RecordingOnlineStore(OnlineStore):
    """Online store that records the rows each write method receives."""

    def __init__(self) -> None:
        self.appended: List[Row] = []
        self.written: List[Row] = []

    def online_write_batch(
        self,
        config: RepoConfig,
        table: FeatureView,
        data: List[Row],
        progress: Optional[Callable[[int], Any]],
    ) -> None:
        self.written.extend(data)

    def online_append(
        self,
        config: RepoConfig,
        table: FeatureView,
        data: List[Row],
        progress: Optional[Callable[[int], Any]] = None,
    ) -> None:
        self.appended.extend(data)

    def online_read(self, config, table, entity_keys, requested_features=None):
        raise NotImplementedError

    def update(
        self,
        config: RepoConfig,
        tables_to_delete: Sequence[FeatureView],
        tables_to_keep: Sequence[FeatureView],
        entities_to_delete: Sequence[Entity],
        entities_to_keep: Sequence[Entity],
        partial: bool,
    ) -> None:
        pass

    def teardown(
        self,
        config: RepoConfig,
        tables: Sequence[FeatureView],
        entities: Sequence[Entity],
    ) -> None:
        pass


class LatestOnlyOnlineStore(RecordingOnlineStore):
    """Online store that keeps the base online_append, which is unsupported."""

    online_append = OnlineStore.online_append  # type: ignore[assignment]


@pytest.fixture
def events_path(tmp_path):
    """Three events for user 1 and one for user 2, all inside [START, END]."""
    path = tmp_path / "events.parquet"
    pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2],
            "movie_id": [10, 20, 30, 40],
            "event_timestamp": [
                datetime(2026, 7, 1, 12, tzinfo=timezone.utc),
                datetime(2026, 7, 15, tzinfo=timezone.utc),
                datetime(2026, 8, 1, tzinfo=timezone.utc),
                datetime(2026, 7, 20, tzinfo=timezone.utc),
            ],
        }
    ).to_parquet(path)
    return str(path)


def _feature_view(
    events_path: str, online_config: Optional[OnlineConfig]
) -> FeatureView:
    return FeatureView(
        name="watched",
        entities=[
            Entity(name="user", join_keys=["user_id"], value_type=ValueType.INT64)
        ],
        ttl=timedelta(days=365),
        schema=[
            Field(name="user_id", dtype=Int64),
            Field(name="movie_id", dtype=Int32),
        ],
        source=FileSource(path=events_path, timestamp_field="event_timestamp"),
        online_config=online_config,
    )


def _materialize(
    feature_view: FeatureView,
    online_store: RecordingOnlineStore,
    tmp_path,
    pull_latest_features: bool = False,
) -> MaterializationTask:
    repo_config = RepoConfig(
        project="test",
        provider="local",
        registry=str(tmp_path / "registry.db"),
        entity_key_serialization_version=3,
        materialization=MaterializationConfig(
            pull_latest_features=pull_latest_features
        ),
    )
    engine = LocalComputeEngine(
        repo_config=repo_config,
        offline_store=DaskOfflineStore(),
        online_store=online_store,
        backend="pandas",
    )
    registry = MagicMock()
    registry.get_entity.return_value = Entity(
        name="user", join_keys=["user_id"], value_type=ValueType.INT64
    )
    task = MaterializationTask(
        project="test", feature_view=feature_view, start_time=START, end_time=END
    )

    jobs = engine.materialize(registry, task)

    assert jobs[0].status() == MaterializationJobStatus.SUCCEEDED, jobs[0].error()
    return task


def _movies_by_user(rows: List[Row]) -> Dict[int, List[int]]:
    movies: Dict[int, List[int]] = {}
    for entity_key, values, _, _ in sorted(rows, key=lambda row: row[2]):
        user_id = entity_key.entity_values[0].int64_val
        movies.setdefault(user_id, []).append(values["movie_id"].int32_val)
    return movies


@pytest.mark.parametrize("pull_latest_features", [False, True])
def test_sequence_view_appends_every_event(events_path, tmp_path, pull_latest_features):
    """All events are appended, even when pull_latest_features is set."""
    online_config = OnlineConfig(mode="sequence", max_length=10, write_mode="append")
    online_store = RecordingOnlineStore()

    _materialize(
        _feature_view(events_path, online_config),
        online_store,
        tmp_path,
        pull_latest_features=pull_latest_features,
    )

    assert online_store.written == []
    assert _movies_by_user(online_store.appended) == {1: [10, 20, 30], 2: [40]}


def test_sequence_view_keeps_newest_max_length_events(events_path, tmp_path):
    """Only the newest max_length events per entity are appended."""
    online_config = OnlineConfig(mode="sequence", max_length=2, write_mode="append")
    online_store = RecordingOnlineStore()

    _materialize(_feature_view(events_path, online_config), online_store, tmp_path)

    assert _movies_by_user(online_store.appended) == {1: [20, 30], 2: [40]}


def test_latest_view_writes_latest_event_per_entity(events_path, tmp_path):
    """Views without sequence mode keep writing one row per entity."""
    online_store = RecordingOnlineStore()

    _materialize(_feature_view(events_path, None), online_store, tmp_path)

    assert online_store.appended == []
    assert _movies_by_user(online_store.written) == {1: [30], 2: [40]}


def test_sequence_view_fails_early_without_online_append(events_path, tmp_path):
    """Online stores without online_append fail before any data is read."""
    online_config = OnlineConfig(mode="sequence", max_length=2, write_mode="append")
    offline_store = MagicMock()
    engine = LocalComputeEngine(
        repo_config=MagicMock(),
        offline_store=offline_store,
        online_store=LatestOnlyOnlineStore(),
    )
    task = MaterializationTask(
        project="test",
        feature_view=_feature_view(events_path, online_config),
        start_time=START,
        end_time=END,
    )

    with pytest.raises(NotImplementedError, match="does not support online_append"):
        engine.materialize(MagicMock(), task)

    offline_store.pull_all_from_table_or_query.assert_not_called()
