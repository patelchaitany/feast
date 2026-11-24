"""Sequence-mode feature views must retrieve "the last N events as of each
entity row's timestamp" from the offline store, matching what the online store
serves, so that training and serving see the same data shape."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, Mock

import pytest
import sqlglot

from feast import Entity, FeatureView, Field, FileSource
from feast.infra.offline_stores import bigquery, offline_utils
from feast.infra.offline_stores.dask import DaskOfflineStore
from feast.infra.offline_stores.duckdb import DuckDBOfflineStore
from feast.infra.offline_stores.offline_utils import (
    FeatureViewQueryContext,
    get_feature_view_query_context,
)
from feast.online_config import OnlineConfig
from feast.types import Float64


def _query_context(
    sequence_length: int, created_timestamp_column=None
) -> FeatureViewQueryContext:
    return FeatureViewQueryContext(
        name="driver_stats",
        ttl=3600,
        entities=["driver_id"],
        features=["conv_rate"],
        field_mapping={},
        timestamp_field="event_timestamp",
        created_timestamp_column=created_timestamp_column,
        table_subquery="(SELECT * FROM feature_table)",
        entity_selections=["driver_id AS driver_id"],
        min_event_timestamp=None,
        max_event_timestamp="2026-01-01T00:00:00",
        date_partition_column=None,
        timestamp_field_type=None,
        sequence_length=sequence_length,
    )


def _render_bigquery_query(context: FeatureViewQueryContext) -> str:
    return offline_utils.build_point_in_time_query(
        [context],
        left_table_query_string="entity_df",
        entity_df_event_timestamp_col="event_timestamp",
        entity_df_columns={"driver_id": None, "event_timestamp": None}.keys(),
        query_template=bigquery.MULTIPLE_FEATURE_VIEW_POINT_IN_TIME_JOIN,
        full_feature_names=False,
    )


def test_bigquery_template_sequence_mode_keeps_newest_n_events():
    sql = _render_bigquery_query(_query_context(sequence_length=3))

    assert "row_number <= 3" in sql
    assert "ARRAY_AGG(" in sql
    # Dropping nulls would shift the lists of one row out of line.
    assert "IGNORE NULLS" not in sql
    assert "ORDER BY event_timestamp DESC" in sql
    # Aggregated straight from __latest, not joined back to __base (which
    # would repeat rows that share an event timestamp).
    assert (
        "FROM driver_stats__latest\n    GROUP BY driver_stats__entity_row_unique_id"
        in sql
    )
    # Entity rows without events get an empty array, not null.
    assert "COALESCE(`conv_rate`, []) AS `conv_rate`" in sql
    sqlglot.parse_one(sql, read="bigquery")


def test_bigquery_template_latest_mode_is_unchanged():
    sql = _render_bigquery_query(_query_context(sequence_length=0))

    assert "row_number = 1" in sql
    assert "ARRAY_AGG" not in sql
    assert "SELECT base.*" in sql
    assert "COALESCE" not in sql


def _sequence_feature_view(online_config: OnlineConfig) -> FeatureView:
    driver = Entity(name="driver_id", join_keys=["driver_id"])
    fv = FeatureView(
        name="driver_stats",
        entities=[driver],
        schema=[Field(name="conv_rate", dtype=Float64)],
        source=FileSource(
            path="unused.parquet",
            timestamp_field="event_timestamp",
            created_timestamp_column="created",
        ),
        online_config=online_config,
    )
    # The SQL query context needs a source with a table query string; swap in
    # a minimal mock after construction (schema inference already happened).
    mock_source = MagicMock()
    mock_source.field_mapping = {}
    mock_source.timestamp_field = "event_timestamp"
    mock_source.created_timestamp_column = "created"
    mock_source.date_partition_column = None
    mock_source.get_table_query_string.return_value = "(SELECT * FROM feature_table)"
    fv.batch_source = mock_source
    return fv


def _mock_registry() -> Mock:
    registry = Mock()
    registry.list_on_demand_feature_views = lambda project, allow_cache=True: []
    return registry


def test_query_context_defaults_to_online_config_max_length():
    fv = _sequence_feature_view(
        OnlineConfig(mode="sequence", write_mode="append", max_length=7)
    )

    context = get_feature_view_query_context(
        ["driver_stats:conv_rate"],
        [fv],
        _mock_registry(),
        "project",
        (
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            datetime(2026, 1, 2, tzinfo=timezone.utc),
        ),
    )[0]

    assert context.sequence_length == 7
    # No max_age: the lookback is unbounded (ttl=0), not the feature view ttl.
    assert context.ttl == 0
    assert context.min_event_timestamp is None


def test_query_context_sequence_length_override():
    fv = _sequence_feature_view(
        OnlineConfig(mode="sequence", write_mode="append", max_length=7)
    )

    context = get_feature_view_query_context(
        ["driver_stats:conv_rate"],
        [fv],
        _mock_registry(),
        "project",
        (
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            datetime(2026, 1, 2, tzinfo=timezone.utc),
        ),
        sequence_length=2,
    )[0]

    assert context.sequence_length == 2


def test_query_context_sequence_length_override_is_capped_at_max_length():
    fv = _sequence_feature_view(
        OnlineConfig(mode="sequence", write_mode="append", max_length=7)
    )

    context = get_feature_view_query_context(
        ["driver_stats:conv_rate"],
        [fv],
        _mock_registry(),
        "project",
        (
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            datetime(2026, 1, 2, tzinfo=timezone.utc),
        ),
        sequence_length=50,
    )[0]

    # Online reads never return more than max_length events.
    assert context.sequence_length == 7


def test_unsupported_offline_store_rejects_sequence_feature_views():
    fv = _sequence_feature_view(
        OnlineConfig(mode="sequence", write_mode="append", max_length=7)
    )

    # Dask would silently return one value per feature instead of a sequence.
    with pytest.raises(NotImplementedError, match="driver_stats"):
        DaskOfflineStore().ensure_sequence_retrieval_supported([fv])
    DuckDBOfflineStore().ensure_sequence_retrieval_supported([fv])
    fv.online_config = None
    DaskOfflineStore().ensure_sequence_retrieval_supported([fv])


def test_query_context_sequence_uses_max_age_not_ttl():
    fv = _sequence_feature_view(
        OnlineConfig(
            mode="sequence",
            write_mode="append",
            max_length=5,
            max_age=timedelta(hours=2),
        ),
    )
    fv.ttl = timedelta(days=365)

    context = get_feature_view_query_context(
        ["driver_stats:conv_rate"],
        [fv],
        _mock_registry(),
        "project",
        (
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            datetime(2026, 1, 2, tzinfo=timezone.utc),
        ),
    )[0]

    assert context.sequence_length == 5
    # The lookback window mirrors online_read_sequence: max_age, not the fv ttl.
    assert context.ttl == 2 * 60 * 60
    assert context.min_event_timestamp is not None


def test_query_context_latest_mode_is_unchanged():
    fv = _sequence_feature_view(
        OnlineConfig(mode="sequence", write_mode="append", max_length=5)
    )
    fv.online_config = None
    fv.ttl = timedelta(days=1)

    context = get_feature_view_query_context(
        ["driver_stats:conv_rate"],
        [fv],
        _mock_registry(),
        "project",
        (
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            datetime(2026, 1, 2, tzinfo=timezone.utc),
        ),
    )[0]

    assert context.sequence_length == 0
    assert context.ttl == 24 * 60 * 60


def test_query_context_rejects_bad_sequence_length_override():
    fv = _sequence_feature_view(
        OnlineConfig(mode="sequence", write_mode="append", max_length=5)
    )
    with pytest.raises(ValueError):
        get_feature_view_query_context(
            ["driver_stats:conv_rate"],
            [fv],
            _mock_registry(),
            "project",
            (
                datetime(2026, 1, 1, tzinfo=timezone.utc),
                datetime(2026, 1, 2, tzinfo=timezone.utc),
            ),
            sequence_length=0,
        )
