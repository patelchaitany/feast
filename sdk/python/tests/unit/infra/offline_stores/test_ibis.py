from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple
from unittest.mock import MagicMock

import ibis
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from feast import Entity, FeatureView, Field, FileSource
from feast.infra.offline_stores.ibis import (
    get_historical_features_ibis,
    point_in_time_join,
    pull_all_from_table_or_query_ibis,
    pull_latest_from_table_or_query_ibis,
)
from feast.online_config import OnlineConfig
from feast.types import Float64, Int64
from feast.value_type import ValueType


def pa_datetime(year, month, day, hour=0, minute=0):
    return pa.scalar(
        datetime(year, month, day, hour, minute), type=pa.timestamp("s", tz="UTC")
    )


def customer_table():
    return pa.Table.from_arrays(
        arrays=[
            pa.array([1, 1, 2, 3]),
            pa.array(
                [
                    pa_datetime(2024, 1, 1),
                    pa_datetime(2024, 1, 2),
                    pa_datetime(2024, 1, 1),
                    pa_datetime(2024, 1, 3),
                ]
            ),
        ],
        names=["customer_id", "event_timestamp"],
    )


def features_table_1():
    return pa.Table.from_arrays(
        arrays=[
            pa.array([1, 1, 1, 2, 3, 3]),
            pa.array(
                [
                    pa_datetime(2023, 12, 31),
                    pa_datetime(2024, 1, 2),
                    pa_datetime(2024, 1, 3),
                    pa_datetime(2023, 1, 3),
                    pa_datetime(2024, 1, 1),
                    pa_datetime(2024, 1, 1),
                ]
            ),
            pa.array(
                [
                    pa_datetime(2023, 12, 31),
                    pa_datetime(2024, 1, 2),
                    pa_datetime(2024, 1, 3),
                    pa_datetime(2023, 1, 3),
                    pa_datetime(2024, 1, 3),
                    pa_datetime(2024, 1, 2),
                ]
            ),
            pa.array([11, 22, 33, 22, 10, 20]),
        ],
        names=["customer_id", "event_timestamp", "created", "feature1"],
    )


def point_in_time_join_brute(
    entity_table: pa.Table,
    feature_tables: List[
        Tuple[pa.Table, str, str, Dict[str, str], List[str], timedelta]
    ],
    event_timestamp_col="event_timestamp",
):
    ret_fields = [entity_table.schema.field(n) for n in entity_table.schema.names]

    from operator import itemgetter

    ret = entity_table.to_pydict()
    batch_dict = entity_table.to_pydict()

    for i, row_timestmap in enumerate(batch_dict[event_timestamp_col]):
        for (
            feature_table,
            timestamp_key,
            created_timestamp_key,
            join_key_map,
            feature_refs,
            ttl,
            _sequence_length,
        ) in feature_tables:
            if i == 0:
                ret_fields.extend(
                    [
                        feature_table.schema.field(f)
                        for f in feature_table.schema.names
                        if f not in join_key_map.values()
                        and f != timestamp_key
                        and f != created_timestamp_key
                    ]
                )

            def check_equality(ft_dict, batch_dict, x, y):
                return all(
                    [ft_dict[k][x] == batch_dict[v][y] for k, v in join_key_map.items()]
                )

            ft_dict = feature_table.to_pydict()

            found_matches = [
                (j, (ft_dict[timestamp_key][j], ft_dict[created_timestamp_key][j]))
                # (j, ft_dict[timestamp_key][j])
                for j in range(feature_table.num_rows)
                if check_equality(ft_dict, batch_dict, j, i)
                and ft_dict[timestamp_key][j] <= row_timestmap
                and ft_dict[timestamp_key][j] >= row_timestmap - ttl
            ]

            index_found = (
                max(found_matches, key=itemgetter(1))[0] if found_matches else None
            )

            for col in ft_dict.keys():
                if col not in feature_refs:
                    continue

                if col not in ret:
                    ret[col] = []

                if index_found is not None:
                    ret[col].append(ft_dict[col][index_found])
                else:
                    ret[col].append(None)

    return pa.Table.from_pydict(ret, schema=pa.schema(ret_fields))


def tables_equal_ignore_order(actual: pa.Table, expected: pa.Table):
    sort_keys = [(name, "ascending") for name in actual.column_names]
    sort_indices = pc.sort_indices(actual, sort_keys)
    actual = pc.take(actual, sort_indices)

    sort_keys = [(name, "ascending") for name in expected.column_names]
    sort_indices = pc.sort_indices(expected, sort_keys)
    expected = pc.take(expected, sort_indices)

    return actual.equals(expected)


def test_point_in_time_join():
    expected = point_in_time_join_brute(
        customer_table(),
        feature_tables=[
            (
                features_table_1(),
                "event_timestamp",
                "created",
                {"customer_id": "customer_id"},
                ["feature1"],
                timedelta(days=10),
                None,
            )
        ],
    )

    actual = point_in_time_join(
        ibis.memtable(customer_table()),
        feature_tables=[
            (
                ibis.memtable(features_table_1()),
                "event_timestamp",
                "created",
                {"customer_id": "customer_id"},
                ["feature1"],
                timedelta(days=10),
                None,
            )
        ],
    ).to_pyarrow()

    assert tables_equal_ignore_order(actual, expected)


def test_point_in_time_join_keeps_entity_rows_apart():
    """Entity rows whose key values concatenate to the same string, or that have
    a null key for another feature view, still get their own feature values."""
    entity_table = pa.table(
        {
            "driver_id": ["1", "11", "7"],
            "customer_id": ["11", "1", None],
            "event_timestamp": [pa_datetime(2024, 1, 2)] * 3,
        }
    )
    drivers = pa.table(
        {
            "driver_id": ["1", "11", "7"],
            "event_timestamp": [pa_datetime(2024, 1, 1)] * 3,
            "created": [pa_datetime(2024, 1, 1)] * 3,
            "driver_rating": [1.0, 11.0, 7.0],
        }
    )
    customers = pa.table(
        {
            "customer_id": ["11", "1"],
            "event_timestamp": [pa_datetime(2024, 1, 1)] * 2,
            "created": [pa_datetime(2024, 1, 1)] * 2,
            "customer_score": [110.0, 10.0],
        }
    )

    actual = point_in_time_join(
        ibis.memtable(entity_table),
        feature_tables=[
            (
                ibis.memtable(drivers),
                "event_timestamp",
                "created",
                {"driver_id": "driver_id"},
                ["driver_rating"],
                timedelta(days=10),
                None,
            ),
            (
                ibis.memtable(customers),
                "event_timestamp",
                "created",
                {"customer_id": "customer_id"},
                ["customer_score"],
                timedelta(days=10),
                None,
            ),
        ],
    ).to_pyarrow()

    rows = {
        row["driver_id"]: (row["driver_rating"], row["customer_score"])
        for row in actual.to_pylist()
    }
    assert rows == {"1": (1.0, 110.0), "11": (11.0, 10.0), "7": (7.0, None)}


def sequence_events_table():
    """Events for driver 1001 at five timestamps; 11:45 is in the future of
    every entity row below."""
    return pa.table(
        {
            "driver_id": [1001] * 5,
            "event_timestamp": [
                pa_datetime(2024, 1, 1, 9, 15),
                pa_datetime(2024, 1, 1, 10, 0),
                pa_datetime(2024, 1, 1, 10, 30),
                pa_datetime(2024, 1, 1, 10, 45),
                pa_datetime(2024, 1, 1, 11, 45),
            ],
            "created": [
                pa_datetime(2024, 1, 1, 9, 15),
                pa_datetime(2024, 1, 1, 10, 0),
                pa_datetime(2024, 1, 1, 10, 30),
                pa_datetime(2024, 1, 1, 10, 45),
                pa_datetime(2024, 1, 1, 11, 45),
            ],
            "conv_rate": [0.60, 0.66, 0.71, 0.78, 0.90],
        }
    )


def test_point_in_time_join_sequence_mode():
    entity_table = pa.table(
        {
            "driver_id": [1001, 1001, 1002],
            "event_timestamp": [
                pa_datetime(2024, 1, 1, 10, 59),
                pa_datetime(2024, 1, 1, 10, 20),
                pa_datetime(2024, 1, 1, 10, 59),
            ],
        }
    )

    # The 10:45 event has no trips value.
    events = sequence_events_table().append_column(
        "trips", pa.array([1, 2, 3, None, 5], type=pa.int64())
    )

    actual = point_in_time_join(
        ibis.memtable(entity_table),
        feature_tables=[
            (
                ibis.memtable(events),
                "event_timestamp",
                "created",
                {"driver_id": "driver_id"},
                ["conv_rate", "trips"],
                timedelta(days=1),
                2,
            ),
        ],
    ).to_pyarrow()

    assert pa.types.is_list(actual.schema.field("conv_rate").type)
    rows = actual.to_pylist()
    assert len(rows) == 3
    by_key = {(row["driver_id"], str(row["event_timestamp"])): row for row in rows}
    # Newest 2 events as of 10:59, newest first; 11:45 is future data. The
    # null stays in place so both lists line up event by event.
    latest = by_key[(1001, "2024-01-01 10:59:00+00:00")]
    assert (latest["conv_rate"], latest["trips"]) == ([0.78, 0.71], [None, 3])
    # Ragged: as of 10:20 only two events exist in total.
    assert by_key[(1001, "2024-01-01 10:20:00+00:00")]["conv_rate"] == [0.66, 0.60]
    # Entity with no events gets an empty sequence, like online serving.
    assert by_key[(1002, "2024-01-01 10:59:00+00:00")]["conv_rate"] == []


def test_point_in_time_join_sequence_max_age_window():
    """online_config.max_age bounds the sequence lookback, mirroring
    online_read_sequence."""
    entity_table = pa.table(
        {
            "driver_id": [1001],
            "event_timestamp": [pa_datetime(2024, 1, 1, 10, 59)],
        }
    )

    actual = point_in_time_join(
        ibis.memtable(entity_table),
        feature_tables=[
            (
                ibis.memtable(sequence_events_table()),
                "event_timestamp",
                "created",
                {"driver_id": "driver_id"},
                ["conv_rate"],
                timedelta(minutes=30),  # max_age window instead of the fv ttl
                5,
            ),
        ],
    ).to_pyarrow()

    # As of 10:59 with a 30 minute window: only 10:30 and 10:45 qualify.
    assert actual.to_pylist()[0]["conv_rate"] == [0.78, 0.71]


def test_get_historical_features_ibis_sequence_mode():
    """Sequence feature views return list columns by default, capped at
    online_config.max_length, newest first and point-in-time correct."""
    driver = Entity(
        name="driver_id", join_keys=["driver_id"], value_type=ValueType.INT64
    )
    driver_stats = FeatureView(
        name="driver_stats",
        entities=[driver],
        schema=[Field(name="conv_rate", dtype=Float64)],
        source=FileSource(
            path="unused.parquet",
            timestamp_field="event_timestamp",
            created_timestamp_column="created",
        ),
        online_config=OnlineConfig(
            mode="sequence",
            write_mode="append",
            max_length=3,
        ),
    )
    entity_df = pd.DataFrame(
        {
            "driver_id": [1001],
            "event_timestamp": [datetime(2024, 1, 1, 10, 59, tzinfo=timezone.utc)],
        }
    )

    def retrieve(**kwargs):
        return get_historical_features_ibis(
            config=MagicMock(),
            feature_views=[driver_stats],
            feature_refs=["driver_stats:conv_rate"],
            entity_df=entity_df,
            registry=MagicMock(),
            project="test",
            data_source_reader=lambda *_: ibis.memtable(sequence_events_table()),
            data_source_writer=MagicMock(),
            **kwargs,
        )

    # Default: the online_config.max_length newest events as of 10:59.
    default_df = retrieve().to_df()
    assert default_df["conv_rate"].iloc[0].tolist() == [0.78, 0.71, 0.66]

    # Override: fewer events for this training run.
    override_df = retrieve(sequence_length=2).to_df()
    assert override_df["conv_rate"].iloc[0].tolist() == [0.78, 0.71]

    # An override above max_length is capped, as online reads are.
    capped_df = retrieve(sequence_length=10).to_df()
    assert capped_df["conv_rate"].iloc[0].tolist() == [0.78, 0.71, 0.66]

    # Arrow output uses list arrays.
    arrow = retrieve().to_arrow()
    assert pa.types.is_list(arrow.schema.field("conv_rate").type)


def test_get_historical_features_ibis_sequence_max_age():
    """The sequence lookback window is online_config.max_age, not the feature
    view ttl, matching what online_read_sequence serves."""
    driver = Entity(
        name="driver_id", join_keys=["driver_id"], value_type=ValueType.INT64
    )
    driver_stats = FeatureView(
        name="driver_stats",
        entities=[driver],
        schema=[Field(name="conv_rate", dtype=Float64)],
        source=FileSource(
            path="unused.parquet",
            timestamp_field="event_timestamp",
            created_timestamp_column="created",
        ),
        ttl=timedelta(days=365),
        online_config=OnlineConfig(
            mode="sequence",
            write_mode="append",
            max_length=3,
            max_age=timedelta(minutes=30),
        ),
    )
    entity_df = pd.DataFrame(
        {
            "driver_id": [1001],
            "event_timestamp": [datetime(2024, 1, 1, 10, 59, tzinfo=timezone.utc)],
        }
    )

    df = get_historical_features_ibis(
        config=MagicMock(),
        feature_views=[driver_stats],
        feature_refs=["driver_stats:conv_rate"],
        entity_df=entity_df,
        registry=MagicMock(),
        project="test",
        data_source_reader=lambda *_: ibis.memtable(sequence_events_table()),
        data_source_writer=MagicMock(),
    ).to_df()

    # ttl would allow all five events; max_age keeps only the last half hour.
    assert df["conv_rate"].iloc[0].tolist() == [0.78, 0.71]


def test_get_historical_features_ibis_latest_mode_unchanged():
    """Feature views without sequence online_config keep returning a single
    latest value."""
    driver = Entity(
        name="driver_id", join_keys=["driver_id"], value_type=ValueType.INT64
    )
    driver_stats = FeatureView(
        name="driver_stats",
        entities=[driver],
        schema=[Field(name="conv_rate", dtype=Float64)],
        source=FileSource(
            path="unused.parquet",
            timestamp_field="event_timestamp",
            created_timestamp_column="created",
        ),
    )
    entity_df = pd.DataFrame(
        {
            "driver_id": [1001],
            "event_timestamp": [datetime(2024, 1, 1, 10, 59, tzinfo=timezone.utc)],
        }
    )

    df = get_historical_features_ibis(
        config=MagicMock(),
        feature_views=[driver_stats],
        feature_refs=["driver_stats:conv_rate"],
        entity_df=entity_df,
        registry=MagicMock(),
        project="test",
        data_source_reader=lambda *_: ibis.memtable(sequence_events_table()),
        data_source_writer=MagicMock(),
    ).to_df()

    assert df["conv_rate"].iloc[0] == 0.78
    assert isinstance(df["conv_rate"].iloc[0], float)


@pytest.mark.parametrize(
    "dtype",
    ["datetime64[ns, UTC]", "datetime64[us, UTC]", "datetime64[ns]", "datetime64[us]"],
)
@pytest.mark.parametrize(
    "pull", [pull_latest_from_table_or_query_ibis, pull_all_from_table_or_query_ibis]
)
def test_pull_filters_parquet_timestamps_of_any_precision(tmp_path, dtype, pull):
    """Materialize pulls read what pandas writes: nanosecond and zone-aware included."""
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    path = tmp_path / "source.parquet"
    pd.DataFrame(
        {
            "driver_id": [1, 2, 3, 4],
            "event_timestamp": [start + timedelta(hours=h) for h in range(4)],
            "trips": [10, 20, 30, 40],
        }
    ).astype(
        {"event_timestamp": dtype if "UTC" in dtype else "datetime64[ns, UTC]"}
    ).pipe(
        lambda df: (
            df
            if "UTC" in dtype
            else df.assign(
                event_timestamp=df.event_timestamp.dt.tz_localize(None).astype(dtype)
            )
        )
    ).to_parquet(path)

    table = pull(
        config=MagicMock(),
        data_source=MagicMock(),
        join_key_columns=["driver_id"],
        feature_name_columns=["trips"],
        timestamp_field="event_timestamp",
        created_timestamp_column=None,
        start_date=start + timedelta(hours=1),
        end_date=start + timedelta(hours=2),
        data_source_reader=lambda *_: ibis.read_parquet(path),
        data_source_writer=MagicMock(),
    ).to_arrow()

    assert sorted(table["driver_id"].to_pylist()) == [2, 3]


def test_get_historical_features_keeps_join_keys_left_out_of_join_key_map():
    """A join_key_map that renames only some join keys still joins on the others."""
    user = Entity(name="user", join_keys=["user_id"], value_type=ValueType.INT64)
    merchant = Entity(
        name="merchant", join_keys=["merchant_id"], value_type=ValueType.INT64
    )
    user_merchant = FeatureView(
        name="user_merchant",
        entities=[user, merchant],
        schema=[
            Field(name="user_id", dtype=Int64),
            Field(name="merchant_id", dtype=Int64),
            Field(name="spend", dtype=Int64),
        ],
        source=FileSource(path="unused.parquet", timestamp_field="event_timestamp"),
    ).with_join_key_map({"user_id": "sender_id"})
    features = pa.table(
        {
            "user_id": [1, 1],
            "merchant_id": [100, 200],
            "event_timestamp": [pa_datetime(2024, 1, 1), pa_datetime(2024, 1, 2)],
            "spend": [10, 99],
        }
    )
    entity_df = pd.DataFrame(
        {
            "sender_id": [1, 1],
            "merchant_id": [100, 200],
            "event_timestamp": [datetime(2024, 1, 3, tzinfo=timezone.utc)] * 2,
        }
    )

    actual = get_historical_features_ibis(
        config=MagicMock(),
        feature_views=[user_merchant],
        feature_refs=["user_merchant:spend"],
        entity_df=entity_df,
        registry=MagicMock(),
        project="test",
        data_source_reader=lambda *_: ibis.memtable(features),
        data_source_writer=MagicMock(),
    ).to_df()

    spend = dict(zip(actual["merchant_id"], actual["spend"]))
    assert spend == {100: 10, 200: 99}
