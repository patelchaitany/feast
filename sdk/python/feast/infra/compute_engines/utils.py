from datetime import datetime
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from feast.data_source import DataSource
from feast.infra.compute_engines.dag.context import ColumnInfo, ExecutionContext
from feast.infra.offline_stores.offline_store import RetrievalJob
from feast.infra.offline_stores.offline_utils import (
    DEFAULT_ENTITY_DF_EVENT_TIMESTAMP_COL,
    infer_event_timestamp_from_entity_df,
)
from feast.infra.online_stores.online_store import OnlineStore
from feast.online_config import uses_append_write_mode
from feast.protos.feast.types.EntityKey_pb2 import EntityKey as EntityKeyProto
from feast.protos.feast.types.Value_pb2 import Value as ValueProto
from feast.repo_config import RepoConfig

ENTITY_TS_ALIAS = "__entity_event_timestamp"
ENTITY_ROW_ID = "__feast_entity_row_id"


def infer_entity_timestamp_column(entity_schema: Mapping[str, Any]) -> str:
    """Resolve the entity timestamp column used for point-in-time joins."""
    if ENTITY_TS_ALIAS in entity_schema:
        return ENTITY_TS_ALIAS
    return infer_event_timestamp_from_entity_df(dict(entity_schema))


def find_entity_timestamp_column(columns: Sequence[str]) -> Optional[str]:
    """Find the timestamp column in an entity DataFrame schema, if present."""
    if ENTITY_TS_ALIAS in columns:
        return ENTITY_TS_ALIAS
    if DEFAULT_ENTITY_DF_EVENT_TIMESTAMP_COL in columns:
        return DEFAULT_ENTITY_DF_EVENT_TIMESTAMP_COL
    return None


def create_offline_store_retrieval_job(
    data_source: DataSource,
    column_info: ColumnInfo,
    context: ExecutionContext,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    pull_all: bool = False,
) -> RetrievalJob:
    """
    Create a retrieval job for the offline store.
    Args:
        data_source: The data source to pull from.
        column_info: Column information containing join keys, feature columns, and timestamps.
        context:
        start_time:
        end_time:
        pull_all: Pull every row even when pull_latest_features is set. Sequence-mode
            feature views need this to keep all events per entity.
    Returns:

    """
    offline_store = context.offline_store

    pull_latest = (
        not pull_all and context.repo_config.materialization_config.pull_latest_features
    )

    if pull_latest:
        if not start_time or not end_time:
            raise ValueError(
                "start_time and end_time must be provided when pull_latest_features is True"
            )

        retrieval_job = offline_store.pull_latest_from_table_or_query(
            config=context.repo_config,
            data_source=data_source,
            join_key_columns=column_info.join_keys,
            feature_name_columns=column_info.feature_cols,
            timestamp_field=column_info.ts_col,
            created_timestamp_column=column_info.created_ts_col,
            start_date=start_time,
            end_date=end_time,
        )
    else:
        # 📥 Reuse Feast's robust query resolver
        retrieval_job = offline_store.pull_all_from_table_or_query(
            config=context.repo_config,
            data_source=data_source,
            join_key_columns=column_info.join_keys,
            feature_name_columns=column_info.feature_cols,
            timestamp_field=column_info.ts_col,
            created_timestamp_column=column_info.created_ts_col,
            start_date=start_time,
            end_date=end_time,
        )

    return retrieval_job


def write_rows_to_online_store(
    online_store: OnlineStore,
    config: RepoConfig,
    feature_view: Any,
    data: List[
        Tuple[EntityKeyProto, Dict[str, ValueProto], datetime, Optional[datetime]]
    ],
    progress: Optional[Callable[[int], Any]] = None,
) -> None:
    """Append rows for sequence-mode feature views, and upsert them otherwise."""
    if uses_append_write_mode(feature_view):
        online_store.online_append(config, feature_view, data, progress)
    else:
        online_store.online_write_batch(config, feature_view, data, progress)
