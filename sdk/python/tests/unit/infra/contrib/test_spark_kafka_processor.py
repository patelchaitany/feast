from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock

import pandas as pd
import pytest

from feast.data_source import PushMode
from feast.online_config import OnlineConfig

pytest.importorskip("pyspark")

from feast.infra.contrib.spark_kafka_processor import (  # noqa: E402
    SparkKafkaProcessor,
)


def _written_rows(online_config: Optional[OnlineConfig]) -> pd.DataFrame:
    """Run one micro-batch through batch_write and return the rows written online."""
    processor = SparkKafkaProcessor.__new__(SparkKafkaProcessor)
    processor.fs = MagicMock()
    processor.sfv = SimpleNamespace(
        name="events", timestamp_field="event_timestamp", online_config=online_config
    )
    processor.join_keys = ["user_id"]
    processor.preprocess_fn = None
    processor.processing_time = "1 second"
    processor.query_timeout = None

    stream = MagicMock()
    processor._write_stream_data(stream, PushMode.ONLINE)
    batch_write = stream.writeStream.outputMode.return_value.option.return_value.trigger.return_value.foreachBatch.call_args.args[
        0
    ]

    t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    micro_batch = MagicMock()
    micro_batch.toPandas.return_value = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2"],
            "item_id": ["i1", "i2", "i3", "i9"],
            "event_timestamp": [t1, t1 + timedelta(1), t1 + timedelta(2), t1],
        }
    )
    batch_write(micro_batch, 0)

    return processor.fs.write_to_online_store.call_args.args[1]


def test_batch_write_keeps_every_event_for_sequence_views():
    sequence = _written_rows(
        OnlineConfig(mode="sequence", max_length=10, write_mode="append")
    )
    latest = _written_rows(None)

    assert sorted(sequence["item_id"]) == ["i1", "i2", "i3", "i9"]
    assert sorted(latest["item_id"]) == ["i3", "i9"]
