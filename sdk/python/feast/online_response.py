# Copyright 2020 The Feast Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import uuid as uuid_module
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import pandas as pd
import pyarrow as pa

from feast.feature_view import DUMMY_ENTITY_ID
from feast.protos.feast.serving.ServingService_pb2 import (
    GetOnlineFeaturesResponse,
    SequenceEvent,
)
from feast.torch_wrapper import get_torch
from feast.type_map import feast_value_type_to_python_type
from feast.value_type import ValueType

if TYPE_CHECKING:
    from torch import Tensor as TorchTensor
else:
    TorchTensor = Any

TIMESTAMP_POSTFIX: str = "__ts"
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


class OnlineResponse:
    """
    Defines an online response in feast.
    """

    def __init__(
        self,
        online_response_proto: GetOnlineFeaturesResponse,
        feature_types: Optional[Dict[str, ValueType]] = None,
    ):
        """
        Construct a native online response from its protobuf version.

        Args:
        online_response_proto: GetOnlineResponse proto object to construct from.
        feature_types: Optional mapping of feature names to ValueType for type-aware deserialization.
        """
        self.proto = online_response_proto
        self._feature_types = feature_types or {}
        # Delete DUMMY_ENTITY_ID from proto if it exists
        for idx, val in enumerate(self.proto.metadata.feature_names.val):
            if val == DUMMY_ENTITY_ID:
                del self.proto.metadata.feature_names.val[idx]
                del self.proto.results[idx]

                break

    def to_dict(self, include_event_timestamps: bool = False) -> Dict[str, Any]:
        """
        Converts GetOnlineFeaturesResponse features into a dictionary form.

        Args:
        include_event_timestamps: bool Optionally include feature timestamps in the dictionary
        """
        response: Dict[str, List[Any]] = {}

        for feature_ref, feature_vector in zip(
            self.proto.metadata.feature_names.val, self.proto.results
        ):
            feature_type = self._feature_types.get(feature_ref)
            response[feature_ref] = [
                feast_value_type_to_python_type(v, feature_type)
                for v in feature_vector.values
            ]

            if include_event_timestamps:
                timestamp_ref = feature_ref + TIMESTAMP_POSTFIX
                response[timestamp_ref] = [
                    ts.seconds for ts in feature_vector.event_timestamps
                ]

        return response

    def to_sequences(self) -> Dict[str, List[List[Dict[str, Any]]]]:
        """
        Converts the sequences of sequence-mode feature views into a dictionary.

        Sequences are only returned when get_online_features is called with
        sequence_length above 1.

        Returns:
            Feature view name -> one list of events per entity row, in entity row
            order. Each event is a dict with "event_timestamp" (a UTC datetime)
            and one key per feature.
        """
        response: Dict[str, List[List[Dict[str, Any]]]] = {}
        for sequence in self.proto.sequences:
            response.setdefault(sequence.feature_view, []).append(
                [self._sequence_event_to_dict(event) for event in sequence.events]
            )
        return response

    def to_sequence_df(self) -> pd.DataFrame:
        """
        Converts the sequences of sequence-mode feature views into a Pandas
        dataframe with one row per event.

        Columns are "feature_view", the entity join keys, "event_timestamp" and
        the features. Entity rows without events have no rows.
        """
        rows = []
        for sequence in self.proto.sequences:
            entity_key = {
                name: feast_value_type_to_python_type(value)
                for name, value in sequence.entity_key.items()
            }
            for event in sequence.events:
                rows.append(
                    {
                        "feature_view": sequence.feature_view,
                        **entity_key,
                        **self._sequence_event_to_dict(event),
                    }
                )
        return pd.DataFrame(rows)

    def _sequence_event_to_dict(self, event: SequenceEvent) -> Dict[str, Any]:
        return {
            "event_timestamp": _EPOCH
            + timedelta(milliseconds=event.event_timestamp_ms),
            **{
                name: feast_value_type_to_python_type(
                    value, self._feature_types.get(name)
                )
                for name, value in event.features.items()
            },
        }

    def to_df(self, include_event_timestamps: bool = False) -> pd.DataFrame:
        """
        Converts GetOnlineFeaturesResponse features into Panda dataframe form.

        Args:
        include_event_timestamps: bool Optionally include feature timestamps in the dataframe
        """

        return pd.DataFrame(self.to_dict(include_event_timestamps))

    def to_arrow(self, include_event_timestamps: bool = False) -> pa.Table:
        """
        Converts GetOnlineFeaturesResponse features into pyarrow Table.

        Args:
        include_event_timestamps: bool Optionally include feature timestamps in the table
        """
        result = self.to_dict(include_event_timestamps)
        result = _convert_uuids_for_arrow(result)
        return pa.Table.from_pydict(result)

    def to_tensor(
        self,
        kind: str = "torch",
        default_value: Any = float("nan"),
    ) -> Dict[str, Union[TorchTensor, List[Any]]]:
        """
        Converts GetOnlineFeaturesResponse features into a dictionary of tensors or lists.

        - Numeric features (int, float, bool) -> torch.Tensor
        - Non-numeric features (e.g., strings) -> list[Any]

        Args:
            kind: Backend tensor type. Currently only "torch" is supported.
            default_value: Value to substitute for missing (None) entries.

        Returns:
            Dict[str, Union[torch.Tensor, List[Any]]]: Mapping of feature names to tensors or lists.
        """
        if kind != "torch":
            raise ValueError(
                f"Unsupported tensor kind: {kind}. Only 'torch' is supported currently."
            )
        torch = get_torch()
        feature_dict = self.to_dict(include_event_timestamps=False)
        feature_keys = set(self.proto.metadata.feature_names.val)
        tensor_dict: Dict[str, Union[TorchTensor, List[Any]]] = {}
        for key in feature_keys:
            raw_values = feature_dict[key]
            # Infer the feature type from the retrieved values, not from the
            # substituted ones: default_value is a stand-in for a missing entity
            # and says nothing about the feature's actual type.
            first_valid = next((v for v in raw_values if v is not None), default_value)
            values = [v if v is not None else default_value for v in raw_values]
            if isinstance(first_valid, (int, float, bool)):
                try:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    tensor_dict[key] = torch.tensor(values, device=device)
                except Exception as e:
                    raise ValueError(
                        f"Failed to convert values for '{key}' to tensor: {e}"
                    )
            else:
                tensor_dict[key] = (
                    values  # Return as-is for strings or unsupported types
                )
        return tensor_dict


def _convert_uuids_for_arrow(result: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    """Convert uuid.UUID objects and sets to Arrow-compatible types."""
    for key, values in result.items():
        first_valid = next((v for v in values if v is not None), None)
        if isinstance(first_valid, uuid_module.UUID):
            result[key] = [
                str(v) if isinstance(v, uuid_module.UUID) else v for v in values
            ]
        elif isinstance(first_valid, list):
            inner = next((e for e in first_valid if e is not None), None)
            if isinstance(inner, uuid_module.UUID):
                result[key] = [
                    [str(e) if isinstance(e, uuid_module.UUID) else e for e in v]
                    if isinstance(v, list)
                    else v
                    for v in values
                ]
        elif isinstance(first_valid, set):
            inner = next((e for e in first_valid if e is not None), None)
            if isinstance(inner, uuid_module.UUID):
                result[key] = [
                    [str(e) for e in v] if isinstance(v, set) else v for v in values
                ]
            else:
                result[key] = [list(v) if isinstance(v, set) else v for v in values]
    return result
