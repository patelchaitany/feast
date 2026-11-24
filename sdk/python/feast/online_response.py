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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import pandas as pd
import pyarrow as pa

from feast.errors import FeatureServiceTextOutputError
from feast.feature_view import DUMMY_ENTITY_ID
from feast.protos.feast.serving.ServingService_pb2 import GetOnlineFeaturesResponse
from feast.torch_wrapper import get_torch
from feast.type_map import feast_value_type_to_python_type
from feast.value_type import ValueType

if TYPE_CHECKING:
    from torch import Tensor as TorchTensor
else:
    TorchTensor = Any

TIMESTAMP_POSTFIX: str = "__ts"


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

    def to_text(self) -> str:
        """
        Returns the plain-text output of a text-mode feature service.

        The feature service must have ``output_format="text"``, and the request
        a single entity row. A null value reads as the empty string. For several
        rows, ``proto.text_response`` holds one string per row.

        Raises:
            FeatureServiceTextOutputError: The response has no text, or more
                than one entity row.
        """
        texts = self.proto.text_response
        if not texts:
            raise FeatureServiceTextOutputError(
                "to_text() needs features from a FeatureService with "
                "output_format='text'; this response has no text."
            )
        if len(texts) != 1:
            raise FeatureServiceTextOutputError(
                f"to_text() returns the text of one entity row, but this response "
                f"holds {len(texts)}; read proto.text_response for all of them."
            )
        return texts[0]

    def _fill_text_response(self, output_field: str) -> None:
        """Copy ``output_field`` into ``proto.text_response``, one string per row."""
        names = self.proto.metadata.feature_names.val
        # Full feature names prefix the field with its view: "view__field".
        positions = [
            i
            for i, name in enumerate(names)
            if name == output_field or name.endswith(f"__{output_field}")
        ]
        if len(positions) != 1:
            raise FeatureServiceTextOutputError(
                f"The response has no single feature '{output_field}' to return "
                f"as text; it has {list(names)}."
            )
        texts = []
        for value in self.proto.results[positions[0]].values:
            kind = value.WhichOneof("val")
            if kind not in (None, "string_val", "null_val"):
                raise FeatureServiceTextOutputError(
                    f"Feature '{output_field}' is returned as text, so it must "
                    f"hold strings, but a row holds {kind}."
                )
            texts.append(value.string_val)
        del self.proto.text_response[:]
        self.proto.text_response.extend(texts)

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
