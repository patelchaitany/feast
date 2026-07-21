"""Tests for CVE-2026-56121: skip_udf prevents dill.loads before authorization."""

import dill
import pytest

from feast.feature_view import FeatureView
from feast.infra.offline_stores.file_source import FileSource
from feast.on_demand_feature_view import OnDemandFeatureView
from feast.protos.feast.core.FeatureView_pb2 import FeatureView as FeatureViewProto
from feast.protos.feast.core.FeatureView_pb2 import (
    FeatureViewSpec as FeatureViewSpecProto,
)
from feast.protos.feast.core.OnDemandFeatureView_pb2 import (
    OnDemandFeatureView as OnDemandFeatureViewProto,
)
from feast.protos.feast.core.OnDemandFeatureView_pb2 import (
    OnDemandFeatureViewSpec,
    OnDemandSource,
)
from feast.protos.feast.core.Transformation_pb2 import (
    FeatureTransformationV2 as FeatureTransformationProto,
)
from feast.protos.feast.core.Transformation_pb2 import (
    UserDefinedFunctionV2 as UserDefinedFunctionProto,
)
from feast.transformation.pandas_transformation import PandasTransformation
from feast.transformation.python_transformation import PythonTransformation

_FILE_SOURCE_PROTO = FileSource(name="test_source", path="test.parquet").to_proto()


def _make_udf_proto(body: bytes, mode: str = "python") -> UserDefinedFunctionProto:
    return UserDefinedFunctionProto(
        name="test_func",
        body=body,
        body_text="def test_func(x): return x",
        mode=mode,
    )


def _valid_udf_body() -> bytes:
    return dill.dumps(lambda x: x, recurse=True)


def _poison_udf_body() -> bytes:
    return b"INVALID_DILL_PAYLOAD_THAT_WOULD_CRASH"


class TestPythonTransformationSkipUdf:
    def test_skip_udf_does_not_call_dill_loads(self):
        proto = _make_udf_proto(_poison_udf_body(), mode="python")
        t = PythonTransformation.from_proto(proto, skip_udf=True)
        assert t is not None
        assert t.udf_string == "def test_func(x): return x"

    def test_skip_udf_placeholder_raises(self):
        proto = _make_udf_proto(_poison_udf_body(), mode="python")
        t = PythonTransformation.from_proto(proto, skip_udf=True)
        with pytest.raises(RuntimeError, match="skip_udf"):
            t.udf("input")

    def test_skip_udf_roundtrip_preserves_bytes(self):
        original_body = _valid_udf_body()
        proto = _make_udf_proto(original_body, mode="python")
        t = PythonTransformation.from_proto(proto, skip_udf=True)
        roundtrip_proto = t.to_proto()
        assert roundtrip_proto.body == original_body

    def test_normal_deserialization_still_works(self):
        body = dill.dumps(lambda x: {"out": 1}, recurse=True)
        proto = _make_udf_proto(body, mode="python")
        t = PythonTransformation.from_proto(proto, skip_udf=False)
        assert t.udf({"in": 0}) == {"out": 1}


class TestPandasTransformationSkipUdf:
    def test_skip_udf_does_not_call_dill_loads(self):
        proto = _make_udf_proto(_poison_udf_body(), mode="pandas")
        t = PandasTransformation.from_proto(proto, skip_udf=True)
        assert t is not None

    def test_skip_udf_placeholder_raises(self):
        proto = _make_udf_proto(_poison_udf_body(), mode="pandas")
        t = PandasTransformation.from_proto(proto, skip_udf=True)
        with pytest.raises(RuntimeError, match="skip_udf"):
            t.udf("input")

    def test_skip_udf_roundtrip_preserves_bytes(self):
        original_body = _valid_udf_body()
        proto = _make_udf_proto(original_body, mode="pandas")
        t = PandasTransformation.from_proto(proto, skip_udf=True)
        roundtrip_proto = t.to_proto()
        assert roundtrip_proto.body == original_body


class TestFeatureViewSkipUdf:
    def _make_feature_view_proto_with_udf(self, udf_body: bytes):
        udf_proto = _make_udf_proto(udf_body, mode="python")
        spec = FeatureViewSpecProto(
            name="test_fv",
            feature_transformation=FeatureTransformationProto(
                user_defined_function=udf_proto,
            ),
        )
        spec.batch_source.CopyFrom(_FILE_SOURCE_PROTO)
        return FeatureViewProto(spec=spec)

    def test_skip_udf_prevents_dill_loads(self):
        proto = self._make_feature_view_proto_with_udf(_poison_udf_body())
        fv = FeatureView.from_proto(proto, skip_udf=True)
        assert fv is not None
        assert fv.name == "test_fv"

    def test_skip_udf_propagates_to_source_views(self):
        child_udf_proto = _make_udf_proto(_poison_udf_body(), mode="python")
        child_spec = FeatureViewSpecProto(
            name="child_fv",
            feature_transformation=FeatureTransformationProto(
                user_defined_function=child_udf_proto,
            ),
        )
        child_spec.batch_source.CopyFrom(_FILE_SOURCE_PROTO)
        parent_spec = FeatureViewSpecProto(
            name="parent_fv",
            source_views=[child_spec],
        )
        parent_spec.batch_source.CopyFrom(_FILE_SOURCE_PROTO)
        parent_proto = FeatureViewProto(spec=parent_spec)
        fv = FeatureView.from_proto(parent_proto, skip_udf=True)
        assert fv is not None
        assert fv.name == "parent_fv"


class TestOnDemandFeatureViewSkipUdf:
    def test_skip_udf_passes_to_source_feature_views(self):
        source_udf_proto = _make_udf_proto(_poison_udf_body(), mode="python")
        source_fv_spec = FeatureViewSpecProto(
            name="source_fv",
            feature_transformation=FeatureTransformationProto(
                user_defined_function=source_udf_proto,
            ),
        )
        source_fv_spec.batch_source.CopyFrom(_FILE_SOURCE_PROTO)
        source_fv_proto = FeatureViewProto(spec=source_fv_spec)

        odfv_udf_proto = _make_udf_proto(_poison_udf_body(), mode="pandas")
        odfv_spec = OnDemandFeatureViewSpec(
            name="test_odfv",
            sources={"src": OnDemandSource(feature_view=source_fv_proto)},
            feature_transformation=FeatureTransformationProto(
                user_defined_function=odfv_udf_proto,
            ),
            mode="pandas",
        )
        odfv_proto = OnDemandFeatureViewProto(spec=odfv_spec)
        odfv = OnDemandFeatureView.from_proto(odfv_proto, skip_udf=True)
        assert odfv is not None
        assert odfv.name == "test_odfv"
