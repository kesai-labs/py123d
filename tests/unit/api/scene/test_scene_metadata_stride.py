import uuid

from py123d.datatypes.metadata import SceneMetadata


def _make_metadata(stride: int = 1, num_future: int = 10, num_history: int = 2, initial_idx: int = 10) -> SceneMetadata:
    effective_duration = 0.1 * stride
    return SceneMetadata(
        dataset="test",
        split="test_train",
        initial_uuid=str(uuid.uuid4()),
        initial_idx=initial_idx,
        num_future_iterations=num_future,
        num_history_iterations=num_history,
        future_duration_s=num_future * effective_duration,
        history_duration_s=num_history * effective_duration,
        iteration_duration_s=effective_duration,
        target_iteration_stride=stride,
    )


class TestSceneMetadataWithStride:
    def test_end_idx_stride_1_unchanged(self):
        meta = _make_metadata(stride=1, num_future=10, initial_idx=5)
        assert meta.end_idx == 5 + 10 + 1  # 16

    def test_end_idx_with_stride(self):
        meta = _make_metadata(stride=5, num_future=4, initial_idx=10)
        # 10 + 4*5 + 1 = 31
        assert meta.end_idx == 31

    def test_end_idx_with_stride_no_future(self):
        meta = _make_metadata(stride=3, num_future=0, initial_idx=5)
        assert meta.end_idx == 6  # 5 + 0*3 + 1

    def test_total_iterations_unchanged_by_stride(self):
        meta_s1 = _make_metadata(stride=1, num_future=10, num_history=2)
        meta_s5 = _make_metadata(stride=5, num_future=10, num_history=2)
        assert meta_s1.total_iterations == meta_s5.total_iterations == 13

    def test_repr_includes_stride_when_not_1(self):
        meta = _make_metadata(stride=5)
        r = repr(meta)
        assert "target_iteration_stride=5" in r

    def test_repr_omits_stride_when_1(self):
        meta = _make_metadata(stride=1)
        r = repr(meta)
        assert "target_iteration_stride" not in r

    def test_default_stride_is_1(self):
        meta = SceneMetadata(
            dataset="test",
            split="test_train",
            initial_uuid="abc",
            initial_idx=0,
            num_future_iterations=10,
            num_history_iterations=0,
            future_duration_s=1.0,
            history_duration_s=0.0,
            iteration_duration_s=0.1,
        )
        assert meta.target_iteration_stride == 1
