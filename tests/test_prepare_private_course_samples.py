from scripts.prepare_private_course_samples import (
    EDGE_MARGIN_SECONDS,
    _even_offsets,
    _nonoverlapping_offsets,
    _workload,
)


def test_audio_offsets_are_nonoverlapping_and_inside_margins():
    durations = (900, 900, 1800)
    source_duration = 7200
    offsets = _nonoverlapping_offsets(source_duration, durations)

    assert offsets[0] >= EDGE_MARGIN_SECONDS
    assert offsets[-1] + durations[-1] <= source_duration - EDGE_MARGIN_SECONDS
    assert all(
        offset + duration < next_offset
        for offset, duration, next_offset in zip(
            offsets,
            durations,
            offsets[1:],
        )
    )


def test_frame_offsets_are_even_and_inside_margins():
    offsets = _even_offsets(7200, 6)

    assert len(offsets) == 6
    assert offsets == sorted(offsets)
    assert offsets[0] > EDGE_MARGIN_SECONDS
    assert offsets[-1] < 7200 - EDGE_MARGIN_SECONDS


def test_private_workload_uses_only_generic_sample_metadata():
    items = [{"id": "audio_001", "path": "local/audio_001.wav"}]

    workload = _workload("asr", items)

    assert workload["workload_class"] == "private_course"
    assert workload["warmup_item_id"] == "audio_001"
    assert "source" not in workload
