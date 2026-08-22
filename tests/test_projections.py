from local_inference_bench.projections import audio_projection, ocr_projection


def test_audio_projection_uses_real_time_ratio():
    result = audio_projection(100, 20)
    assert result["real_time_factor"] == 0.2
    assert result["projected_wall_hours"]["2.5"] == 0.5


def test_ocr_projection_uses_images_per_hour():
    result = ocr_projection(10, 5)
    assert result["images_per_hour"] == 7200
    assert result["projected_wall_hours"]["50-80"] == [50 / 7200, 80 / 7200]
