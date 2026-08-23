from scripts.check_ocrllm_image_facade import (
    _looks_formula_like,
    _normalize_visible,
)


def test_formula_like_detection_and_normalization():
    assert _looks_formula_like(r"x = \frac{a}{b}")
    assert not _looks_formula_like("ordinary heading")
    assert _normalize_visible(" CPU\n二十四 ") == "cpu二十四"
