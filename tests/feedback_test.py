import pytest

from local_rag.feedback import (
    apply_feedback,
    create_feedback_event,
    normalize_feedback,
)


def test_normalize_store_feedback():
    assert normalize_feedback("kaydet") == "store"
    assert normalize_feedback("yararlı") == "store"
    assert normalize_feedback("beğendim") == "store"


def test_normalize_ignore_feedback():
    assert normalize_feedback("kaydetme") == "ignore"
    assert normalize_feedback("yararsız") == "ignore"
    assert normalize_feedback("beğenmedim") == "ignore"


def test_apply_feedback_increases_score_for_store_feedback():
    assert apply_feedback(0.50, "kaydet") == 0.60


def test_apply_feedback_decreases_score_for_ignore_feedback():
    assert apply_feedback(0.50, "kaydetme") == 0.35


def test_apply_feedback_clamps_to_valid_range():
    assert apply_feedback(0.95, "yararlı") == 1.0
    assert apply_feedback(0.05, "yararsız") == 0.0


def test_apply_feedback_rejects_invalid_importance():
    with pytest.raises(ValueError):
        apply_feedback(1.2, "kaydet")


def test_normalize_feedback_rejects_unknown_value():
    with pytest.raises(ValueError):
        normalize_feedback("emin değilim")


def test_normalize_feedback_rejects_non_text_values():
    invalid_values = [True, False, 1, 0, "👍", "👎"]

    for value in invalid_values:
        with pytest.raises(ValueError):
            normalize_feedback(value)


def test_create_feedback_event_returns_explainable_result():
    event = create_feedback_event(
        memory_id="memory-123",
        importance=0.50,
        feedback="beğendim",
        reason="Cevap kişisel tercihi doğru kullandı.",
    )

    assert event.memory_id == "memory-123"
    assert event.feedback == "store"
    assert event.previous_importance == 0.50
    assert event.updated_importance == 0.60
    assert event.reason == "Cevap kişisel tercihi doğru kullandı."
