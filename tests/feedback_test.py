import pytest

from local_rag.feedback import (
    apply_feedback,
    create_feedback_event,
    normalize_feedback,
)


def test_normalize_positive_feedback():
    assert normalize_feedback("yararlı") == 1
    assert normalize_feedback(True) == 1
    assert normalize_feedback(1) == 1


def test_normalize_negative_feedback():
    assert normalize_feedback("yararsız") == -1
    assert normalize_feedback(False) == -1
    assert normalize_feedback(0) == -1


def test_apply_feedback_increases_positive_signal():
    assert apply_feedback(0.50, "like") == 0.60


def test_apply_feedback_decreases_negative_signal():
    assert apply_feedback(0.50, "dislike") == 0.35


def test_apply_feedback_clamps_to_valid_range():
    assert apply_feedback(0.95, "yararlı") == 1.0
    assert apply_feedback(0.05, "yararsız") == 0.0


def test_apply_feedback_rejects_invalid_importance():
    with pytest.raises(ValueError):
        apply_feedback(1.2, "like")


def test_normalize_feedback_rejects_unknown_signal():
    with pytest.raises(ValueError):
        normalize_feedback("emin değilim")


def test_create_feedback_event_returns_explainable_result():
    event = create_feedback_event(
        memory_id="memory-123",
        importance=0.50,
        feedback="beğendim",
        reason="Cevap kişisel tercihi doğru kullandı.",
    )

    assert event.memory_id == "memory-123"
    assert event.signal == 1
    assert event.previous_importance == 0.50
    assert event.updated_importance == 0.60
    assert event.reason == "Cevap kişisel tercihi doğru kullandı."
