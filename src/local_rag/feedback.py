import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from local_rag.importance import ImportanceDecision


DEFAULT_FEEDBACK_PATH = Path("data/importance_feedback.jsonl")
POSITIVE_DELTA = 0.10
NEGATIVE_DELTA = -0.15

POSITIVE_SIGNALS = {
    "+",
    "+1",
    "1",
    "up",
    "like",
    "liked",
    "good",
    "helpful",
    "positive",
    "yes",
    "true",
    "evet",
    "e",
    "iyi",
    "yararlı",
    "faydali",
    "faydalı",
    "beğendim",
    "begendim",
    "👍",
}

NEGATIVE_SIGNALS = {
    "-",
    "-1",
    "0",
    "down",
    "dislike",
    "disliked",
    "bad",
    "unhelpful",
    "negative",
    "no",
    "false",
    "hayır",
    "hayir",
    "h",
    "kötü",
    "kotu",
    "yararsız",
    "yararsiz",
    "beğenmedim",
    "begenmedim",
    "👎",
}


@dataclass(frozen=True)
class FeedbackEvent:
    """Bir hafıza kaydına uygulanan geri bildirim sonucunu tutar."""

    memory_id: str
    signal: int
    previous_importance: float
    updated_importance: float
    reason: str | None = None


@dataclass(frozen=True)
class FeedbackRecord:
    """Kullanıcı geri bildirimiyle oluşan eğitim verisi kaydı."""

    message: str
    predicted_score: float
    predicted_label: str
    predicted_category: str | None
    predicted_should_store: bool
    user_feedback: str
    final_should_store: bool
    created_at: str


def _clamp_score(score: float) -> float:
    """Skoru 0.0 ile 1.0 arasına sıkıştırır."""

    return max(0.0, min(1.0, score))


def normalize_feedback(feedback: str | int | bool) -> int:
    """Geri bildirim değerini +1 veya -1 sinyaline çevirir."""

    if isinstance(feedback, bool):
        return 1 if feedback else -1

    if isinstance(feedback, int):
        return 1 if feedback > 0 else -1

    normalized = str(feedback).strip().lower()

    if normalized in POSITIVE_SIGNALS:
        return 1

    if normalized in NEGATIVE_SIGNALS:
        return -1

    raise ValueError(f"Bilinmeyen feedback sinyali: {feedback!r}")


def _normalize_feedback(user_feedback: str) -> str:
    """Kullanıcı feedback değerini standart hale getirir."""

    return "up" if normalize_feedback(user_feedback) > 0 else "down"


def apply_feedback(
    importance: float,
    feedback: str | int | bool,
    *,
    positive_delta: float = POSITIVE_DELTA,
    negative_delta: float = NEGATIVE_DELTA,
) -> float:
    """Mevcut önem skorunu geri bildirime göre günceller."""

    if not 0.0 <= importance <= 1.0:
        raise ValueError("importance 0.0 ile 1.0 arasında olmalı")

    signal = normalize_feedback(feedback)
    delta = positive_delta if signal > 0 else negative_delta

    return _clamp_score(importance + delta)


def create_feedback_event(
    memory_id: str,
    importance: float,
    feedback: str | int | bool,
    *,
    reason: str | None = None,
) -> FeedbackEvent:
    """Geri bildirim sonucunu açıklanabilir bir event olarak döndürür."""

    signal = normalize_feedback(feedback)
    updated_importance = apply_feedback(importance, signal)

    return FeedbackEvent(
        memory_id=memory_id,
        signal=signal,
        previous_importance=importance,
        updated_importance=updated_importance,
        reason=reason,
    )


def create_feedback_record(
    message: str,
    decision: ImportanceDecision,
    user_feedback: str,
) -> FeedbackRecord:
    """Mesaj, sistem kararı ve kullanıcı feedback'inden kayıt oluşturur."""

    normalized_feedback = _normalize_feedback(user_feedback)
    final_should_store = normalized_feedback == "up"

    return FeedbackRecord(
        message=message,
        predicted_score=decision.score,
        predicted_label=decision.label,
        predicted_category=decision.category,
        predicted_should_store=decision.should_store,
        user_feedback=normalized_feedback,
        final_should_store=final_should_store,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def save_feedback_record(
    record: FeedbackRecord,
    path: Path = DEFAULT_FEEDBACK_PATH,
) -> None:
    """Feedback kaydını JSONL dosyasına ekler."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def save_feedback(
    message: str,
    decision: ImportanceDecision,
    user_feedback: str,
    path: Path = DEFAULT_FEEDBACK_PATH,
) -> FeedbackRecord:
    """Feedback kaydı oluşturur ve JSONL dosyasına kaydeder."""

    record = create_feedback_record(
        message=message,
        decision=decision,
        user_feedback=user_feedback,
    )

    save_feedback_record(record, path)

    return record


def load_feedback_records(
    path: Path = DEFAULT_FEEDBACK_PATH,
) -> list[FeedbackRecord]:
    """JSONL dosyasından feedback kayıtlarını okur."""

    if not path.exists():
        return []

    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            data = json.loads(line)
            records.append(FeedbackRecord(**data))

    return records
