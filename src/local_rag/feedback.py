import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from local_rag.importance import ImportanceDecision


DEFAULT_FEEDBACK_PATH = Path("data/importance_feedback.jsonl")
STORE_DELTA = 0.10
IGNORE_DELTA = -0.15

STORE_FEEDBACK_VALUES = {
    "store",
    "save",
    "keep",
    "kaydet",
    "sakla",
    "hatırla",
    "evet",
    "yes",
    "doğru",
    "dogru",
    "uygun",
    "yararlı",
    "faydali",
    "faydalı",
    "beğendim",
    "begendim",
}

IGNORE_FEEDBACK_VALUES = {
    "ignore",
    "skip",
    "discard",
    "kaydetme",
    "saklama",
    "unut",
    "hayır",
    "hayir",
    "no",
    "yanlış",
    "yanlis",
    "uygunsuz",
    "gereksiz",
    "yararsız",
    "yararsiz",
    "beğenmedim",
    "begenmedim",
}


@dataclass(frozen=True)
class FeedbackEvent:
    """Bir hafıza kaydına uygulanan geri bildirim sonucunu tutar."""

    memory_id: str
    feedback: str
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


def normalize_feedback(feedback: str) -> str:
    """Geri bildirimi standart store/ignore etiketine çevirir."""

    if not isinstance(feedback, str):
        raise ValueError("feedback metin olarak verilmelidir")

    normalized = str(feedback).strip().lower()

    if normalized in STORE_FEEDBACK_VALUES:
        return "store"

    if normalized in IGNORE_FEEDBACK_VALUES:
        return "ignore"

    raise ValueError(f"Bilinmeyen feedback değeri: {feedback!r}")


def _normalize_feedback(user_feedback: str) -> str:
    """Kullanıcı feedback değerini standart hale getirir."""

    return normalize_feedback(user_feedback)


def apply_feedback(
    importance: float,
    feedback: str,
    *,
    store_delta: float = STORE_DELTA,
    ignore_delta: float = IGNORE_DELTA,
) -> float:
    """Mevcut önem skorunu geri bildirime göre günceller."""

    if not 0.0 <= importance <= 1.0:
        raise ValueError("importance 0.0 ile 1.0 arasında olmalı")

    normalized_feedback = normalize_feedback(feedback)
    delta = store_delta if normalized_feedback == "store" else ignore_delta

    return _clamp_score(importance + delta)


def create_feedback_event(
    memory_id: str,
    importance: float,
    feedback: str,
    *,
    reason: str | None = None,
) -> FeedbackEvent:
    """Geri bildirim sonucunu açıklanabilir bir event olarak döndürür."""

    normalized_feedback = normalize_feedback(feedback)
    updated_importance = apply_feedback(importance, normalized_feedback)

    return FeedbackEvent(
        memory_id=memory_id,
        feedback=normalized_feedback,
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
    final_should_store = normalized_feedback == "store"

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
