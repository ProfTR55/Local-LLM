"""Feedback verisiyle importance classifier eğitir ve tahmin yapar.

v0.5 yaklaşımı:
    - data/importance_feedback.jsonl dosyasından eğitim verisi okunur
    - TF-IDF + Logistic Regression modeli eğitilir
    - Model, mesaj için should_store olasılığı üretir
    - Rule-based skor ile ML skoru hibrit şekilde birleştirilebilir
"""

import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
except ModuleNotFoundError:
    joblib = None
    TfidfVectorizer = None
    LogisticRegression = None
    classification_report = None
    train_test_split = None
    Pipeline = Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from local_rag.importance import (
    STORE_THRESHOLD,
    _clamp_score,
    _score_to_label,
    explain_importance,
    get_memory_action,
    should_ask_feedback,
)


DEFAULT_FEEDBACK_PATH = Path("data/importance_feedback.jsonl")
DEFAULT_MODEL_PATH = Path("models/importance_model.joblib")
DEFAULT_TEST_RATIO = 0.25
RANDOM_SEED = 42

RULE_SCORE_WEIGHT = 0.40
MODEL_SCORE_WEIGHT = 0.60


@dataclass(frozen=True)
class TrainingExample:
    """Model eğitimi için tek bir örnek."""

    message: str
    label: int


@dataclass(frozen=True)
class ImportanceExample:
    """Standart kütüphane fallback modeli için etiketli örnek."""

    message: str
    should_store: bool


@dataclass(frozen=True)
class EvaluationReport:
    """Model değerlendirme metriklerini tutar."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int


@dataclass(frozen=True)
class HybridImportanceDecision:
    """Rule-based skor ve ML skorunun birleşmiş kararı."""

    message: str
    rule_score: float
    model_score: float
    final_score: float
    label: str
    category: str | None
    should_store: bool
    should_ask_feedback: bool
    action: str
    rule_reasons: list[str]


def load_training_examples(
    path: Path = DEFAULT_FEEDBACK_PATH,
) -> list[TrainingExample]:
    """JSONL feedback dosyasından eğitim örneklerini okur."""

    if not path.exists():
        raise FileNotFoundError(
            f"Feedback dosyası bulunamadı: {path}. "
            "Önce seed veya kullanıcı feedback verisi oluşturmalısın."
        )

    examples: list[TrainingExample] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)

            message = record.get("message")
            final_should_store = record.get("final_should_store")

            if message is None or final_should_store is None:
                continue

            examples.append(
                TrainingExample(
                    message=message,
                    label=1 if final_should_store else 0,
                )
            )

    if not examples:
        raise ValueError("Eğitim için geçerli feedback kaydı bulunamadı.")

    return examples


def train_importance_model(
    feedback_path: Path = DEFAULT_FEEDBACK_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> Pipeline:
    """TF-IDF + Logistic Regression modelini eğitir ve diske kaydeder."""

    if not _sklearn_available():
        raise ImportError(
            "TF-IDF + Logistic Regression eğitimi için scikit-learn ve joblib gerekir. "
            "Bağımlılık yoksa standart kütüphane baseline için train_model() kullan."
        )

    examples = load_training_examples(feedback_path)

    texts = [example.message for example in examples]
    labels = [example.label for example in examples]

    if len(set(labels)) < 2:
        raise ValueError(
            "Model eğitimi için hem True hem False örnekleri gerekir."
        )

    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    print("Model değerlendirme raporu:")
    print(classification_report(y_test, y_pred, target_names=["ignore", "store"]))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    print(f"Model kaydedildi: {model_path}")

    return model


def load_importance_model(
    model_path: Path = DEFAULT_MODEL_PATH,
) -> Pipeline:
    """Kaydedilmiş importance modelini yükler."""

    if not _sklearn_available():
        raise ImportError("Kaydedilmiş sklearn modelini yüklemek için joblib gerekir.")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model bulunamadı: {model_path}. "
            "Önce scripts/train_importance_model.py çalıştırmalısın."
        )

    return joblib.load(model_path)


def predict_model_score(
    message: str,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> float:
    """ML modelinin should_store olasılığını döndürür."""

    model = load_importance_model(model_path)

    probabilities = model.predict_proba([message])[0]

    return float(probabilities[1])


def combine_scores(
    rule_score: float,
    model_score: float,
    rule_weight: float = RULE_SCORE_WEIGHT,
    model_weight: float = MODEL_SCORE_WEIGHT,
) -> float:
    """Rule-based skor ile ML skorunu birleştirir."""

    if rule_weight < 0 or model_weight < 0:
        raise ValueError("Ağırlıklar negatif olamaz.")

    total_weight = rule_weight + model_weight

    if total_weight == 0:
        raise ValueError("Toplam ağırlık sıfır olamaz.")

    final_score = (
        rule_score * rule_weight
        + model_score * model_weight
    ) / total_weight

    return _clamp_score(final_score)


def explain_importance_hybrid(
    message: str,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> HybridImportanceDecision:
    """Rule-based sistem ve ML modelini birlikte kullanarak karar üretir."""

    rule_decision = explain_importance(message)

    model_score = predict_model_score(
        message=message,
        model_path=model_path,
    )

    final_score = combine_scores(
        rule_score=rule_decision.score,
        model_score=model_score,
    )

    label = _score_to_label(final_score)

    should_store = (
        final_score >= STORE_THRESHOLD
        and rule_decision.category not in {
            "temporary",
            "temporary_state",
            "weather",
        }
    )

    ask_feedback = should_ask_feedback(final_score)

    action = get_memory_action(
        should_store=should_store,
        should_ask=ask_feedback,
    )

    return HybridImportanceDecision(
        message=message,
        rule_score=rule_decision.score,
        model_score=model_score,
        final_score=final_score,
        label=label,
        category=rule_decision.category,
        should_store=should_store,
        should_ask_feedback=ask_feedback,
        action=action,
        rule_reasons=rule_decision.reasons,
    )


def tokenize(text: str) -> list[str]:
    """Türkçe karakterleri koruyarak metni küçük token'lara böler."""

    return re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜ0-9]+", text.lower())


def load_feedback_dataset(path: Path = DEFAULT_FEEDBACK_PATH) -> list[ImportanceExample]:
    """JSONL feedback dataset'ini fallback model formatında okur."""

    return [
        ImportanceExample(
            message=example.message,
            should_store=bool(example.label),
        )
        for example in load_training_examples(path)
    ]


def split_dataset(
    examples: list[ImportanceExample],
    *,
    test_ratio: float = DEFAULT_TEST_RATIO,
    seed: int = RANDOM_SEED,
) -> tuple[list[ImportanceExample], list[ImportanceExample]]:
    """Dataset'i deterministik train/test bölümlerine ayırır."""

    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio 0.0 ile 1.0 arasında olmalı")

    shuffled = examples[:]
    random.Random(seed).shuffle(shuffled)

    test_size = max(1, round(len(shuffled) * test_ratio))

    return shuffled[test_size:], shuffled[:test_size]


class NaiveImportanceClassifier:
    """Bag-of-words Multinomial Naive Bayes önem sınıflandırıcı."""

    def __init__(self, *, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.class_counts: Counter[bool] = Counter()
        self.token_counts: dict[bool, Counter[str]] = {
            True: Counter(),
            False: Counter(),
        }
        self.total_tokens: Counter[bool] = Counter()
        self.vocabulary: set[str] = set()
        self.is_fitted = False

    def fit(self, examples: list[ImportanceExample]) -> None:
        """Modeli etiketli örneklerle eğitir."""

        if not examples:
            raise ValueError("Eğitim için en az bir örnek gerekli")

        self.class_counts.clear()
        self.token_counts = {True: Counter(), False: Counter()}
        self.total_tokens.clear()
        self.vocabulary.clear()

        for example in examples:
            label = example.should_store
            tokens = tokenize(example.message)

            self.class_counts[label] += 1
            self.token_counts[label].update(tokens)
            self.total_tokens[label] += len(tokens)
            self.vocabulary.update(tokens)

        if len(self.class_counts) < 2:
            raise ValueError("Model için hem True hem False etiketli örnek gerekli")

        self.is_fitted = True

    def predict_proba(self, message: str) -> float:
        """Mesajın hafızaya kaydedilme olasılığını döndürür."""

        self._ensure_fitted()

        log_scores = {
            label: self._class_log_probability(label, tokenize(message))
            for label in (True, False)
        }

        max_log_score = max(log_scores.values())
        exp_store = math.exp(log_scores[True] - max_log_score)
        exp_ignore = math.exp(log_scores[False] - max_log_score)

        return exp_store / (exp_store + exp_ignore)

    def predict(self, message: str, *, threshold: float = 0.5) -> bool:
        """Mesaj için final_should_store tahmini yapar."""

        return self.predict_proba(message) >= threshold

    def _class_log_probability(self, label: bool, tokens: list[str]) -> float:
        total_examples = sum(self.class_counts.values())
        class_prior = self.class_counts[label] / total_examples
        score = math.log(class_prior)

        vocabulary_size = max(1, len(self.vocabulary))
        denominator = self.total_tokens[label] + self.alpha * vocabulary_size

        for token in tokens:
            numerator = self.token_counts[label][token] + self.alpha
            score += math.log(numerator / denominator)

        return score

    def _ensure_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Model önce fit() ile eğitilmeli")


def evaluate(
    model: NaiveImportanceClassifier,
    examples: list[ImportanceExample],
) -> EvaluationReport:
    """Modeli test örnekleri üzerinde değerlendirir."""

    if not examples:
        raise ValueError("Değerlendirme için en az bir örnek gerekli")

    confusion: dict[tuple[bool, bool], int] = defaultdict(int)

    for example in examples:
        predicted = model.predict(example.message)
        confusion[(example.should_store, predicted)] += 1

    true_positive = confusion[(True, True)]
    true_negative = confusion[(False, False)]
    false_positive = confusion[(False, True)]
    false_negative = confusion[(True, False)]

    total = len(examples)
    accuracy = (true_positive + true_negative) / total
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    f1 = _safe_divide(2 * precision * recall, precision + recall)

    return EvaluationReport(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        true_positive=true_positive,
        true_negative=true_negative,
        false_positive=false_positive,
        false_negative=false_negative,
    )


def train_model(
    path: Path = DEFAULT_FEEDBACK_PATH,
) -> tuple[NaiveImportanceClassifier, list[ImportanceExample], list[ImportanceExample]]:
    """Dataset'i yükler, böler ve standart kütüphane baseline modelini eğitir."""

    examples = load_feedback_dataset(path)
    train_examples, test_examples = split_dataset(examples)

    model = NaiveImportanceClassifier()
    model.fit(train_examples)

    return model, train_examples, test_examples


def _safe_divide(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _sklearn_available() -> bool:
    return all(
        dependency is not None
        for dependency in (
            joblib,
            TfidfVectorizer,
            LogisticRegression,
            classification_report,
            train_test_split,
        )
    )


def main() -> None:
    """Bağımlılıksız baseline modeli eğitir ve kısa değerlendirme çıktısı basar."""

    model, train_examples, test_examples = train_model()
    report = evaluate(model, test_examples)

    print("ML importance baseline")
    print(f"Dataset: {DEFAULT_FEEDBACK_PATH}")
    print(f"Train örnekleri: {len(train_examples)}")
    print(f"Test örnekleri: {len(test_examples)}")
    print(f"Accuracy: {report.accuracy:.2f}")
    print(f"Precision: {report.precision:.2f}")
    print(f"Recall: {report.recall:.2f}")
    print(f"F1: {report.f1:.2f}")
    print(
        "Confusion matrix: "
        f"TP={report.true_positive}, TN={report.true_negative}, "
        f"FP={report.false_positive}, FN={report.false_negative}"
    )


if __name__ == "__main__":
    main()
