"""KVKK uyumlu PII (Personally Identifiable Information) maskeleme.

Bu modül anonimleştirme mimarisinin ilk katmanıdır. Regex detector doğrudan
string replace yapmaz; bulunan PII adaylarını span olarak döndürür. Maskeleme
tek bir merkezden, `merge_spans` ve `apply_spans` üzerinden yapılır. Sonraki
TR-NER ve yerel LLM verifier katmanları da aynı span yapısını kullanabilir.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable


SOURCE_PRIORITY = {"regex": 3, "ner": 2, "llm": 1}


@dataclass(frozen=True)
class DetectedPII:
    """Metin içindeki tek bir PII span'ini temsil eder."""

    start: int
    end: int
    pii_type: str
    source: str
    confidence: float


@dataclass(frozen=True)
class AnonymizationReport:
    """Anonimleştirme çıktısı ve açıklanabilir tespit raporu."""

    masked_text: str
    counts: dict[str, int]
    spans: list[DetectedPII]


@dataclass(frozen=True)
class RegexRule:
    """Regex detector için tek bir PII kuralı."""

    pii_type: str
    pattern: re.Pattern[str]
    validator: Callable[[str], bool] | None = None


def is_valid_tckn(value: str) -> bool:
    """T.C. kimlik numarasını checksum kurallarıyla doğrular."""

    if len(value) != 11 or not value.isdigit() or value[0] == "0":
        return False

    digits = [int(char) for char in value]
    odd_sum = sum(digits[0:9:2])
    even_sum = sum(digits[1:8:2])

    rule1 = ((odd_sum * 7) - even_sum) % 10 == digits[9]
    rule2 = sum(digits[0:10]) % 10 == digits[10]

    return rule1 and rule2


def is_valid_luhn(value: str) -> bool:
    """Kredi kartı adayını Luhn algoritmasıyla doğrular."""

    digits = [int(char) for char in value if char.isdigit()]

    if not 13 <= len(digits) <= 19:
        return False

    checksum = 0
    parity = len(digits) % 2

    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9

        checksum += digit

    return checksum % 10 == 0


class RegexDetector:
    """Regex tabanlı PII detector.

    Her kural `DetectedPII` döndürür. Validator kullanan kurallarda geçerli
    adaylar 1.0, checksum'dan geçmeyen adaylar 0.4 confidence alır.
    """

    def __init__(self, rules: list[RegexRule] | None = None) -> None:
        self.rules = rules or _DEFAULT_REGEX_RULES

    def detect(self, text: str) -> list[DetectedPII]:
        """Metindeki PII adaylarını span listesi olarak döndürür."""

        spans: list[DetectedPII] = []

        for rule in self.rules:
            for match in rule.pattern.finditer(text):
                matched_text = match.group(0)
                confidence = 1.0

                if rule.validator is not None and not rule.validator(matched_text):
                    confidence = 0.4

                spans.append(
                    DetectedPII(
                        start=match.start(),
                        end=match.end(),
                        pii_type=rule.pii_type,
                        source="regex",
                        confidence=confidence,
                    )
                )

        return sorted(spans, key=lambda span: (span.start, span.end))


def merge_spans(
    spans: list[DetectedPII],
    policy: str = "priority_then_longest",
) -> list[DetectedPII]:
    """Overlap eden span'leri deterministik şekilde birleştirir.

    `priority_then_longest` politikası:
        1. Kaynak önceliği yüksek olan kazanır: regex > ner > llm.
        2. Aynı kaynak önceliğinde daha uzun span kazanır.
        3. Sonuç metindeki başlangıç sırasına göre döndürülür.
    """

    if policy != "priority_then_longest":
        raise ValueError(f"Desteklenmeyen merge politikası: {policy}")

    selected: list[DetectedPII] = []
    candidates = sorted(
        spans,
        key=lambda span: (
            -SOURCE_PRIORITY.get(span.source, 0),
            -(span.end - span.start),
            -span.confidence,
            span.start,
            span.end,
        ),
    )

    for candidate in candidates:
        if any(_spans_overlap(candidate, existing) for existing in selected):
            continue

        selected.append(candidate)

    return sorted(selected, key=lambda span: (span.start, span.end))


def apply_spans(text: str, spans: list[DetectedPII]) -> str:
    """Span'leri sondan başa doğru `[MASKED_<TYPE>]` ile değiştirir."""

    masked_text = text

    for span in sorted(spans, key=lambda item: item.start, reverse=True):
        _validate_span(text, span)
        replacement = f"[MASKED_{span.pii_type}]"
        masked_text = masked_text[: span.start] + replacement + masked_text[span.end :]

    return masked_text


def anonymize_with_report(text: str) -> AnonymizationReport:
    """Metni anonimleştirir ve maskeleme raporunu döndürür."""

    detector = RegexDetector()
    spans = merge_spans(detector.detect(text))
    masked_text = apply_spans(text, spans)
    counts = dict(Counter(span.pii_type for span in spans))

    return AnonymizationReport(masked_text=masked_text, counts=counts, spans=spans)


def anonymize(text: str) -> str:
    """Metindeki PII değerlerini maskeleyip sadece metni döndürür."""

    return anonymize_with_report(text).masked_text


def _spans_overlap(left: DetectedPII, right: DetectedPII) -> bool:
    return left.start < right.end and right.start < left.end


def _validate_span(text: str, span: DetectedPII) -> None:
    if span.start < 0 or span.end > len(text) or span.start >= span.end:
        raise ValueError(f"Geçersiz span: {span}")


_DEFAULT_REGEX_RULES = [
    RegexRule("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    RegexRule(
        "PHONE",
        re.compile(r"(?<!\w)(?:\+?90[\s-]?)?(?:0?5\d{2})[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}(?!\w)"),
    ),
    RegexRule("IBAN", re.compile(r"\bTR(?:\s?\d){24}\b", re.IGNORECASE)),
    RegexRule("TCKN", re.compile(r"\b[1-9]\d{10}\b"), is_valid_tckn),
    RegexRule("CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b"), is_valid_luhn),
]
