"""Smoke testler.

Bu dosyada issue bazlı modüller tamamlandıkça hızlı doğrulama testleri aktifleşir.
"""

from local_rag.anonymizer import (
    DetectedPII,
    RegexDetector,
    anonymize,
    anonymize_with_report,
    apply_spans,
    is_valid_luhn,
    is_valid_tckn,
    merge_spans,
)


def test_anonymizer_imports_and_masks_email():
    """E-posta maskeleniyor mu?"""

    out = anonymize("Bana mail at: ali@example.com")

    assert "ali@example.com" not in out
    assert "[MASKED_EMAIL]" in out


def test_anonymizer_masks_phone_tr():
    """Türkiye formatında telefon maskeleniyor mu?"""

    out = anonymize("Telefonum 0532 123 45 67, yedek +90 532 123 45 67")

    assert "0532 123 45 67" not in out
    assert "+90 532 123 45 67" not in out
    assert out.count("[MASKED_PHONE]") == 2


def test_anonymizer_report_counts():
    """AnonymizationReport doğru sayım yapıyor mu?"""

    report = anonymize_with_report(
        "Mail ali@example.com, tel 0532 123 45 67, "
        "TCKN 10000000146, IBAN TR330006100519786457841326, "
        "kart 4111 1111 1111 1111"
    )

    assert report.counts == {
        "EMAIL": 1,
        "PHONE": 1,
        "TCKN": 1,
        "IBAN": 1,
        "CARD": 1,
    }
    assert "[MASKED_EMAIL]" in report.masked_text
    assert "[MASKED_PHONE]" in report.masked_text
    assert "[MASKED_TCKN]" in report.masked_text
    assert "[MASKED_IBAN]" in report.masked_text
    assert "[MASKED_CARD]" in report.masked_text


def test_anonymizer_regex_detector_returns_spans():
    spans = RegexDetector().detect("Mailim ali@x.com")

    assert spans == [
        DetectedPII(start=7, end=16, pii_type="EMAIL", source="regex", confidence=1.0)
    ]


def test_anonymizer_merge_resolves_overlap_with_priority():
    email = DetectedPII(start=0, end=15, pii_type="EMAIL", source="regex", confidence=1.0)
    person = DetectedPII(start=0, end=3, pii_type="PERSON", source="ner", confidence=0.85)

    merged = merge_spans([person, email])

    assert merged == [email]


def test_anonymizer_apply_spans_preserves_offsets():
    text = "X ali@example.com Y 0532 123 45 67"
    email_start = text.index("ali@example.com")
    phone_start = text.index("0532")
    spans = [
        DetectedPII(
            start=email_start,
            end=email_start + len("ali@example.com"),
            pii_type="EMAIL",
            source="regex",
            confidence=1.0,
        ),
        DetectedPII(
            start=phone_start,
            end=phone_start + len("0532 123 45 67"),
            pii_type="PHONE",
            source="regex",
            confidence=1.0,
        ),
    ]

    assert apply_spans(text, spans) == "X [MASKED_EMAIL] Y [MASKED_PHONE]"


def test_anonymizer_tckn_validator_rejects_invalid_checksum():
    assert not is_valid_tckn("12345678901")
    assert is_valid_tckn("10000000146")


def test_anonymizer_luhn_validator_rejects_invalid_card():
    assert not is_valid_luhn("1234 5678 9012 3456")
    assert is_valid_luhn("4111 1111 1111 1111")


def test_anonymizer_false_positive_invalid_tckn_lower_confidence():
    spans = RegexDetector().detect("TCKN adayım 12345678901")

    assert spans == [
        DetectedPII(start=12, end=23, pii_type="TCKN", source="regex", confidence=0.4)
    ]
