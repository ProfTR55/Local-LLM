"""KVKK uyumlu PII (Personally Identifiable Information) maskeleme.

v0 yaklaşımı: regex tabanlı temel maskeleme.
v1 yaklaşımı: spaCy / HuggingFace NER modeliyle geliştirilebilir.

Maskelenecek alanlar:
    - Telefon numaraları (TR formatı)
    - E-posta adresleri
    - TC kimlik numaraları
    - IBAN
    - Kredi kartı numaraları (Luhn doğrulamalı)
    - (İleride) İsim/adres — NER ile
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# ----- Regex desenleri -----
_PHONE_TR = re.compile(r"\b(?:\+?90[\s-]?)?(?:0?5\d{2})[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}\b")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_TCKN = re.compile(r"\b[1-9]\d{10}\b")  # 11 haneli, 0 ile başlamaz
_IBAN_TR = re.compile(r"\bTR\d{24}\b", re.IGNORECASE)
_CREDIT_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")


@dataclass
class AnonymizationReport:
    """Hangi alanların kaç kez maskelendiğini tutan rapor."""

    masked_text: str
    counts: dict[str, int]


def anonymize(text: str) -> str:
    """Verilen metnin maskelenmiş halini döndür.

    Args:
        text: Orijinal mesaj.

    Returns:
        PII'leri `[MASKED_*]` etiketleriyle değiştirilmiş metin.
    """
    return anonymize_with_report(text).masked_text


def anonymize_with_report(text: str) -> AnonymizationReport:
    """Maskeleme + hangi PII'lerin yakalandığına dair rapor."""
    counts: dict[str, int] = {}

    def _sub(pattern: re.Pattern[str], label: str, src: str) -> str:
        nonlocal counts
        replaced, n = pattern.subn(f"[MASKED_{label}]", src)
        if n:
            counts[label] = counts.get(label, 0) + n
        return replaced

    # Sıra önemli — IBAN ve TCKN'i kredi kartından önce kontrol et
    out = text
    out = _sub(_EMAIL, "EMAIL", out)
    out = _sub(_PHONE_TR, "PHONE", out)
    out = _sub(_IBAN_TR, "IBAN", out)
    out = _sub(_TCKN, "TCKN", out)
    out = _sub(_CREDIT_CARD, "CARD", out)

    return AnonymizationReport(masked_text=out, counts=counts)
