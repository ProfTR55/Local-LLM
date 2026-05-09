"""Mesajın önem skorunu (0–1) hesaplar.

Heuristik yaklaşım (v0):
    - Kişisel kimlik / sağlık / tercih ifadeleri yüksek puan alır.
    - Geçici durumlar (hava, anlık duygu) düşük puan alır.
    - Mesaj uzunluğu marjinal katkı yapar.

İleride:
    - Etiketli veri ile fine-tuned bir küçük sınıflandırıcı modeli ile değiştirilebilir.
    - Kullanıcı geri bildirimi (👍/👎) skoru günceller.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# Yüksek önem işaretleyiciler — kişisel kimlik, sağlık, kalıcı tercihler
_HIGH_IMPORTANCE_PATTERNS = [
    r"\badım\b", r"\bismim\b", r"\byaşım\b", r"\bdoğum\s*tarihim\b",
    r"\balerji\b", r"\bhastalı(k|ğım)\b", r"\bilac\b", r"\bdiyet\b",
    r"\bçalışıyorum\b", r"\bokuyorum\b", r"\bmesleğim\b",
    r"\bsevmiyorum\b", r"\bsevmem\b", r"\btercih\s*etmem\b",
]

# Düşük önem işaretleyiciler — geçici durumlar
_LOW_IMPORTANCE_PATTERNS = [
    r"\bbugün\b", r"\bşu\s*an\b", r"\bşimdi\b", r"\bhava\b",
    r"\baçım\b", r"\btokum\b", r"\bsıkıldım\b",
]


@dataclass
class ImportanceFeatures:
    """Açıklanabilirlik için kullanılan ara özellikler."""

    matched_high: list[str]
    matched_low: list[str]
    length_score: float
    final: float


def calculate_importance(message: str) -> float:
    """Bir mesaj için 0–1 arası önem skoru döndür.

    Args:
        message: Kullanıcı mesajı (Türkçe).

    Returns:
        0.0 (önemsiz) – 1.0 (kritik) arası float.
    """
    return _calculate_features(message).final


def _calculate_features(message: str) -> ImportanceFeatures:
    """Skor + ara özellikler — testlerde ve loglarda kullanmak için."""
    text = message.lower().strip()

    matched_high = [p for p in _HIGH_IMPORTANCE_PATTERNS if re.search(p, text)]
    matched_low = [p for p in _LOW_IMPORTANCE_PATTERNS if re.search(p, text)]

    score = 0.5  # nötr başlangıç
    score += 0.15 * len(matched_high)
    score -= 0.15 * len(matched_low)

    # Uzunluk: çok kısa mesajlar (< 5 kelime) az değerli
    word_count = len(text.split())
    length_score = min(word_count / 30, 0.2)  # max 0.2 katkı
    score += length_score

    final = max(0.0, min(1.0, score))

    return ImportanceFeatures(
        matched_high=matched_high,
        matched_low=matched_low,
        length_score=length_score,
        final=final,
    )
