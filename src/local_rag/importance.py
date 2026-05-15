"""Mesajın önem skorunu (0–1) hesaplar.

Bu modül projenin BİLİMSEL KATKISININ kalbidir. Doğru kalibrasyonu için
deney yapman ve etiketli bir test seti oluşturman gerekir.

v0 yaklaşımı (heuristik):
    - Kişisel kimlik / sağlık / açık hobi-fobi / iletişim-adres bilgileri
      yüksek puan alır
    - Geçici durumlar (hava, anlık duygu) düşük puan alır
    - v0.2 itibarıyla mesaj uzunluğu katkısı kaldırıldı
    - Regex kuralları category + weight + reason yapısına taşındı
    - should_store kararı eklendi

v0.3 yaklaşımı:
    - Kullanıcı geri bildirimi JSONL dataset olarak kaydedilir

v0.4 yaklaşımı:
    - Active learning katmanı eklendi
    - Sistem sadece kararsız kaldığı skor aralığında kullanıcıdan feedback ister

v1 yaklaşımı (ileride):
    - Etiketli veri ile fine-tuned küçük bir BERT classifier
    - Kullanıcı geri bildirimi skoru günceller

DİKKAT (bilimsel uyarı):
    Buradaki katsayılar keyfi seçimlerdir. Final raporda bunları
    savunabilmek için ablation study yapmalısın.
"""

import re
import sys
from dataclasses import dataclass


BASE_SCORE = 0.5
STORE_THRESHOLD = 0.7

ASK_FEEDBACK_MIN_SCORE = 0.45
ASK_FEEDBACK_MAX_SCORE = 0.70


@dataclass(frozen=True)
class PatternRule:
    """Bir regex kuralının kategori, ağırlık ve açıklamasını tutar."""

    pattern: str
    category: str
    weight: float
    reason: str


@dataclass(frozen=True)
class ImportanceDecision:
    """Mesajın önem kararını ve kararın nedenlerini tutar."""

    score: float
    label: str
    category: str | None
    should_store: bool
    should_ask_feedback: bool
    action: str
    matched_rules: list[PatternRule]
    reasons: list[str]


PATTERN_RULES = [
    PatternRule(
        pattern=r"\bad[ıi]m\b|\bismim\b",
        category="identity",
        weight=0.30,
        reason="Kullanıcının adı olabilir.",
    ),
    PatternRule(
        pattern=r"\bsoyad[ıi]m\b",
        category="identity",
        weight=0.30,
        reason="Kullanıcının soyadı olabilir.",
    ),
    PatternRule(
        pattern=r"\balerji(?:m|m var|k)?\b",
        category="health",
        weight=0.40,
        reason="Kullanıcının alerji bilgisi olabilir.",
    ),
    PatternRule(
        pattern=r"\bhastal[ıi]ğ[ıi]m\b",
        category="health",
        weight=0.40,
        reason="Kullanıcının hastalık bilgisi olabilir.",
    ),
    PatternRule(
        pattern=r"\bilac[ıi]m\b|\bilaç kullanıyorum\b",
        category="health",
        weight=0.35,
        reason="Kullanıcının ilaç bilgisi olabilir.",
    ),
    PatternRule(
        pattern=r"\bhobi(?:m|lerim)?\b",
        category="hobby",
        weight=0.30,
        reason="Kullanıcının açık hobi bilgisi olabilir.",
    ),
    PatternRule(
        pattern=r"\bfobi(?:m|lerim)?\b",
        category="phobia",
        weight=0.30,
        reason="Kullanıcının açık fobi bilgisi olabilir.",
    ),
    PatternRule(
        pattern=r"\badresim\b|\b(?:ev|iş) adresim\b",
        category="address",
        weight=0.35,
        reason="Kullanıcının adres bilgisi olabilir.",
    ),
    PatternRule(
        pattern=(
            r"\b(?:telefon(?:um| numaram)|cep numaram)\b|"
            r"\b(?:\+?90[\s.-]*)?(?:0[\s.-]*)?(?:5\d{2}|[2-4]\d{2})"
            r"[\s.-]*\d{3}[\s.-]*\d{2}[\s.-]*\d{2}\b"
        ),
        category="contact",
        weight=0.35,
        reason="Kullanıcının telefon bilgisi olabilir.",
    ),
    PatternRule(
        pattern=(
            r"\b(?:e-?posta(?:m| adresim)?|email(?:im| adresim)?)\b|"
            r"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b"
        ),
        category="email",
        weight=0.35,
        reason="Kullanıcının e-posta bilgisi olabilir.",
    ),
    PatternRule(
        pattern=r"\bdoğum (?:tarihim|günüm)\b",
        category="birth_date",
        weight=0.35,
        reason="Kullanıcının doğum tarihi bilgisi olabilir.",
    ),
    PatternRule(
        pattern=r"\b(?:yaşadığım şehir|yaşadığım yer|memleketim|ikamet ettiğim şehir)\b",
        category="location",
        weight=0.30,
        reason="Kullanıcının açık konum/şehir bilgisi olabilir.",
    ),
    PatternRule(
        pattern=r"\bbug[üu]n\b|\bşu an\b|\bşimdi\b|\bgeçici\b",
        category="temporary",
        weight=-0.30,
        reason="Geçici zaman bilgisi olabilir.",
    ),
    PatternRule(
        pattern=r"\bhava\b",
        category="weather",
        weight=-0.25,
        reason="Hava durumu genellikle kalıcı hafıza için önemli değildir.",
    ),
    PatternRule(
        pattern=r"\baç[ıi]m\b|\bsusad[ıi]m\b|\byorgunum\b|\bcan[ıi]m s[ıi]kk[ıi]n\b",
        category="temporary_state",
        weight=-0.35,
        reason="Geçici kullanıcı durumu olabilir.",
    ),
]


def _clamp_score(score: float) -> float:
    """Skoru 0.0 ile 1.0 arasına sıkıştırır."""

    return max(0.0, min(1.0, score))


def _score_to_label(score: float) -> str:
    """Sayısal skoru low, medium veya high etiketine çevirir."""

    if score >= 0.7:
        return "high"

    if score >= 0.4:
        return "medium"

    return "low"


def _get_primary_category(matched_rules: list[PatternRule]) -> str | None:
    """Pozitif ağırlıklı kurallar arasından en baskın kategoriyi seçer."""

    positive_rules = [rule for rule in matched_rules if rule.weight > 0]

    if not positive_rules:
        return None

    strongest_rule = max(positive_rules, key=lambda rule: rule.weight)

    return strongest_rule.category


def _should_store(score: float, category: str | None) -> bool:
    """Mesajın hafızaya kaydedilip kaydedilmeyeceğine karar verir."""

    if category is None:
        return False

    if category in {"temporary", "temporary_state", "weather"}:
        return False

    return score >= STORE_THRESHOLD


def should_ask_feedback(score: float) -> bool:
    """Model kararsızsa kullanıcıdan feedback istenip istenmeyeceğini döndürür.

    Active learning mantığı:
        - Çok düşük skor: sorma, kaydetme
        - Orta skor: kullanıcıya sor
        - Yüksek skor: hafızaya aday
    """

    return ASK_FEEDBACK_MIN_SCORE <= score < ASK_FEEDBACK_MAX_SCORE


def get_memory_action(should_store: bool, should_ask: bool) -> str:
    """Karara göre sistemin ne yapması gerektiğini döndürür."""

    if should_ask:
        return "ask_feedback"

    if should_store:
        return "store"

    return "ignore"


def explain_importance(message: str) -> ImportanceDecision:
    """Mesajın önem skorunu, kategorisini ve kayıt kararını döndürür."""

    text = message.strip().lower()

    if not text:
        return ImportanceDecision(
            score=0.0,
            label="low",
            category=None,
            should_store=False,
            should_ask_feedback=False,
            action="ignore",
            matched_rules=[],
            reasons=["Boş mesaj önem bilgisi içermez."],
        )

    matched_rules = [
        rule
        for rule in PATTERN_RULES
        if re.search(rule.pattern, text)
    ]

    score = BASE_SCORE + sum(rule.weight for rule in matched_rules)
    score = _clamp_score(score)

    label = _score_to_label(score)
    category = _get_primary_category(matched_rules)
    should_store = _should_store(score, category)
    ask_feedback = should_ask_feedback(score)
    action = get_memory_action(
        should_store=should_store,
        should_ask=ask_feedback,
    )

    return ImportanceDecision(
        score=score,
        label=label,
        category=category,
        should_store=should_store,
        should_ask_feedback=ask_feedback,
        action=action,
        matched_rules=matched_rules,
        reasons=[rule.reason for rule in matched_rules],
    )


def calculate_importance(message: str) -> float:
    """Mesajın önem skorunu 0.0 ile 1.0 arasında hesaplar."""

    return explain_importance(message).score


def main(argv: list[str] | None = None) -> None:
    """Dosya doğrudan çalıştırıldığında örnek önem kararlarını yazdırır."""

    args = sys.argv[1:] if argv is None else argv
    messages = [" ".join(args)] if args else [
        "Benim adım Doğukan",
        "Fıstığa alerjim var",
        "Bugün hava çok güzel",
        "Hobim satranç",
        "Telefon numaram 0532 123 45 67",
        "E-postam dogukan@example.com",
    ]

    for message in messages:
        decision = explain_importance(message)
        print(f"Mesaj: {message}")
        print(f"Skor: {decision.score:.2f}")
        print(f"Etiket: {decision.label}")
        print(f"Kategori: {decision.category}")
        print(f"Hafızaya kaydet: {decision.should_store}")
        print(f"Feedback sor: {decision.should_ask_feedback}")
        print(f"Aksiyon: {decision.action}")
        print()


if __name__ == "__main__":
    main()
