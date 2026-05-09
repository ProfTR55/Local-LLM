"""KVKK uyumlu PII (Personally Identifiable Information) maskeleme.

Sorumluluk:
    Kullanıcı mesajlarındaki kişisel verileri otomatik maskelemek.
    Maskelenmiş metin hafızaya yazılır; orijinal asla diske yazılmaz.

v0 yaklaşımı: Regex tabanlı temel maskeleme
v1 yaklaşımı: spaCy / HuggingFace NER ile isim/yer maskeleme

Maskelenecek PII tipleri:
    - EMAIL    : ali@example.com
    - PHONE    : +90 532 123 45 67  (TR formatları)
    - TCKN     : 12345678901  (11 hane, 0 ile başlamaz)
    - IBAN     : TR + 24 rakam
    - CARD     : 13–19 hane (Luhn ile doğrulanabilir, opsiyonel)

Yapılacaklar:
    1. Her PII tipi için bir re.compile() ile regex hazırla
    2. anonymize(text: str) -> str: PII'leri [MASKED_TIP] ile değiştir
    3. anonymize_with_report(text) -> AnonymizationReport:
        - masked_text: maskelenmiş hali
        - counts: hangi tipten kaç tane bulundu (dict)
    4. Sıra önemli: TCKN ve IBAN'ı kredi kartından ÖNCE kontrol et
       (kart regex'i 13–19 hane, TCKN 11 hane → karışabilir)

İpucu (regex'ler):
    EMAIL: r"\\b[\\w.+-]+@[\\w-]+\\.[\\w.-]+\\b"
    PHONE_TR: r"\\b(?:\\+?90[\\s-]?)?(?:0?5\\d{2})[\\s-]?\\d{3}[\\s-]?\\d{2}[\\s-]?\\d{2}\\b"
    TCKN: r"\\b[1-9]\\d{10}\\b"
    IBAN_TR: r"\\bTR\\d{24}\\b"

Örnek kullanım:
    >>> anonymize("Mailim ali@x.com, telefon 0532 123 45 67")
    'Mailim [MASKED_EMAIL], telefon [MASKED_PHONE]'

Test:
    Yazdıktan sonra `pytest tests/test_smoke.py::test_anonymizer*` ile doğrula
"""

# TODO: PII regex'lerini tanımla
# TODO: anonymize() fonksiyonunu yaz
# TODO: AnonymizationReport dataclass'ı + anonymize_with_report() yaz
