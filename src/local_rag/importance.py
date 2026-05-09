"""Mesajın önem skorunu (0–1) hesaplar.

Bu modül projenin BİLİMSEL KATKISININ kalbidir. Doğru kalibrasyonu için
deney yapman ve etiketli bir test seti oluşturman gerekir.

v0 yaklaşımı (heuristik):
    - Kişisel kimlik / sağlık / kalıcı tercih ifadeleri yüksek puan alır
    - Geçici durumlar (hava, anlık duygu) düşük puan alır
    - Mesaj uzunluğu marjinal katkı yapar

v1 yaklaşımı (ileride):
    - Etiketli veri ile fine-tuned küçük bir BERT classifier
    - Kullanıcı geri bildirimi (👍/👎) skoru günceller

Yapılacaklar:
    1. Yüksek önem regex desenlerini tanımla (HIGH_IMPORTANCE_PATTERNS)
       Örnekler: "adım", "alerji", "okuyorum", "mesleğim", "sevmem"
    2. Düşük önem desenlerini tanımla (LOW_IMPORTANCE_PATTERNS)
       Örnekler: "bugün", "şu an", "hava", "açım"
    3. calculate_importance(message: str) -> float fonksiyonunu yaz:
        - Mesajı küçük harfe çevir
        - Yüksek desenler için her eşleşmede +0.15
        - Düşük desenler için her eşleşmede -0.15
        - Uzunluk katkısı: min(word_count/30, 0.2)
        - Başlangıç değeri: 0.5 (nötr)
        - Sonucu [0, 1] aralığına sıkıştır (clamp)
    4. (Opsiyonel) ImportanceFeatures dataclass'ı — açıklanabilirlik için
       hangi desenlerin eşleştiğini, son skoru ve ara hesapları döndürür

İpucu:
    import re
    re.search(pattern, text)  # eşleşme var mı?
    max(0.0, min(1.0, score))  # clamp

DİKKAT (bilimsel uyarı):
    Buradaki katsayılar (0.15, 0.2, 0.5) keyfi seçimlerdir. Final raporda
    bunları savunabilmek için ablation study yapmalısın.
"""

# TODO: Yüksek önem regex listesini oluştur
# TODO: Düşük önem regex listesini oluştur
# TODO: calculate_importance fonksiyonunu yaz
