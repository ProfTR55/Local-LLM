"""Sentence-BERT ile metin → vektör dönüşümü.

Sorumluluk:
    - Verilen cümleyi sabit boyutlu bir embedding vektörüne çevirmek.
    - Tek tek veya batch (liste) olarak kodlama yapmak.
    - Modeli bir kez yükleyip belleğe almak (lazy load).

Yapılacaklar:
    1. Embedder adında bir sınıf oluştur
    2. __init__: model_name parametresi al (default: settings.embedding_model)
       Model yüklemeyi __init__'te DEĞİL, ilk encode çağrısında yap (lazy)
    3. encode(text): tek string veya liste alıp np.ndarray döndür
    4. dimension property: embedding boyutunu döndür (örn. MiniLM için 384)
    5. get_embedder() yardımcı fonksiyonu — süreç boyunca tek örnek paylaş
       (functools.lru_cache(maxsize=1) kullan)

İpucu:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    vec = model.encode("metin", convert_to_numpy=True)

Örnek kullanım:
    >>> emb = Embedder()
    >>> vec = emb.encode("Merhaba dünya")
    >>> vec.shape
    (384,)
"""

# TODO: Embedder sınıfını yaz
# TODO: get_embedder() singleton fonksiyonunu yaz
