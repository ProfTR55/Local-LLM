"""ChromaDB tabanlı yerel hafıza CRUD katmanı.

Her hafıza kaydı şunları içerir:
    - id              : benzersiz string (uuid)
    - embedding       : vektör (Embedder tarafından üretilir)
    - metadata        : { text, importance, timestamp, role, ... }
    - document        : ham metin (ChromaDB'nin documents alanı)

ChromaDB native olarak metadata filtreleme destekler:
    where={"importance": {"$gte": 0.5}}

Yapılacaklar:
    1. MemoryStore sınıfını oluştur
    2. __init__: persist_dir ve collection_name parametreleri al
       (default: settings'den)
    3. _ensure_collection(): chromadb.PersistentClient ve collection'ı
       lazy olarak oluştur. Cosine mesafesi için:
       collection.create(metadata={"hnsw:space": "cosine"})
    4. add(text, embedding, importance, role, extra_metadata) -> str (id)
       - uuid4 ile id üret
       - timestamp'i UTC ISO formatında ekle
       - collection.add(ids=, embeddings=, metadatas=, documents=)
       - Üretilen id'yi döndür
    5. query(embedding, top_k, where) -> list[dict]
       - collection.query(query_embeddings=, n_results=, where=)
       - Sonucu normalize et: liste içinde dict'ler
       - similarity = 1.0 - distance hesapla
    6. delete(ids: list[str])
    7. delete_low_importance(threshold, older_than_days) -> int
       Düşük önem + TTL'i geçmiş kayıtları sil. SİLİNEN sayıyı döndür.
    8. count() -> int

İpucu:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    client = chromadb.PersistentClient(
        path="./chroma_db",
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(name="personal_memory")

ÖNEMLİ:
    chromadb ilk kullanımda 200+ MB indirme yapabilir.
    pip install chromadb komutunu önceden çalıştır.
"""

# TODO: MemoryStore sınıfını yaz
# TODO: add() metodu
# TODO: query() metodu (sonucu normalize et)
# TODO: delete() metodu
# TODO: delete_low_importance() metodu (TTL temizleme)
# TODO: count() metodu
