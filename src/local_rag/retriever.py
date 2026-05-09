"""Hibrit (benzerlik + önem) skorlu geri çağırma.

Bu modül projenin BİLİMSEL KATKISIDIR.
Standart RAG sadece embedding cosine benzerliğini kullanır.
Bizim katkımız: önem skoru ile ağırlıklandırılmış hibrit skor.

    final_score = SIM_WEIGHT * similarity + IMP_WEIGHT * importance

Default ağırlıklar: 0.7 / 0.3 (KEYFİ — final raporda ablation ile savun!)

Yapılacaklar:
    1. RetrievedMemory dataclass'ı tanımla:
        - id, text, similarity, importance, final_score, timestamp, role
    2. Retriever sınıfını oluştur
    3. __init__: memory, embedder, sim_weight, imp_weight parametreleri
       Default'ları settings'ten oku.
       sim_weight + imp_weight ≈ 1.0 olmalı, değilse ValueError fırlat
    4. retrieve(query, top_k, candidate_pool, where) -> list[RetrievedMemory]
       - Sorguyu embed et
       - memory.query(top_k=candidate_pool) ile aday havuzunu çek
         (candidate_pool > top_k olmalı; örn. 20 aday içinden top 5)
       - Her aday için _hybrid_score() hesapla
       - final_score'a göre azalan sıraya diz
       - İlk top_k tanesini döndür
    5. _hybrid_score(similarity, importance) -> float
       sim_weight * similarity + imp_weight * importance

İpucu:
    candidates = self.memory.query(query_vec, top_k=20)
    scored = [
        RetrievedMemory(
            ...,
            final_score=self._hybrid_score(c["similarity"], c["importance"])
        )
        for c in candidates
    ]
    scored.sort(key=lambda m: m.final_score, reverse=True)
    return scored[:top_k]

DENEY FIRSATLARI (raporda kullanılacak):
    - Farklı (sim_w, imp_w) kombinasyonlarını test et
    - Sadece similarity (saf RAG) vs hibrit karşılaştır
    - Top-k değerinin etkisi
"""

# TODO: RetrievedMemory dataclass'ı
# TODO: Retriever sınıfı + __init__ + ağırlık doğrulama
# TODO: retrieve() metodu
# TODO: _hybrid_score() metodu
