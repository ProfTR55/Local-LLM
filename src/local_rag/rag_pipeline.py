"""Uçtan uca RAG akışı.

Bu modül, diğer tüm modülleri (anonymizer, embedder, importance,
memory_store, retriever, llm_client) birleştirir.

İki ana akış:

YAZMA (ingest):
    text → anonymize → importance → embed → memory.add

OKUMA (ask):
    query → anonymize → embed → retriever.retrieve → context oluştur →
    LLM prompt → cevap → (sorguyu da hafızaya ekle)

Yapılacaklar:
    1. _SYSTEM_PROMPT sabiti tanımla — LLM'e nasıl davranacağını söyleyen
       sistem mesajı (Türkçe, kişisel asistan tonunda)
    2. PipelineResult dataclass'ı:
        - answer: str
        - used_memories: list[RetrievedMemory]
        - masked_query: str
    3. RAGPipeline sınıfı, __init__:
        - memory, embedder, retriever, llm parametreleri (default: yeni örnekler)
    4. ingest(text, role="user") -> str (kayıt id'si)
        - Anonimleştir
        - Önem hesapla
        - Embed et
        - memory.add ile kaydet
    5. ask(query) -> PipelineResult
        - Sorguyu anonimleştir
        - retriever.retrieve(masked_query) ile anıları çek
        - _build_context() ile prompt için bağlam metni oluştur
        - llm.generate(prompt, system=_SYSTEM_PROMPT) ile cevap al
        - Sorguyu da hafızaya ingest et (kullanıcının kendi tarihçesi)
        - PipelineResult döndür
    6. _build_context(memories) -> str
        - Anılar boşsa: "kullanıcı hakkında bilgi yok"
        - Doluysa: numaralı liste, her satırda "metin (önem: 0.X)"

Örnek:
    >>> pipe = RAGPipeline()
    >>> pipe.ingest("Adım Doğukan, KTÜ bilgisayar müh. okuyorum")
    >>> result = pipe.ask("Bana proje fikri ver")
    >>> print(result.answer)
"""

# TODO: _SYSTEM_PROMPT sabiti
# TODO: PipelineResult dataclass'ı
# TODO: RAGPipeline sınıfı
# TODO: ingest() metodu
# TODO: ask() metodu
# TODO: _build_context() yardımcı fonksiyonu
