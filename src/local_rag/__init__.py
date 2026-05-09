"""Local RAG — kişisel hafıza destekli yerel chatbot.

Modules:
    embedder       : Sentence-BERT ile metin → vektör dönüşümü.
    importance     : Mesajın önem skorunu (0–1) hesaplar.
    anonymizer     : KVKK uyumlu PII maskeleme.
    memory_store   : ChromaDB tabanlı yerel hafıza CRUD.
    retriever      : Hibrit (benzerlik + önem) skorlu geri çağırma.
    llm_client     : Ollama ile yerel LLM bağlantısı.
    rag_pipeline   : Uçtan uca RAG akışı.
    cli            : Terminal arayüzü.
"""

__version__ = "0.1.0"
__author__ = "Muhammet Doğukan Bingöl, Beyza Sude Aydın"
