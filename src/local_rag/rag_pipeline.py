"""Uçtan uca RAG akışı.

Kullanım:
    >>> pipe = RAGPipeline()
    >>> pipe.ingest("Adım Doğukan, KTÜ bilgisayar müh. okuyorum")
    >>> pipe.ask("Bana proje fikri verir misin?")
    'Bilgisayar mühendisliği öğrencisi olarak ...'
"""

from __future__ import annotations

from dataclasses import dataclass

from .anonymizer import anonymize_with_report
from .embedder import Embedder, get_embedder
from .importance import calculate_importance
from .llm_client import LLMClient
from .memory_store import MemoryStore
from .retriever import Retriever, RetrievedMemory


_SYSTEM_PROMPT = """Sen, kullanıcıyı kişisel olarak tanıyan bir asistansın.
Aşağıda kullanıcı hakkında geçmiş konuşmalardan elde edilmiş bilgiler var.
Cevap üretirken bu bilgileri uygun yerlerde, doğal bir şekilde kullan.
Bilmediğin bir şey hakkında uydurma; emin değilsen kullanıcıya sor.
"""


@dataclass
class PipelineResult:
    answer: str
    used_memories: list[RetrievedMemory]
    masked_query: str


class RAGPipeline:
    """Uçtan uca: anonimleştir → kaydet/sorgula → LLM."""

    def __init__(
        self,
        memory: MemoryStore | None = None,
        embedder: Embedder | None = None,
        retriever: Retriever | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        self.memory = memory or MemoryStore()
        self.embedder = embedder or get_embedder()
        self.retriever = retriever or Retriever(memory=self.memory, embedder=self.embedder)
        self.llm = llm or LLMClient()

    # ----- Ingest -----

    def ingest(self, text: str, role: str = "user") -> str:
        """Bir mesajı anonimleştir, embed et, hafızaya kaydet. Kayıt id'sini döndür."""
        report = anonymize_with_report(text)
        importance = calculate_importance(report.masked_text)
        embedding = self.embedder.encode(report.masked_text).tolist()

        return self.memory.add(
            text=report.masked_text,
            embedding=embedding,
            importance=importance,
            role=role,
            extra_metadata={"pii_counts": str(report.counts)} if report.counts else None,
        )

    # ----- Sorgu -----

    def ask(self, query: str) -> PipelineResult:
        """Soruyu cevapla — retrieval + LLM."""
        masked_query = anonymize_with_report(query).masked_text
        memories = self.retriever.retrieve(masked_query)
        context = self._build_context(memories)
        prompt = f"{context}\n\nKullanıcı sorusu: {masked_query}"

        answer = self.llm.generate(prompt=prompt, system=_SYSTEM_PROMPT)

        # Kullanıcı sorusunu da hafızaya ekle (kendi geçmişi olarak)
        self.ingest(masked_query, role="user")

        return PipelineResult(
            answer=answer,
            used_memories=memories,
            masked_query=masked_query,
        )

    @staticmethod
    def _build_context(memories: list[RetrievedMemory]) -> str:
        if not memories:
            return "Kullanıcı hakkında bilinen bilgi: (henüz hafızada kayıt yok)"

        lines = ["Kullanıcı hakkında bilinen bilgiler:"]
        for i, m in enumerate(memories, 1):
            lines.append(f"  {i}. {m.text}  (önem: {m.importance:.2f})")
        return "\n".join(lines)
