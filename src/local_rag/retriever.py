"""Hibrit (benzerlik + önem) skorlu geri çağırma.

Standart RAG: sadece embedding cosine benzerliği kullanılır.
Bu projenin katkısı: önem skoru ile ağırlıklandırılmış hibrit skor.

    final_score = SIM_WEIGHT * similarity + IMP_WEIGHT * importance
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import settings
from .embedder import Embedder, get_embedder
from .memory_store import MemoryStore


@dataclass
class RetrievedMemory:
    """Geri çağrılan tek bir hafıza parçası."""

    id: str
    text: str
    similarity: float
    importance: float
    final_score: float
    timestamp: str | None
    role: str


class Retriever:
    """Embedder + MemoryStore + hibrit skorlama orkestrasyonu."""

    def __init__(
        self,
        memory: MemoryStore | None = None,
        embedder: Embedder | None = None,
        sim_weight: float | None = None,
        imp_weight: float | None = None,
    ) -> None:
        self.memory = memory or MemoryStore()
        self.embedder = embedder or get_embedder()
        self.sim_weight = sim_weight if sim_weight is not None else settings.retrieval_sim_weight
        self.imp_weight = imp_weight if imp_weight is not None else settings.retrieval_imp_weight

        # Ağırlıkların toplamı 1 olmalı (uyarı için)
        total = self.sim_weight + self.imp_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"sim_weight + imp_weight = {total}, beklenen: 1.0"
            )

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        candidate_pool: int = 20,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedMemory]:
        """Sorguya göre hibrit skorlu top-k hafıza parçası döndür.

        Args:
            query: Kullanıcı sorgusu.
            top_k: Döndürülecek sonuç sayısı (varsayılan: settings).
            candidate_pool: Yeniden sıralama için ChromaDB'den çekilecek aday sayısı.
            where: ChromaDB metadata filtresi.

        Returns:
            final_score'a göre azalan sırada hafıza listesi.
        """
        top_k = top_k or settings.retrieval_top_k
        query_vec = self.embedder.encode(query).tolist()

        candidates = self.memory.query(
            embedding=query_vec,
            top_k=candidate_pool,
            where=where,
        )

        scored = [
            RetrievedMemory(
                id=c["id"],
                text=c["text"],
                similarity=c["similarity"],
                importance=c["importance"],
                final_score=self._hybrid_score(c["similarity"], c["importance"]),
                timestamp=c.get("timestamp"),
                role=c.get("role", "user"),
            )
            for c in candidates
        ]

        scored.sort(key=lambda m: m.final_score, reverse=True)
        return scored[:top_k]

    def _hybrid_score(self, similarity: float, importance: float) -> float:
        """final = sim_weight * similarity + imp_weight * importance"""
        return self.sim_weight * similarity + self.imp_weight * importance
