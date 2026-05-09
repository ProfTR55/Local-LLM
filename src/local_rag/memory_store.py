"""ChromaDB tabanlı yerel hafıza CRUD katmanı.

Her hafıza kaydı:
    - id              : benzersiz string
    - embedding       : vektör (Embedder tarafından üretilir)
    - metadata        : { text, importance, timestamp, role, ... }

ChromaDB native olarak metadata filtreleme destekler:
    where={"importance": {"$gte": 0.5}}
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from .config import settings


class MemoryStore:
    """ChromaDB üzerine ince bir sarmalayıcı."""

    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        self.persist_dir = str(persist_dir or settings.chroma_persist_dir)
        self.collection_name = collection_name or settings.chroma_collection_name
        self._client = None
        self._collection = None

    def _ensure_collection(self) -> None:
        if self._collection is not None:
            return
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self._client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ----- Yazma -----

    def add(
        self,
        text: str,
        embedding: list[float],
        importance: float,
        role: str = "user",
        extra_metadata: dict[str, Any] | None = None,
    ) -> str:
        """Tek bir hafıza kaydı ekle ve oluşturulan id'yi döndür."""
        self._ensure_collection()
        assert self._collection is not None

        record_id = str(uuid.uuid4())
        metadata: dict[str, Any] = {
            "text": text,
            "importance": float(importance),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "role": role,
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        self._collection.add(
            ids=[record_id],
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[text],
        )
        return record_id

    # ----- Sorgu -----

    def query(
        self,
        embedding: list[float],
        top_k: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """En benzer top_k kaydı döndür (ham ChromaDB cevabı normalize edilmiş)."""
        self._ensure_collection()
        assert self._collection is not None

        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=where,
        )

        # Normalize: list[dict] formatına çevir
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]

        out: list[dict[str, Any]] = []
        for i, _id in enumerate(ids):
            md = metadatas[i] if i < len(metadatas) else {}
            distance = distances[i] if i < len(distances) else 1.0
            similarity = 1.0 - float(distance)  # cosine: distance ∈ [0,2], similarity ∈ [-1,1]
            out.append(
                {
                    "id": _id,
                    "text": md.get("text", ""),
                    "importance": float(md.get("importance", 0.5)),
                    "timestamp": md.get("timestamp"),
                    "role": md.get("role", "user"),
                    "similarity": similarity,
                    "distance": distance,
                }
            )
        return out

    # ----- Silme -----

    def delete(self, ids: list[str]) -> None:
        self._ensure_collection()
        assert self._collection is not None
        self._collection.delete(ids=ids)

    def delete_low_importance(self, threshold: float, older_than_days: int) -> int:
        """Belirli eşik altındaki ve TTL'i geçmiş kayıtları sil. Silinen sayıyı döndür.

        TODO: ChromaDB metadata filter ile id'leri çek, sonra delete çağır.
        """
        raise NotImplementedError("TTL temizleme henüz implement edilmedi.")

    # ----- Yardımcı -----

    def count(self) -> int:
        self._ensure_collection()
        assert self._collection is not None
        return self._collection.count()
