"""Sentence-BERT ile metin → vektör dönüşümü.

Sorumluluk:
    - Verilen cümleyi sabit boyutlu bir embedding vektörüne çevirmek.
    - Tek tek veya batch olarak kodlama yapmak.
    - Modeli bir kez yükleyip belleğe almak.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

from .config import settings


class Embedder:
    """Sentence-BERT tabanlı embedder.

    Example:
        >>> emb = Embedder()
        >>> vec = emb.encode("Merhaba dünya")
        >>> vec.shape
        (384,)
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.embedding_model
        self._model = None  # lazy load

    def _load(self) -> None:
        """Modeli ihtiyaç anında yükle (ilk encode çağrısında)."""
        if self._model is None:
            # NOTE: import burada — paket yüklemesi pahalı
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)

    def encode(self, text: str | Sequence[str]):
        """Tek cümle veya cümle listesini embedding'e çevir.

        Args:
            text: Tek string veya string listesi.

        Returns:
            np.ndarray — tek girişte (dim,), liste girişte (n, dim).
        """
        self._load()
        assert self._model is not None
        return self._model.encode(text, convert_to_numpy=True, show_progress_bar=False)

    @property
    def dimension(self) -> int:
        """Embedding boyutu (örn. MiniLM için 384)."""
        self._load()
        assert self._model is not None
        return self._model.get_sentence_embedding_dimension()


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Süreç boyunca tek bir Embedder örneği paylaş."""
    return Embedder()
