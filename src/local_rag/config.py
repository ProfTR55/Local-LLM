"""Tüm modüller için merkezi konfigürasyon.

`.env` dosyasından (varsa) ortam değişkenlerini okur ve tip-güvenli
bir Settings nesnesi olarak sunar.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Repo kökü
ROOT_DIR = Path(__file__).resolve().parents[2]

# .env dosyasını yükle (varsa)
load_dotenv(ROOT_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    # Embedding
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
    )

    # ChromaDB
    chroma_persist_dir: Path = Path(
        os.getenv("CHROMA_PERSIST_DIR", str(ROOT_DIR / "chroma_db"))
    )
    chroma_collection_name: str = os.getenv("CHROMA_COLLECTION_NAME", "personal_memory")

    # LLM
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    llm_model: str = os.getenv("LLM_MODEL", "llama3.2")

    # Retrieval
    retrieval_sim_weight: float = float(os.getenv("RETRIEVAL_SIM_WEIGHT", "0.7"))
    retrieval_imp_weight: float = float(os.getenv("RETRIEVAL_IMP_WEIGHT", "0.3"))
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))

    # Memory
    memory_low_importance_threshold: float = float(
        os.getenv("MEMORY_LOW_IMPORTANCE_THRESHOLD", "0.3")
    )
    memory_ttl_days: int = int(os.getenv("MEMORY_TTL_DAYS", "30"))

    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


# Singleton — tüm modüller buradan okumalı
settings = Settings()
