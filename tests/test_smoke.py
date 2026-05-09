"""Smoke testler — ağır bağımlılık (Sentence-BERT, ChromaDB, Ollama) yüklemeden
modüllerin import edilebildiğini ve saf-Python fonksiyonların çalıştığını doğrula.

Daha kapsamlı entegrasyon testleri ileride `test_integration.py` içine yazılacak.
"""

from __future__ import annotations

import pytest


# ----- Anonimleştirme -----

def test_anonymizer_imports_and_masks_email():
    from local_rag.anonymizer import anonymize

    out = anonymize("Bana mail at: ali@example.com diyorlar")
    assert "ali@example.com" not in out
    assert "[MASKED_EMAIL]" in out


def test_anonymizer_masks_phone_tr():
    from local_rag.anonymizer import anonymize

    out = anonymize("Numaram 0532 123 45 67")
    assert "0532" not in out
    assert "[MASKED_PHONE]" in out


def test_anonymizer_report_counts():
    from local_rag.anonymizer import anonymize_with_report

    text = "Mail: a@b.com, telefon: +905551234567, başka mail: c@d.io"
    report = anonymize_with_report(text)
    assert report.counts.get("EMAIL") == 2
    assert report.counts.get("PHONE") == 1


# ----- Önem skorlama -----

def test_importance_score_in_range():
    from local_rag.importance import calculate_importance

    score = calculate_importance("Adım Doğukan, KTÜ'de okuyorum")
    assert 0.0 <= score <= 1.0


def test_importance_high_for_personal_identity():
    from local_rag.importance import calculate_importance

    high = calculate_importance("Adım Doğukan, glütene alerjim var")
    low = calculate_importance("Bugün hava çok güzel şu an")
    assert high > low, f"yüksek önemli mesaj ({high}) düşük önemliden ({low}) büyük olmalı"


def test_importance_features_explainable():
    from local_rag.importance import _calculate_features

    features = _calculate_features("Adım Doğukan")
    assert len(features.matched_high) >= 1
    assert features.final > 0.5


# ----- Config -----

def test_settings_loads_with_defaults():
    from local_rag.config import settings

    assert 0.0 <= settings.retrieval_sim_weight <= 1.0
    assert 0.0 <= settings.retrieval_imp_weight <= 1.0
    assert abs(settings.retrieval_sim_weight + settings.retrieval_imp_weight - 1.0) < 1e-6
    assert settings.retrieval_top_k > 0


# ----- Retriever ağırlık doğrulaması -----

def test_retriever_rejects_invalid_weights():
    from local_rag.retriever import Retriever

    with pytest.raises(ValueError):
        Retriever(sim_weight=0.5, imp_weight=0.6)  # toplam 1.1


# ----- Modül import edilebilirliği (heavy deps yüklenmeden) -----

@pytest.mark.parametrize(
    "module_name",
    [
        "local_rag",
        "local_rag.config",
        "local_rag.anonymizer",
        "local_rag.importance",
        "local_rag.embedder",
        "local_rag.memory_store",
        "local_rag.retriever",
        "local_rag.llm_client",
        "local_rag.rag_pipeline",
        "local_rag.cli",
    ],
)
def test_module_imports(module_name: str):
    import importlib

    module = importlib.import_module(module_name)
    assert module is not None
