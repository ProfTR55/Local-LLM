"""Smoke testler — modüller doldukça eklenecek.

Smoke test = "duman testi". Sistemin temel parçaları çalışıyor mu, hızlı doğrulama.

Yazılması gereken testler (her modülü implement ettikten sonra ekle):

# ----- Anonimleştirme -----
def test_anonymizer_imports_and_masks_email():
    \"\"\"E-posta maskeleniyor mu?\"\"\"
    from local_rag.anonymizer import anonymize
    out = anonymize("Bana mail at: ali@example.com")
    assert "ali@example.com" not in out
    assert "[MASKED_EMAIL]" in out

def test_anonymizer_masks_phone_tr():
    \"\"\"Türkiye formatında telefon maskeleniyor mu?\"\"\"
    # Test et: "0532 123 45 67" → "[MASKED_PHONE]"

def test_anonymizer_report_counts():
    \"\"\"AnonymizationReport doğru sayım yapıyor mu?\"\"\"
    # Birden fazla PII içeren metni geçir, counts dict'ini doğrula

# ----- Önem skorlama -----
def test_importance_score_in_range():
    \"\"\"Skor [0, 1] aralığında mı?\"\"\"
    from local_rag.importance import calculate_importance
    score = calculate_importance("herhangi bir metin")
    assert 0.0 <= score <= 1.0

def test_importance_high_for_personal_identity():
    \"\"\"Kişisel bilgi içeren mesaj geçici mesajdan yüksek skor mu alıyor?\"\"\"
    high = calculate_importance("Adım Doğukan, alerjim var")
    low = calculate_importance("Bugün hava güzel")
    assert high > low

# ----- Config -----
def test_settings_loads_with_defaults():
    \"\"\"settings nesnesi default değerlerle yükleniyor mu?\"\"\"
    from local_rag.config import settings
    assert settings.retrieval_top_k > 0
    assert abs(settings.retrieval_sim_weight + settings.retrieval_imp_weight - 1.0) < 1e-6

# ----- Retriever ağırlık doğrulaması -----
def test_retriever_rejects_invalid_weights():
    \"\"\"sim_weight + imp_weight != 1.0 ise ValueError fırlatıyor mu?\"\"\"
    import pytest
    from local_rag.retriever import Retriever
    with pytest.raises(ValueError):
        Retriever(sim_weight=0.5, imp_weight=0.6)

# ----- İmport edilebilirlik -----
@pytest.mark.parametrize("module_name", [
    "local_rag", "local_rag.config", "local_rag.anonymizer",
    "local_rag.importance", "local_rag.embedder", "local_rag.memory_store",
    "local_rag.retriever", "local_rag.llm_client", "local_rag.rag_pipeline",
    "local_rag.cli",
])
def test_module_imports(module_name):
    \"\"\"Her modül import edilebilmeli (ağır bağımlılık yüklenmeden).\"\"\"
    import importlib
    importlib.import_module(module_name)

İPUÇLARI:
    - pytest çalıştır: `pytest -v`
    - Tek modül: `pytest tests/test_smoke.py::test_anonymizer_imports_and_masks_email`
    - Coverage: `pytest --cov=local_rag`
"""

# TODO: Yukarıdaki testleri tek tek implement et
# (Önce ilgili modülü doldur, sonra testini aktif hale getir)
