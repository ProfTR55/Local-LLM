"""Tüm modüller için merkezi konfigürasyon.

Amaç:
    .env dosyasından (varsa) ortam değişkenlerini okumak ve
    tip-güvenli bir Settings nesnesi olarak sunmak.

Yapılacaklar:
    1. python-dotenv ile .env dosyasını yükle
    2. Settings adında bir dataclass oluştur (frozen=True önerilir)
    3. Aşağıdaki alanları ekle (.env.example'a bak):
        - embedding_model: str
        - chroma_persist_dir: Path
        - chroma_collection_name: str
        - ollama_host: str
        - llm_model: str
        - retrieval_sim_weight: float
        - retrieval_imp_weight: float
        - retrieval_top_k: int
        - memory_low_importance_threshold: float
        - memory_ttl_days: int
        - log_level: str
    4. Modül seviyesinde tek bir `settings` nesnesi oluştur (singleton)

İpucu:
    from dotenv import load_dotenv
    import os
    os.getenv("KEY", "default") ile oku, tipini float()/int() ile çevir
"""

# TODO: .env yükle
# TODO: Settings dataclass'ı tanımla
# TODO: settings singleton nesnesini oluştur
