# Local RAG — Kişisel Hafıza Destekli Yerel Chatbot

Tamamen yerel çalışan, kullanıcıyla konuştukça **kişisel hafıza** oluşturan ve hiçbir veriyi buluta göndermeyen bir RAG (Retrieval-Augmented Generation) chatbot prototipi.

> TÜBİTAK 2209-A Üniversite Öğrencileri Araştırma Projeleri Destekleme Programı kapsamında geliştirilmektedir.
> **Karadeniz Teknik Üniversitesi** — Danışman: Prof. Dr. Vasif Nabiyev

---

## Özellikler

- **Yerel-öncelikli mimari** — embedding, vektör veritabanı ve LLM aynı makinede; hiçbir veri dışarı çıkmaz
- **Önem ağırlıklı RAG** — her mesaja 0–1 arası önem skoru atanır, geri çağırma `0.7 × benzerlik + 0.3 × önem` ile yapılır
- **Otomatik anonimleştirme** — KVKK uyumlu, kişisel veriler (isim, telefon, adres) maskelenir
- **TTL hafıza yönetimi** — düşük önemli mesajlar zamanla silinir, yüksek önemli olanlar kalır
- **Çoklu LLM desteği** — Ollama üzerinden Llama 3, Mistral, Gemma vs. değiştirilebilir

## Mimari

```
Kullanıcı mesajı
    ↓
[Anonimleştirici]  →  [Önem hesaplayıcı]  →  [Embedder (Sentence-BERT)]
    ↓
[ChromaDB] (yerel vektör veritabanı)
    ↓
Sorgu geldiğinde: hibrit skor ile top-k geri çağırma
    ↓
[LLM (Ollama)]  →  cevap
```

## Teknoloji Yığını

| Katman | Araç |
|--------|------|
| Dil | Python 3.11+ |
| Embedding | `sentence-transformers` (paraphrase-multilingual-MiniLM-L12-v2) |
| Vektör DB | ChromaDB (yerel persist) |
| LLM çalıştırma | Ollama (Llama 3 / Mistral / Gemma) |
| RAG çatısı | Doğrudan Python (LangChain opsiyonel) |
| Test | pytest |

## Klasör Yapısı

```
Local_Llm/
├── src/local_rag/         # Ana paket
│   ├── embedder.py         # Sentence-BERT sarmalayıcı
│   ├── importance.py       # Önem skorlama
│   ├── anonymizer.py       # PII maskeleme
│   ├── memory_store.py     # ChromaDB CRUD
│   ├── retriever.py        # Hibrit skor + top-k
│   ├── llm_client.py       # Ollama bağlantısı
│   ├── rag_pipeline.py     # Uçtan uca akış
│   └── cli.py              # Terminal arayüzü
├── tests/                  # Birim testler
├── notebooks/              # Deney/keşif notebook'ları
├── data/                   # Kullanıcı verisi (git'e dahil DEĞİL)
├── docs/                   # Dökümantasyon
├── requirements.txt        # Çalışma zamanı bağımlılıkları
├── requirements-dev.txt    # Geliştirme bağımlılıkları
├── pyproject.toml          # Paket konfigürasyonu
├── .env.example            # Örnek ortam değişkenleri
└── README.md
```

## Kurulum

### 1. Repo'yu klonla
```bash
git clone https://github.com/<kullanici-adi>/local-rag.git
cd local-rag
```

### 2. Sanal ortam oluştur
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Bağımlılıkları yükle
```bash
pip install -r requirements.txt
# Geliştirme için ek olarak:
pip install -r requirements-dev.txt
```

### 4. Ollama'yı kur ve model indir
```bash
# https://ollama.com/download adresinden Ollama'yı kur
ollama pull llama3.2          # veya: mistral, gemma2
```

### 5. Ortam değişkenlerini ayarla
```bash
cp .env.example .env
# .env dosyasını düzenle (model adı, persist dizini vb.)
```

## Kullanım

```bash
# Paket olarak çalıştır
python -m local_rag

# veya CLI doğrudan
python src/local_rag/cli.py
```

## Geliştirme

```bash
# Testleri çalıştır
pytest

# Kod stili kontrolü
ruff check src/

# Tip kontrolü
mypy src/
```

## Yol Haritası

- [x] Repo iskeleti
- [ ] Embedding modülü (Sentence-BERT)
- [ ] ChromaDB entegrasyonu
- [ ] Önem skoru hesaplama
- [ ] Anonimleştirme katmanı
- [ ] Hibrit skorlu retriever
- [ ] Ollama LLM bağlantısı
- [ ] CLI arayüzü
- [ ] Değerlendirme protokolü (BLEU/BERTScore + insan testi)
- [ ] Streamlit arayüzü
- [ ] Akademik rapor

## Lisans

MIT — bkz. [LICENSE](./LICENSE)

## Atıf

Bu projeyi akademik bir çalışmada kullanırsanız aşağıdaki şekilde atıfta bulunabilirsiniz:

```bibtex
@software{local_rag_2026,
  author = {Bingöl, Muhammet Doğukan and Aydın, Beyza Sude},
  title  = {Local RAG: Kişisel Hafıza Destekli Yerel Chatbot},
  year   = {2026},
  url    = {https://github.com/<kullanici-adi>/local-rag}
}
```

## Teşekkür

Bu çalışma TÜBİTAK 2209-A programı kapsamında desteklenmiştir.
