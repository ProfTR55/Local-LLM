"""Ollama üzerinden yerel LLM bağlantısı.

Ollama: Llama 3, Mistral, Gemma gibi LLM'leri yerel olarak çalıştıran araç.
İndir: https://ollama.com

Önkoşullar:
    1. Ollama kurulu ve çalışıyor olmalı
    2. Bir model indirilmiş olmalı: `ollama pull llama3.2`
    3. pip install ollama

Yapılacaklar:
    1. LLMClient sınıfını oluştur
    2. __init__: model adı ve host (default: settings)
       Client oluşturmayı lazy yap (ilk çağrıda)
    3. generate(prompt, system, temperature) -> str
       Tek seferde tüm cevabı döndür (blocking)
    4. stream(prompt, system, temperature) -> Iterator[str]
       Token-by-token streaming generator
       Her chunk için yield kullan

İpucu:
    import ollama
    client = ollama.Client(host="http://localhost:11434")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Blocking
    response = client.chat(model="llama3.2", messages=messages,
                            options={"temperature": 0.7})
    return response["message"]["content"]

    # Streaming
    for chunk in client.chat(model=..., messages=..., stream=True):
        yield chunk["message"]["content"]

TEST:
    Ollama servisi çalışıyor olmalı:
    > ollama serve         # arka planda çalıştır
    > ollama pull llama3.2 # modeli indir (3-4 GB)
    > ollama list          # kurulu modelleri listele
"""

# TODO: LLMClient sınıfını yaz
# TODO: generate() metodu
# TODO: stream() metodu
