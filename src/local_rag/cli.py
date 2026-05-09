"""Terminal arayüzü — `python -m local_rag` veya `local-rag` ile çalıştırılır.

Sorumluluk:
    Kullanıcıdan terminal'den giriş al, RAGPipeline'a yönlendir,
    cevabı renkli/formatlı göster.

Yapılacaklar:
    1. main() fonksiyonu — programın giriş noktası
    2. Karşılama mesajı (rich.Panel ile çerçeveli)
    3. RAGPipeline'ı başlat
    4. while True döngüsü:
        - Kullanıcıdan input al (rich.Prompt.ask veya input())
        - Boş giriş → atla
        - "/exit" veya "/quit" → çıkış
        - "/help" → komut listesi
        - "/memory" → hafızadaki kayıt sayısı
        - Diğer → pipeline.ask() çağır, cevabı yazdır
    5. KeyboardInterrupt / EOFError yakala (Ctrl+C, Ctrl+D)
    6. Hata yönetimi: try/except ile çökme yerine kullanıcıya hata göster

İpucu (rich kütüphanesi):
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt

    console = Console()
    console.print(Panel.fit("Hoşgeldin", border_style="cyan"))
    user_input = Prompt.ask("[bold green]sen[/bold green]")
    console.print(f"[bold magenta]asistan[/bold magenta]: {answer}")

İLERİ:
    - Streaming yanıt (her token geldikçe yazdır) — llm.stream() kullan
    - "/forget <id>" komutu — bir anıyı sil
    - "/recall" komutu — son N anıyı göster
    - Renkli prompt'lar
"""

# TODO: main() fonksiyonu
# TODO: Karşılama paneli
# TODO: RAGPipeline başlat
# TODO: REPL döngüsü
# TODO: Slash komutları (/exit, /help, /memory)
