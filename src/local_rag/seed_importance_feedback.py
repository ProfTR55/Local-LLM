import json
from datetime import datetime, timezone
from pathlib import Path

from local_rag.importance import explain_importance


FEEDBACK_PATH = Path("data/importance_feedback.jsonl")


SEED_EXAMPLES = [
    # HIGH / should_store=True — identity
    ("Benim adım Doğukan", True),
    ("Adım Beyza", True),
    ("İsmim Ahmet", True),
    ("Soyadım Bingöl", True),
    ("Benim soyadım Yılmaz", True),
    ("Adım Elif", True),
    ("İsmim Mert", True),
    ("Benim adım Zeynep", True),
    ("Soyadım Kaya", True),
    ("Benim ismim Deniz", True),

    # HIGH / should_store=True — location / language
    ("Ankara'da yaşıyorum", True),
    ("İstanbul'da oturuyorum", True),
    ("İzmir'de yaşıyorum", True),
    ("Türkçe konuşmayı tercih ederim", True),
    ("İngilizce cevapları da anlayabiliyorum", True),

    # HIGH / should_store=True — health
    ("Fıstığa alerjim var", True),
    ("Penisiline alerjim var", True),
    ("Alerjim var, özellikle toza karşı", True),
    ("Astım hastalığım var", True),
    ("Düzenli kullandığım ilacım var", True),
    ("İlaç kullanıyorum", True),
    ("Lateks alerjim var", True),
    ("Kedi tüyüne alerjim var", True),
    ("Glutene karşı hassasiyetim var", True),
    ("Laktoz bana dokunuyor", True),
    ("Migren hastalığım var", True),
    ("Diyabet hastalığım var", True),
    ("Kalp ritim problemim var", True),
    ("Antibiyotik kullanıyorum", True),
    ("Her gün tansiyon ilacı kullanıyorum", True),
    ("Doktorum aspirin kullanmamamı söyledi", True),

    # HIGH / should_store=True — work
    ("Yazılım geliştirici olarak çalışıyorum", True),
    ("Öğretmen olarak çalışıyorum", True),
    ("Mesleğim bilgisayar mühendisliği", True),
    ("Backend developer olarak çalışıyorum", True),
    ("Veri bilimci olarak çalışıyorum", True),
    ("Frontend developer olarak çalışıyorum", True),
    ("Makine mühendisi olarak çalışıyorum", True),
    ("Freelance tasarımcı olarak çalışıyorum", True),
    ("Mesleğim doktorluk", True),
    ("Hemşire olarak çalışıyorum", True),
    ("Ürün yöneticisi olarak çalışıyorum", True),
    ("Mobil uygulama geliştiriyorum", True),
    ("Python geliştiricisi olarak çalışıyorum", True),

    # HIGH / should_store=True — education
    ("Bilgisayar mühendisliği okuyorum", True),
    ("Üniversitede okuyorum", True),
    ("Tıp fakültesinde okuyorum", True),
    ("Yüksek lisans okuyorum", True),
    ("Üniversitede hukuk okuyorum", True),
    ("Psikoloji okuyorum", True),
    ("Makine mühendisliği okuyorum", True),
    ("İngilizce öğretmenliği okuyorum", True),
    ("Lise öğrencisiyim", True),
    ("Doktora yapıyorum", True),

    # HIGH / should_store=True — goals / projects / skills
    ("Python öğreniyorum", True),
    ("React öğreniyorum", True),
    ("Makine öğrenmesi öğreniyorum", True),
    ("Veri analizi öğreniyorum", True),
    ("Yapay zeka alanında kendimi geliştirmek istiyorum", True),
    ("Kariyer hedefim yazılım mühendisi olmak", True),
    ("Amacım İngilizcemi geliştirmek", True),
    ("Bu dönem tez yazıyorum", True),
    ("Local LLM projesi geliştiriyorum", True),
    ("RAG tabanlı chatbot geliştiriyorum", True),

    # HIGH / should_store=True — preferences
    ("Kahve sevmem", True),
    ("Çay seviyorum", True),
    ("Baharatlı yemek sevmem", True),
    ("Sessiz ortamda çalışmayı tercih ederim", True),
    ("Kısa cevapları tercih ederim", True),
    ("Uzun açıklamaları seviyorum", True),
    ("Kitap okumayı seviyorum", True),
    ("Korku filmi sevmem", True),
    ("Futbol izlemeyi seviyorum", True),
    ("Kalabalık ortamları sevmem", True),
    ("Sabah erken çalışmayı tercih ederim", True),
    ("Gece çalışmayı sevmem", True),
    ("Detaylı açıklamaları tercih ederim", True),
    ("Özet cevapları seviyorum", True),
    ("Kod örnekleriyle anlatmanı tercih ederim", True),
    ("Resmi olmayan bir dil kullanmanı tercih ederim", True),
    ("Acı yemek sevmem", True),
    ("Vegan besleniyorum", True),
    ("Et yemem", True),
    ("Deniz ürünleri sevmem", True),
    ("Şekersiz kahve seviyorum", True),
    ("Sütlü kahve tercih ederim", True),

    # HIGH / should_store=True — communication / routine / constraints
    ("Toplantılarda kısa notlar tercih ederim", True),
    ("Mail yazarken resmi ton tercih ederim", True),
    ("Bana Türkçe cevap vermeni tercih ederim", True),
    ("Teknik konularda adım adım anlatmanı severim", True),
    ("Hafta içi sabahları spor yaparım", True),
    ("Her pazartesi rapor hazırlarım", True),
    ("Düzenli olarak meditasyon yaparım", True),
    ("Genelde toplu taşıma kullanırım", True),
    ("Hafta sonları ders çalışırım", True),
    ("Evden çalışıyorum", True),
    ("Uzaktan çalışmayı tercih ederim", True),
    ("Ofiste çalışmayı sevmem", True),
    ("Takım çalışmasını seviyorum", True),
    ("Bireysel çalışmayı tercih ederim", True),

    # LOW / should_store=False — temporary
    ("Bugün hava çok güzel", False),
    ("Bugün dışarı çıktım", False),
    ("Şu an çok yorgunum", False),
    ("Şimdi kahve içiyorum", False),
    ("Bugün biraz canım sıkkın", False),
    ("Şu an açım", False),
    ("Susadım", False),
    ("Hava çok sıcak", False),
    ("Bugün yağmur yağıyor", False),
    ("Bu geçici bir durum", False),
    ("Kahve içtim", False),
    ("Bugün Python çalıştım", False),
    ("Bugün erken kalktım", False),
    ("Bugün spor yaptım", False),
    ("Bu aralar çok yorgunum", False),
    ("Bugün kalabalık bir yere gittim", False),
    ("Şimdi dışarı çıkıyorum", False),
    ("Az önce yemek yedim", False),
    ("Birazdan uyuyacağım", False),
    ("Bugün markete gittim", False),
    ("Şu an müzik dinliyorum", False),
    ("Bugün film izledim", False),
    ("Bugün çok yürüdüm", False),
    ("Şu anda bilgisayar başındayım", False),

    # LOW / should_store=False — weather / environment
    ("Bugün hava kapalı", False),
    ("Şimdi yağmur başladı", False),
    ("Bugün güneşli bir gün", False),
    ("Hava biraz rüzgarlı", False),
    ("Bugün dışarısı soğuk", False),
    ("Bu sabah trafik vardı", False),
    ("Bugün otobüs gecikti", False),
    ("Şu an dışarısı gürültülü", False),
    ("Bugün hava nemli", False),
    ("Şimdi pencereyi açtım", False),

    # LOW / should_store=False — temporary physical / emotional state
    ("Bugün biraz başım ağrıyor", False),
    ("Şu an uykum var", False),
    ("Bugün canım tatlı istedi", False),
    ("Şimdi su içiyorum", False),
    ("Biraz önce kahvaltı yaptım", False),
    ("Bugün öğle yemeği yedim", False),
    ("Akşam ne yesem diye düşünüyorum", False),
    ("Şu an biraz meşgulüm", False),
    ("Şimdi mola verdim", False),
    ("Bugün biraz sıkıldım", False),
    ("Şu an kafam karışık", False),

    # LOW / should_store=False — casual
    ("Merhaba", False),
    ("Nasılsın", False),
    ("Selam", False),
    ("Teşekkür ederim", False),
    ("Tamamdır", False),
    ("Görüşürüz", False),
    ("Selam nasılsın", False),
    ("İyi akşamlar", False),
    ("Günaydın", False),
    ("İyi geceler", False),
    ("Hoşça kal", False),
    ("Sağ ol", False),
    ("Rica ederim", False),
    ("Harika oldu", False),
    ("Anladım", False),
    ("Olur", False),
    ("Peki", False),

    # LOW / should_store=False — one-time activity
    ("Bugün Python videosu izledim", False),
    ("Bugün React dersi çalıştım", False),
    ("Bugün makale okudum", False),
    ("Bugün not aldım", False),
    ("Bugün proje dosyalarını düzenledim", False),
    ("Bugün bilgisayarı temizledim", False),
    ("Bugün kod yazdım", False),
    ("Bugün test çalıştırdım", False),
    ("Bugün hata aldım", False),
    ("Bugün terminal açılmadı", False),
    ("Bugün kahve aldım", False),
    ("Bugün çay içtim", False),
    ("Bugün geç kaldım", False),
    ("Bugün alışveriş yaptım", False),
    ("Bugün eve geldim", False),
    ("Bugün arkadaşımı gördüm", False),
    ("Bugün kısa bir yürüyüş yaptım", False),
    ("Şimdi bir şeyler okuyorum", False),
    ("Bugün defterimi bulamadım", False),
    ("Bugün telefonumun şarjı bitti", False),
    ("Şu an odadayım", False),
    ("Bugün bilgisayarım yavaşladı", False),
    ("Bugün internet kesildi", False),
    ("Şimdi dosya indiriyorum", False),
    ("Bugün masa düzenledim", False),
    ("Bugün kargo geldi", False),
    ("Bugün ders notlarına baktım", False),
    ("Bugün kısa bir toplantı oldu", False),
    ("Bugün e-posta kontrol ettim", False),
    ("Şu an cevap yazıyorum", False),
    ("Bugün planım değişti", False),
    ("Birazdan yemek yapacağım", False),
]

def append_seed_feedback() -> None:
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)

    existing_messages = set()

    if FEEDBACK_PATH.exists():
        with FEEDBACK_PATH.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                    existing_messages.add(record.get("message"))
                except json.JSONDecodeError:
                    continue

    added_count = 0

    with FEEDBACK_PATH.open("a", encoding="utf-8") as file:
        for message, final_should_store in SEED_EXAMPLES:
            if message in existing_messages:
                continue

            decision = explain_importance(message)

            record = {
                "message": message,
                "predicted_score": decision.score,
                "predicted_label": decision.label,
                "predicted_category": decision.category,
                "predicted_should_store": decision.should_store,
                "predicted_should_ask_feedback": decision.should_ask_feedback,
                "predicted_action": decision.action,
                "user_feedback": "up" if final_should_store else "down",
                "final_should_store": final_should_store,
                "source": "seed",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            added_count += 1

    print(f"{added_count} yeni seed feedback kaydı eklendi.")
    print(f"Dataset yolu: {FEEDBACK_PATH}")


if __name__ == "__main__":
    append_seed_feedback()