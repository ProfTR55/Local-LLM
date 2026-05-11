from local_rag.ml_importance import (
    ImportanceExample,
    evaluate,
    load_feedback_dataset,
    train_model,
)


UNSEEN_EXAMPLES = [
    ImportanceExample("Benim adım Elif", True),
    ImportanceExample("Adım Mert", True),
    ImportanceExample("Benim ismim Selin", True),
    ImportanceExample("İsmim Deniz", True),
    ImportanceExample("Soyadım Demir", True),
    ImportanceExample("Benim soyadım Kaya", True),
    ImportanceExample("Laktoza alerjim var", True),
    ImportanceExample("Glutene alerjim var", True),
    ImportanceExample("Cevize alerjim var", True),
    ImportanceExample("Kronik migrenim var", True),
    ImportanceExample("Düzenli kullandığım ilacım tansiyon ilacı", True),
    ImportanceExample("Diş hekimi olarak çalışıyorum", True),
    ImportanceExample("Veri analisti olarak görev yapıyorum", True),
    ImportanceExample("Acil serviste hemşireyim", True),
    ImportanceExample("Mesleğim mimarlık", True),
    ImportanceExample("Hukuk fakültesinde okuyorum", True),
    ImportanceExample("Psikoloji bölümünde öğrenciyim", True),
    ImportanceExample("Doktora okuyorum", True),
    ImportanceExample("Vanilyalı kahve seviyorum", True),
    ImportanceExample("Acı sos sevmem", True),
    ImportanceExample("Sessiz çalışma ortamlarını seviyorum", True),
    ImportanceExample("Kısa ve net yanıtları tercih ederim", True),
    ImportanceExample("Kalabalık toplantıları sevmem", True),
    ImportanceExample("Gece çalışmayı tercih ederim", True),
    ImportanceExample("Uzun yürüyüşleri seviyorum", True),
    ImportanceExample("Bugün biraz uykum var", False),
    ImportanceExample("Bugün sinemaya gittim", False),
    ImportanceExample("Bugün çok mutluyum", False),
    ImportanceExample("Bugün başım ağrıyor", False),
    ImportanceExample("Şu an kahvaltı yapıyorum", False),
    ImportanceExample("Şu an dışarıdayım", False),
    ImportanceExample("Şimdi bilgisayarı kapatıyorum", False),
    ImportanceExample("Şimdi toplantıdayım", False),
    ImportanceExample("Birazdan su içeceğim", False),
    ImportanceExample("Hava bugün rüzgarlı", False),
    ImportanceExample("Bugün hava serin", False),
    ImportanceExample("Hava yağmurlu görünüyor", False),
    ImportanceExample("Bugün trafik yoğundu", False),
    ImportanceExample("Şu an çok açım", False),
    ImportanceExample("Az önce kahve içtim", False),
    ImportanceExample("Bugün spor salonuna gittim", False),
    ImportanceExample("Bu hafta biraz yoğunum", False),
    ImportanceExample("Merhaba nasılsın", False),
    ImportanceExample("Selam, ne haber", False),
    ImportanceExample("Tamam görüşürüz", False),
    ImportanceExample("Teşekkürler iyi geceler", False),
    ImportanceExample("Peki sonra konuşuruz", False),
    ImportanceExample("Tamamdır hallederim", False),
    ImportanceExample("Şimdi yemek yiyorum", False),
    ImportanceExample("Bugün markete uğradım", False),
]


def test_unseen_examples_are_not_in_seed_dataset():
    seed_messages = {example.message for example in load_feedback_dataset()}
    unseen_messages = {example.message for example in UNSEEN_EXAMPLES}

    assert unseen_messages.isdisjoint(seed_messages)


def test_model_generalizes_to_unseen_examples():
    model, _, _ = train_model()
    report = evaluate(model, UNSEEN_EXAMPLES)

    assert report.accuracy >= 0.8
    assert report.true_positive >= 20
    assert report.true_negative >= 20
