import json
from pathlib import Path

from local_rag.ml_importance import (
    ImportanceExample,
    evaluate,
    load_feedback_dataset,
    train_model,
)


TRAINING_EXAMPLES = [
    ImportanceExample("Benim adım Doğukan", True),
    ImportanceExample("Adım Beyza", True),
    ImportanceExample("İsmim Ahmet", True),
    ImportanceExample("Soyadım Bingöl", True),
    ImportanceExample("Benim soyadım Yılmaz", True),
    ImportanceExample("Fıstığa alerjim var", True),
    ImportanceExample("Penisiline alerjim var", True),
    ImportanceExample("Astım hastalığım var", True),
    ImportanceExample("Düzenli kullandığım ilacım var", True),
    ImportanceExample("Yazılım geliştirici olarak çalışıyorum", True),
    ImportanceExample("Öğretmen olarak çalışıyorum", True),
    ImportanceExample("Mesleğim bilgisayar mühendisliği", True),
    ImportanceExample("Backend developer olarak çalışıyorum", True),
    ImportanceExample("Bilgisayar mühendisliği okuyorum", True),
    ImportanceExample("Üniversitede okuyorum", True),
    ImportanceExample("Tıp fakültesinde okuyorum", True),
    ImportanceExample("Kahve sevmem", True),
    ImportanceExample("Çay seviyorum", True),
    ImportanceExample("Baharatlı yemek sevmem", True),
    ImportanceExample("Kısa cevapları tercih ederim", True),
    ImportanceExample("Uzun açıklamaları seviyorum", True),
    ImportanceExample("Bugün hava çok güzel", False),
    ImportanceExample("Bugün dışarı çıktım", False),
    ImportanceExample("Şu an çok yorgunum", False),
    ImportanceExample("Şimdi kahve içiyorum", False),
    ImportanceExample("Bugün biraz canım sıkkın", False),
    ImportanceExample("Şu an açım", False),
    ImportanceExample("Susadım", False),
    ImportanceExample("Hava çok sıcak", False),
    ImportanceExample("Bugün yağmur yağıyor", False),
    ImportanceExample("Bu geçici bir durum", False),
    ImportanceExample("Merhaba", False),
    ImportanceExample("Nasılsın", False),
    ImportanceExample("Selam", False),
    ImportanceExample("Teşekkür ederim", False),
    ImportanceExample("Tamamdır", False),
    ImportanceExample("Görüşürüz", False),
    ImportanceExample("Kahve içtim", False),
    ImportanceExample("Bugün Python çalıştım", False),
    ImportanceExample("Bugün erken kalktım", False),
    ImportanceExample("Bugün spor yaptım", False),
]


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


def _write_feedback_dataset(
    path: Path,
    examples: list[ImportanceExample],
) -> None:
    records = [
        {
            "message": example.message,
            "final_should_store": example.should_store,
        }
        for example in examples
    ]

    path.write_text(
        "\n".join(
            json.dumps(record, ensure_ascii=False)
            for record in records
        )
        + "\n",
        encoding="utf-8",
    )


def test_unseen_examples_are_not_in_training_dataset(tmp_path):
    feedback_path = tmp_path / "importance_feedback.jsonl"
    _write_feedback_dataset(feedback_path, TRAINING_EXAMPLES)

    training_messages = {
        example.message
        for example in load_feedback_dataset(path=feedback_path)
    }
    unseen_messages = {example.message for example in UNSEEN_EXAMPLES}

    assert unseen_messages.isdisjoint(training_messages)


def test_model_generalizes_to_unseen_examples(tmp_path):
    feedback_path = tmp_path / "importance_feedback.jsonl"
    _write_feedback_dataset(feedback_path, TRAINING_EXAMPLES)

    model, _, _ = train_model(path=feedback_path)
    report = evaluate(model, UNSEEN_EXAMPLES)

    assert report.accuracy >= 0.8
    assert report.true_positive >= 20
    assert report.true_negative >= 20
