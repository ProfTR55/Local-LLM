from local_rag.importance import calculate_importance, explain_importance, should_ask_feedback


def test_identity_should_store():
    decision = explain_importance("Benim adım Doğukan")

    assert decision.score >= 0.7
    assert decision.label == "high"
    assert decision.category == "identity"
    assert decision.should_store is True


def test_health_should_store():
    decision = explain_importance("Fıstığa alerjim var")

    assert decision.score >= 0.7
    assert decision.label == "high"
    assert decision.category == "health"
    assert decision.should_store is True


def test_temporary_should_not_store():
    decision = explain_importance("Şu an açım ve yorgunum")

    assert decision.should_store is False
    assert decision.score < 0.7


def test_weather_should_not_store():
    decision = explain_importance("Bugün hava çok güzel")

    assert decision.should_store is False


def test_explicit_hobby_should_store():
    decision = explain_importance("Hobim satranç")

    assert decision.category == "hobby"
    assert decision.should_store is True
    assert decision.should_ask_feedback is False
    assert decision.action == "store"


def test_explicit_phobia_should_store():
    decision = explain_importance("Yükseklik fobim var")

    assert decision.category == "phobia"
    assert decision.should_store is True


def test_address_should_store():
    decision = explain_importance("Adresim Kadıköy İstanbul")

    assert decision.category == "address"
    assert decision.should_store is True


def test_phone_number_should_store():
    decision = explain_importance("Telefon numaram 0532 123 45 67")

    assert decision.category == "contact"
    assert decision.should_store is True


def test_email_should_store():
    decision = explain_importance("E-postam ali@example.com")

    assert decision.category == "email"
    assert decision.should_store is True


def test_birth_date_should_store_when_explicit():
    decision = explain_importance("Doğum tarihim 01.01.2000")

    assert decision.category == "birth_date"
    assert decision.should_store is True


def test_location_should_store_when_explicit():
    decision = explain_importance("Yaşadığım şehir İstanbul")

    assert decision.category == "location"
    assert decision.should_store is True


def test_context_dependent_verbs_are_not_stored_by_default():
    messages = [
        "Şu an çalışıyorum",
        "Hukuk fakültesinde okuyorum",
        "Kahve seviyorum",
        "Bugün İstanbul'dayım",
    ]

    for message in messages:
        decision = explain_importance(message)

        assert decision.category is None
        assert decision.should_store is False


def test_uncertain_score_should_ask_feedback():
    decision = explain_importance("Herhangi bir metin")

    assert decision.should_store is False
    assert decision.should_ask_feedback is True
    assert decision.action == "ask_feedback"


def test_blank_message_should_be_ignored():
    decision = explain_importance("   ")

    assert decision.score == 0.0
    assert decision.label == "low"
    assert decision.category is None
    assert decision.should_store is False
    assert decision.should_ask_feedback is False
    assert decision.action == "ignore"


def test_feedback_range_excludes_store_threshold():
    assert should_ask_feedback(0.45) is True
    assert should_ask_feedback(0.69) is True
    assert should_ask_feedback(0.70) is False


def test_score_is_clamped_between_zero_and_one():
    score = calculate_importance(
        "Adım Doğukan ismim Doğukan soyadım Bingöl alerjim var hastalığım var ilacım var"
    )

    assert 0.0 <= score <= 1.0
