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


def test_preference_should_store():
    decision = explain_importance("Kahve sevmem")

    assert decision.category == "preference"
    assert decision.should_store is True
    assert decision.should_ask_feedback is False
    assert decision.action == "store"


def test_uncertain_score_should_ask_feedback():
    decision = explain_importance("Herhangi bir metin")

    assert decision.should_store is False
    assert decision.should_ask_feedback is True
    assert decision.action == "ask_feedback"


def test_feedback_range_excludes_store_threshold():
    assert should_ask_feedback(0.45) is True
    assert should_ask_feedback(0.69) is True
    assert should_ask_feedback(0.70) is False


def test_score_is_clamped_between_zero_and_one():
    score = calculate_importance(
        "Adım Doğukan ismim Doğukan soyadım Bingöl alerjim var hastalığım var ilacım var"
    )

    assert 0.0 <= score <= 1.0
