from russian_sensitive_topics import prediction


def test_post_predict_one():
    assert prediction("подстава") == "offline_crime"
#