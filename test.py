from russian_sensitive_topics import text, preds

def test_post_predict_one():
    if text == "подстава":
        assert preds == "offline_crime"
