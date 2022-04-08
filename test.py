from russian_sensitive_topics import tokenizer, preds


def test_post_predict_one():
    tokenized = tokenizer.batch_encode_plus(['взорвать дом'], max_length=40,
                                            pad_to_max_length=True,
                                            truncation=True,
                                            return_token_type_ids=False)

    assert preds == "offline_crime"
