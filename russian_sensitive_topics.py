# https://huggingface.co/SkolkovoInstitute/russian_toxicity_classifier?text=дурацкий
import json

import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification

text = "подстава"


model_name = 'Skoltech/russian-sensitive-topics'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

with open("id2topic.json") as f:
    target_vaiables_id2topic_dict = json.load(f)

# target_vaiables_id2topic_dict['0']
tokenized = tokenizer.batch_encode_plus([text], max_length=40,
                                        pad_to_max_length=True,
                                        truncation=True,
                                        return_token_type_ids=False)
tokens_ids, mask = torch.tensor(tokenized['input_ids']), torch.tensor(tokenized['attention_mask'])

with torch.no_grad():
    model_output = model(tokens_ids, mask)


def adjust_multilabel(y):
    for y_c in y:
        index = str(int(np.argmax(y_c)))
        y_c = target_vaiables_id2topic_dict[index]
        return y_c


preds = adjust_multilabel(model_output['logits'])
print(preds)
