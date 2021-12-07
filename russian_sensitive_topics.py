# https://huggingface.co/SkolkovoInstitute/russian_toxicity_classifier?text=дурацкий
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification

model_name = 'Skoltech/russian-sensitive-topics'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)
import json

with open("id2topic.json") as f:
    target_vaiables_id2topic_dict = json.load(f)

target_vaiables_id2topic_dict['0']
tokenized = tokenizer.batch_encode_plus(['взорвать дом'], max_length=512,
                                        pad_to_max_length=True,
                                        truncation=True,
                                        return_token_type_ids=False)
tokens_ids, mask = torch.tensor(tokenized['input_ids']), torch.tensor(tokenized['attention_mask'])

with torch.no_grad():
    model_output = model(tokens_ids, mask)


def adjust_multilabel(y, is_pred=False):
    y_adjusted = []
    for y_c in y:
        y_test_curr = [0] * 19
        index = str(int(np.argmax(y_c)))
        y_c = target_vaiables_id2topic_dict[index]
    return y_c


preds = adjust_multilabel(model_output['logits'], is_pred=True)
print(preds)
