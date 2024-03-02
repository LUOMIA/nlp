import json
from sklearn.model_selection import train_test_split
import torch


def prepare_sequence(seq, to_ix):
    idxs = [to_ix.get(w, to_ix["<UNK>"]) for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


datas_ = []
with open('segmented_text.json', 'r', encoding='gb18030') as file:
    segmented_text = json.load(file)

with open('label.json', 'r', encoding='gb18030') as file:
    label = json.load(file)

for i in range(len(se)):
    data_ = (segmented_text[i], label[i])
    datas_.append(data_)
training_data, test_data = train_test_split(datas_, test_size=0.1, random_state=42)

word_to_ix = {"<UNK>": 10000}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

for sentence, true_tags in test_data:
    predicted_word = []
    sentence1 = [word if word in word_to_ix else "<UNK>" for word in sentence]
    sentence_in = prepare_sequence(sentence1, word_to_ix)
    print(sentence_in)

