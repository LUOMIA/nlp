import json

labels = []

with open('pre_vocab.txt', 'r', encoding='gb18030') as file:
    lines = file.readlines()
for line in lines:
    label = []
    j = 0
    words = line.split()
    for i in range(len(words)):
        word = words[i]
        length = len(word)
        if len(word) == 1:
            label.append('S')
        if len(word) == 2:
            label.append('B')
            label.append('E')
        if len(word) > 2:
            for k in range(len(word)):
                if k == 0:
                    label.append('B')
                elif k == len(word) - 1:
                    label.append('E')
                else:
                    label.append('M')

    labels.append(label)

file_path = 'label.json'
with open(file_path, 'w', encoding='gb18030') as file:
    json.dump(labels, file, ensure_ascii=False)