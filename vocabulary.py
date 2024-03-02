from collections import Counter

file_path = 'pre_vocab.txt'
with open(file_path, 'r', encoding='gb18030') as file:
    file_content = file.read()


# 通过空格切分词语
words = file_content.split()
# 使用Counter统计词语频率
word_freq = Counter(words)
sorted_word_freq = word_freq.most_common()
sorted_vocab = list(word for word, freq in sorted_word_freq)

file_path = 'vocabulary.txt'
with open(file_path, 'w', encoding='gb18030') as file:
    for item in sorted_vocab:
        file.write(item + '\n')
