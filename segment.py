import re
import json


def clean_text(text):
    segtext = []
    # 去除每条新闻最前面的日期
    pattern = re.compile(r'\d{8}-\d{2}-\d{3}-\d{3}')
    cleaned_text = re.sub(pattern, '', text)
    # 去除不同天新闻之间的换行符
    cleaned_text = re.sub(r'\n{2,}', '\n', cleaned_text)
    # 去除斜杠及词性标注
    pattern_slash = re.compile(r'/[a-zA-Z]+')
    cleaned_text = re.sub(pattern_slash, '', cleaned_text)
    # 去除空格
    cleaned_text = re.sub(' ', '', cleaned_text)
    # 去除中括号及词性标注
    cleaned_text = re.sub(r'\[*', '', cleaned_text)
    cleaned_text = re.sub(r'\][a-zA-Z]*', '', cleaned_text)

    lines = cleaned_text.split('\n')
    for line in lines:
        characters = list(line)
        segtext.append(characters)
    return segtext


file_path = 'ChineseCorpus199801.txt'
with open(file_path, 'r', encoding='gb18030') as file:
    file_content = file.read()

result = clean_text(file_content)

file_path = 'segmented_text.json'
with open(file_path, 'w', encoding='gb18030') as file:
    json.dump(result, file, ensure_ascii=False)
