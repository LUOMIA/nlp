import json
class ForwardMaximumMatching:
    def __init__(self, dictionary):
        self.dictionary = set(dictionary)

    def cut(self, text):
        result = []
        max_len = max(len(word) for word in self.dictionary)

        while text:
            # 选择最大长度的词进行匹配
            word = text[:max_len]
            while word not in self.dictionary and len(word) > 1:
                word = word[:-1]

            result.append(word)
            text = text[len(word):]

        return result


# 示例用法
if __name__ == "__main__":

    with open('vocabulary.txt', 'r', encoding='gb18030') as file:
        vocabulary_list = [line.strip() for line in file]

    # 初始化正向最大匹配对象
    fmm = ForwardMaximumMatching(vocabulary_list)

    with open('cleaned_text.txt', 'r', encoding='gb18030') as file:
        lines = file.readlines()

    seg_lines = []
    with open('FMM_result.txt', 'w', encoding='gb18030') as file:
        for line in lines:
            seg_line = fmm.cut(line)
            seg_line = seg_line[:-1]
            seg_lines.append(seg_line)
            seg_line_str = '/ '.join(seg_line)
            file.write(seg_line_str + '\n')

    with open('FMM_result.json', 'w', encoding='gb18030') as file:
        json.dump(seg_lines, file, ensure_ascii=False)

