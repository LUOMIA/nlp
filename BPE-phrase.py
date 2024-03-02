from nltk.probability import FreqDist
from nltk.text import Text
import collections
import json
import logging

logging.basicConfig(filename='BPE-phrase.log', level=logging.INFO, encoding='gb18030')

def bpe(iteration_num, context_split):
    context_raw = context_split

    chara = Text(context_raw)
    char_fre = FreqDist(chara)


    for j in range(iteration_num):
        pairs = collections.defaultdict(int)
        for i in range(len(context_raw) - 1):
            pairs[context_raw[i], context_raw[i + 1]] += 1

        best = max(pairs, key=pairs.get)
        fre = pairs.get(best)
        if fre == 1:
            print("Too many iterations")
            break

        temp_1 = best[0]
        temp_2 = best[1]
        print("第{}次迭代，将{}与{}合并".format(j + 1, temp_1, temp_2))
        logging.info("第{}次迭代，将{}与{}合并".format(j + 1, temp_1, temp_2))

        for i in range(len(context_raw) - fre - 1):
            if (context_raw[i] == temp_1) and (context_raw[i + 1] == temp_2):
                context_raw[i] = temp_1 + temp_2
                context_raw.pop(i + 1)

        char_fre[temp_1 + temp_2] = fre
        if char_fre.get(temp_1) == fre:
            char_fre.pop(temp_1)
        if char_fre.get(temp_2) == fre:
            char_fre.pop(temp_2)
        print("这是子词压缩词典")
        print(list(char_fre.items()))

        if j == iteration_num-1:
            logging.info(list(char_fre.items()))

if __name__ == '__main__':
    iteration_num = 1000
    context_split = []
    with open('FMM_result.json', 'r', encoding='gb18030') as file:
        seg_result = json.load(file)
    for i in range(len(seg_result)):
        context_split +=seg_result[i]

    bpe(iteration_num, context_split)






