'''
Descripttion: 
version: 
Author: Tingyu Liu
Date: 2021-10-15 20:03:13
LastEditors: Tingyu Liu
LastEditTime: 2021-10-15 20:03:13
'''
# %%
# afqmc_public转化
import glob
import json


def read_json_pair():
    """Reads data."""
    ps = glob.glob(r"/home/lty/code/question_matching/data/NLP_Datasets/afqmc_public/*.json")
    l = []
    for p in ps:
        with open(p, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                q1, q2, label = data['sentence1'], data['sentence2'], data['label']
                l.append(q2 + "\t" + q1 + "\t" + label)
    print("length", len(l))
    with open("/home/lty/code/question_matching/data/afqmc_public_train_dev.txt", "w") as t:
        for line in l:
            t.write(line)
            t.write("\n")
    print("finish")


read_json_pair()


# %%
# ==================================idea 7========================================================
# 将训练集反过来训练
def read_text_pair(data_path, is_test=False):
    """Reads data."""
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = line.rstrip().split("\t")
            if is_test == False:
                if len(data) != 3:
                    continue
                yield {'query1': data[0], 'query2': data[1], 'label': data[2]}
            else:
                if len(data) != 2:
                    continue
                yield {'query1': data[0], 'query2': data[1]}


dev = read_text_pair("/home/lty/code/question_matching/data/train.txt", is_test=False)
with open("/home/lty/code/question_matching/data/train_reverse.txt", "w") as f:
    for d in dev:
        f.write(d['query2'])
        f.write("\t")
        f.write(d['query1'])
        f.write("\t")
        f.write(d['label'])
        f.write("\n")
print("idea 7 finish")

# %% 数据增强
from nlpcda import Randomword


def read_text_pair(data_path, is_test=False):
    """Reads data."""
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = line.rstrip().split("\t")
            if is_test == False:
                if len(data) != 3:
                    continue
                yield {'query1': data[0], 'query2': data[1], 'label': data[2]}
            else:
                if len(data) != 2:
                    continue
                yield {'query1': data[0], 'query2': data[1]}


dev = read_text_pair("/home/lty/code/question_matching/data/dev.txt")
smw = Randomword(create_num=10, change_rate=0.3)

change_list = []
for line in dev:
    rs1 = smw.replace(line['query1'])
    if len(rs1) > 1:
        for s in rs1[1:]:  # 第一个是本身
            change_list.append([line['query1'], s, line['query2'], line['label']])
print("finish")
# %%
with open("/home/lty/code/question_matching/data/entity_change.txt", "w") as f:
    for line in change_list:
        f.write(line[1])
        f.write("\t")
        f.write(line[2])
        f.write("\t")
        f.write(line[3])
        f.write("\n")


# %%
# 使用微软的数字识别工具
from recognizers_text import Culture, ModelResult
from recognizers_number import NumberRecognizer


def read_text_pair(data_path, is_test=False):
    """Reads data."""
    ret = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = line.rstrip().split("\t")
            if is_test == False:
                if len(data) != 3:
                    continue
                ret.append({'query1': data[0], 'query2': data[1], 'label': data[2]})
            else:
                if len(data) != 2:
                    continue
                ret.append({'query1': data[0], 'query2': data[1]})
    return ret


def unify_number(data):
    """ 将text中的文本数字统一转换成阿拉伯数字 """
    recognizer = NumberRecognizer(Culture.Chinese)
    model = recognizer.get_number_model()

    cnt = 0
    for i, d in tqdm(enumerate(data), desc="#unify_number#"):
        for qk in ["query1", "query2"]:
            query = d[qk]
            res = model.parse(query)
            if len(res) == 0:
                continue

            idx = 0
            new_query = ""
            for item in res:
                new_query += query[idx: item.start]
                new_query += item.resolution['value']
                idx = item.end + 1
                if not item.text.isnumeric():
                    cnt += 1
            new_query += query[idx:]
            data[i][qk] = new_query

    print("convert count: ", cnt)
    return data


test_path = "./data/raw_data/test_B_1118.tsv"
test_data = read_text_pair(test_path, is_test=True)
test_unify_number = unify_number(test_data)
with open('data/tmp_data/test_B_1118_unify_number.tsv', 'w', encoding='utf-8') as f:
    for d in test_unify_number:
        f.write(d['query1'] + '\t' + d['query2'] + '\n')

print()
