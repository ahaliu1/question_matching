'''
Descripttion: 
version: 
Author: Tingyu Liu
Date: 2021-10-15 20:03:13
LastEditors: Tingyu Liu
LastEditTime: 2021-10-15 20:03:13
'''
#%%
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
                q1,q2,label = data['sentence1'],data['sentence2'],data['label']
                l.append(q2+"\t"+q1+"\t"+label)
    print("length",len(l))
    with open("/home/lty/code/question_matching/data/afqmc_public_train_dev.txt","w") as t:
        for line in l:
            t.write(line)
            t.write("\n")
    print("finish")

read_json_pair()

# %%
#==================================idea 7========================================================
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

dev = read_text_pair("/home/lty/code/question_matching/data/train.txt",is_test=False)
with open("/home/lty/code/question_matching/data/train_reverse.txt","w") as f:
    for d in dev:
        f.write(d['query2'])
        f.write("\t")
        f.write(d['query1'])
        f.write("\t")
        f.write(d['label'])
        f.write("\n")
print("idea 7 finish")


#%% 数据增强
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

change_list=[]
for line in dev:
    rs1 = smw.replace(line['query1'])
    if len(rs1) >1:
        for s in rs1[1:]: # 第一个是本身
            change_list.append([line['query1'],s,line['query2'],line['label']])
print("finish")
# %%
with open("/home/lty/code/question_matching/data/entity_change.txt","w") as f:
    for line in change_list:
        f.write(line[1])
        f.write("\t")
        f.write(line[2])
        f.write("\t")
        f.write(line[3])
        f.write("\n")

# %%
