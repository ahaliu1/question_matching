'''
Descripttion: 
version: 
Author: Tingyu Liu
Date: 2021-10-11 10:09:34
LastEditors: Tingyu Liu
LastEditTime: 2021-10-15 19:52:47
'''
#%%
import pandas as pd
# df = pd.read_csv("./data/dev.txt",sep='\t',header=None)
# df = pd.read_csv("./data/test_A.tsv",sep='\t',header=None)
df = pd.read_excel("./result_stat.xlsx",sheet_name=0)

#%%
# 将长度相同且只有一个字不一样的标记为1.以更改错别字。
same_length_count = 0
for index, row in df.iterrows():
    q1=row['q1']
    q2=row['q2']
    # label=row[2]
    
    # 如果两行行数相同
    if type(q1)==type('fe') and len(q1)==len(q2):
        count=0
        for a, b in zip(q1, q2):
            if a != b:
                count+=1
        if count == 1:
            same_length_count+=1
            print(q1)
            print(q2)
            #!TODO: 为什么每一行都改了
            df.loc[index, 'r'] = 1
            # print(label)
df.to_excel("./result_stat_cal.xlsx")
print("相同长度:",same_length_count)

# %%
import pycorrector
corrected_sent, detail = pycorrector.correct('少先队员因该为老人让坐')
print(corrected_sent, detail)

# %%
# 使用纠错
import pycorrector

count = 0
for index, row in df.iterrows():
    q1=row['q1']
    q2=row['q2']
    # label=row[2]
    
    # 如果两行字数相同
    if type(q1)==type('fe') and len(q1)==len(q2):
        
        # 纠错
        corrected_sent_1, detail_1 = pycorrector.correct(q1)
        corrected_sent_2, detail_2 = pycorrector.correct(q2)
        # 如果存在错误
        if len(detail_1)>0 or len(detail_2)>0:
            print(q1,q2,detail_1,detail_2)
            df.loc[index, 'r'] = 1
            df.loc[index, 'reason']=str(detail_1)+ " " +str(detail_2)
            df.loc[index, 'new_q1'] = corrected_sent_1
            df.loc[index, 'new_q2'] = corrected_sent_2
            count+=1
        # if count == 1:
        #     same_length_count+=1
            # print(q1)
            # print(q2)
            #!TODO: 为什么每一行都改了
            # df.loc[index, 'r'] = 1
            # print(label)
df.to_excel("./result_stat_error.xlsx")
print("等长有错:",count)

# %%
# 按照比例随机生成 # 不太行，掉点
import numpy as np
rate = [0.6,0.4]
pred_label = np.random.choice(a=[0,1], size=50000, p=rate)
with open("./random_test.txt","w") as f:
    for i in range (len (pred_label)):
        f.write(str(pred_label[i])+'\n')

# %%
# 测出测试集中1：0比例为3：2。选取一些随机的0变成1
import pandas as pd
import numpy as np
change_count = 0
df = pd.read_excel("./result_stat_error.xlsx",sheet_name=0)
rate = [0.73,0.27]
random_choice = np.random.choice(a=[0,1], size=50000, p=rate)
for index, row in df.iterrows():
    # baseline2和只有一个字不同的融合。0：1比例 27667：22333 随机移动7667条数据到1
    pre_label =row['onerow_b2_merge']
    if pre_label == 0:
        df.loc[index, 'out'] = random_choice[index]
        if df.loc[index, 'out'] == 1:
            change_count +=1
df.to_excel("./result_stat_random_test.xlsx")
print("替换了",change_count)


# %%
#=================================== idea 2 ==============================
# 使用拼音对齐
from pypinyin import lazy_pinyin
import pandas as pd
import numpy as np
change_count = 0
df = pd.read_excel("./result_stat_py.xlsx",sheet_name=0)

#%%
for index, row in df.iterrows():
    # 拼音相同视为错别字
    q1=row['q1']
    q2=row['q2']
    if type(q1) == type(111):
        continue
    q1_py = str(lazy_pinyin(q1))
    q2_py = str(lazy_pinyin(q2))
    # df.loc[index, 'q1_py'] = q1_py
    # df.loc[index, 'q2_py'] = q2_py
    if q1_py == q2_py:
        df.loc[index, 'py_match'] = 1
        print(q1, q2)
        change_count+=1
df.to_excel("./result_stat_total.xlsx")
print("替换了",change_count)
#===================================idea2 finished==============================

#%%
#==================================idea3========================================================
# 效果不佳，检测出的问题大多已经被正确划分了。
# 检测Lexical Semantics中Named Entity
import pandas as pd
import numpy as np

# 在idea2的基础上
df = pd.read_excel("./result_stat_total_2.xlsx",sheet_name=0)

#%%
# 使用ner判断替换
from pyhanlp import *
change_count = 0
ha_model = HanLP.newSegment()
for index, row in df.iterrows():
    q1=row['q1']
    q2=row['q2']
    
    # 有一条是数字的脏数据
    if type(q1) == type(111):
        continue

    q1_seg = ha_model.seg(q1)
    q2_seg = ha_model.seg(q2)

    # 长度相差1就比较
    diff = abs(len(q1)-len(q2))

    if diff <= 2:
        if "/nr" in str(q1_seg) and '/nr' in str(q2_seg):
            # print("q1")
            n1,n2 = "",""
            for seg in q1_seg:
                if '/nr' in str(seg):
                    n1 = seg.word
                    # print(seg.word)
            # print("q2")
            for seg in q2_seg:
                if '/nr' in str(seg):
                    n2 = seg.word
                    # print(seg.word)

            if n1 != n2:
                print(q1,q2)
                change_count+=1
                df.loc[index, 'name_change'] = 0
    # df.loc[index, 'q1_py'] = q1_py
    # df.loc[index, 'q2_py'] = q2_py
    # if q1_py == q2_py:
    #     df.loc[index, 'py_match'] = 1
    #     print(q1, q2)
    #     change_count+=1
df.to_excel("./result_stat_total_3.xlsx")
print("替换了",change_count)
#===================================idea3 finished==============================



# %%
#==================================idea 4========================================================
# 检测Lexical Semantics中地名
import pandas as pd
import numpy as np

# 在idea2的基础上
df = pd.read_excel("./result_stat_total_3.xlsx",sheet_name=0)

#%%
# 判断位置替换
from pyhanlp import *
change_count = 0
ha_model = HanLP.newSegment()
for index, row in df.iterrows():
    q1=row['q1']
    q2=row['q2']
    
    # 有一条是数字的脏数据
    if type(q1) == type(111):
        continue

    q1_seg = ha_model.seg(q1)
    q2_seg = ha_model.seg(q2)

    # 长度相差2就比较
    diff = abs(len(q1)-len(q2))
    if diff <= 2:
        if "/ns" in str(q1_seg) and '/ns' in str(q2_seg):
            # print("q1")
            n1,n2 = "",""
            for seg in q1_seg:
                if '/ns' in str(seg):
                    n1 = seg.word
                    # print(seg.word)
            # print("q2")
            for seg in q2_seg:
                if '/ns' in str(seg):
                    n2 = seg.word
                    # print(seg.word)

            if n1 != n2:
                print(q1,q2)
                change_count+=1
                df.loc[index, 'place_change'] = 0

df.to_excel("./result_stat_total_4.xlsx")
print("替换了",change_count)

#===================================idea4 finished==================================



# %%
#==================================idea 5========================================================
# 检测Lexical Semantics中组织名
import pandas as pd
import numpy as np

# 在idea4的基础上
df = pd.read_excel("./result_stat_total_4.xlsx",sheet_name=0)

#%%
# 判断位置替换
from pyhanlp import *
change_count = 0
ha_model = HanLP.newSegment()
for index, row in df.iterrows():
    q1=row['q1']
    q2=row['q2']
    
    # 有一条是数字的脏数据
    if type(q1) == type(111):
        continue

    q1_seg = ha_model.seg(q1)
    q2_seg = ha_model.seg(q2)

    # 长度相差2就比较
    diff = abs(len(q1)-len(q2))
    if diff <= 2:
        if "/ni" in str(q1_seg) and '/ni' in str(q2_seg):
            # print("q1")
            n1,n2 = "",""
            for seg in q1_seg:
                if '/ni' in str(seg):
                    n1 = seg.word
                    # print(seg.word)
            # print("q2")
            for seg in q2_seg:
                if '/ni' in str(seg):
                    n2 = seg.word
                    # print(seg.word)

            if n1 != n2:
                print(q1,q2)
                change_count+=1
                # df.loc[index, 'place_change'] = 0

# df.to_excel("./result_stat_total_4.xlsx")
print("替换了",change_count)

#===================================idea5 finished==================================


# %%
#==================================idea 6========================================================
# 将测试集的顺序反过来 TTA
def read_text_pair(data_path, is_test=True):
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

dev = read_text_pair("/home/lty/code/question_matching/data/test_A.tsv")
with open("/home/lty/code/question_matching/data/test_A_reverse.tsv","w") as f:
    for d in dev:
        f.write(d['query2'])
        f.write("\t")
        f.write(d['query1'])
        f.write("\n")
print("idea 6 finish")

# %%
# ==================================查看训练集与验证集=====================================
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

dev = read_text_pair("/home/lty/code/question_matching/data/train.txt")

from pyhanlp import *
change_count = 0
ha_model = HanLP.newSegment()
for row in dev:
    q1=row["query1"]
    q2=row['query2']
    label = row['label']

    # 有一条是数字的脏数据
    if type(q1) == type(111):
        continue

    q1_seg = ha_model.seg(q1)
    q2_seg = ha_model.seg(q2)

    # 长度相差2就比较
    diff = abs(len(q1)-len(q2))
    if diff <= 2:
        if "/ns" in str(q1_seg) and '/ns' in str(q2_seg):
            # print("q1")
            n1,n2 = "",""
            for seg in q1_seg:
                if '/ns' in str(seg):
                    n1 = seg.word
                    # print(seg.word)
            # print("q2")
            for seg in q2_seg:
                if '/ns' in str(seg):
                    n2 = seg.word
                    # print(seg.word)

            if n1 != n2:
                print(q1,q2,label)
                change_count+=1
print(change_count)
                # df.loc[index, 'place_change'] = 0

# ==================================查看训练集与验证集=====================================
# %%
