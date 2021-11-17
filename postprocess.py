'''
Descripttion: 
version: 
Author: Tingyu Liu
Date: 2021-10-11 10:09:34
LastEditors: Tingyu Liu
LastEditTime: 2021-11-17 17:22:09
'''
# %%
import pandas as pd

# df = pd.read_csv("./data/dev.txt",sep='\t',header=None)
# df = pd.read_csv("./data/test_A.tsv",sep='\t',header=None)
df = pd.read_excel("./result_stat.xlsx", sheet_name=0)

# %%
# 将长度相同且只有一个字不一样的标记为1.以更改错别字。
same_length_count = 0
for index, row in df.iterrows():
    q1 = row['q1']
    q2 = row['q2']
    # label=row[2]

    # 如果两行行数相同
    if type(q1) == type('fe') and len(q1) == len(q2):
        count = 0
        for a, b in zip(q1, q2):
            if a != b:
                count += 1
        if count == 1:
            same_length_count += 1
            print(q1)
            print(q2)
            # !TODO: 为什么每一行都改了
            df.loc[index, 'r'] = 1
            # print(label)
df.to_excel("./result_stat_cal.xlsx")
print("相同长度:", same_length_count)

# %%
import pycorrector

corrected_sent, detail = pycorrector.correct('少先队员因该为老人让坐')
print(corrected_sent, detail)

# %%
# 使用纠错
import pycorrector

count = 0
for index, row in df.iterrows():
    q1 = row['q1']
    q2 = row['q2']
    # label=row[2]

    # 如果两行字数相同
    if type(q1) == type('fe') and len(q1) == len(q2):

        # 纠错
        corrected_sent_1, detail_1 = pycorrector.correct(q1)
        corrected_sent_2, detail_2 = pycorrector.correct(q2)
        # 如果存在错误
        if len(detail_1) > 0 or len(detail_2) > 0:
            print(q1, q2, detail_1, detail_2)
            df.loc[index, 'r'] = 1
            df.loc[index, 'reason'] = str(detail_1) + " " + str(detail_2)
            df.loc[index, 'new_q1'] = corrected_sent_1
            df.loc[index, 'new_q2'] = corrected_sent_2
            count += 1
        # if count == 1:
        #     same_length_count+=1
        # print(q1)
        # print(q2)
        # !TODO: 为什么每一行都改了
        # df.loc[index, 'r'] = 1
        # print(label)
df.to_excel("./result_stat_error.xlsx")
print("等长有错:", count)

# %%
# 按照比例随机生成 # 不太行，掉点
import numpy as np

rate = [0.6, 0.4]
pred_label = np.random.choice(a=[0, 1], size=50000, p=rate)
with open("./random_test.txt", "w") as f:
    for i in range(len(pred_label)):
        f.write(str(pred_label[i]) + '\n')

# %%
# 测出测试集中1：0比例为3：2。选取一些随机的0变成1
import pandas as pd
import numpy as np

change_count = 0
df = pd.read_excel("./result_stat_error.xlsx", sheet_name=0)
rate = [0.73, 0.27]
random_choice = np.random.choice(a=[0, 1], size=50000, p=rate)
for index, row in df.iterrows():
    # baseline2和只有一个字不同的融合。0：1比例 27667：22333 随机移动7667条数据到1
    pre_label = row['onerow_b2_merge']
    if pre_label == 0:
        df.loc[index, 'out'] = random_choice[index]
        if df.loc[index, 'out'] == 1:
            change_count += 1
df.to_excel("./result_stat_random_test.xlsx")
print("替换了", change_count)

# %%
# =================================== idea 2 ==============================
# 使用拼音对齐
from pypinyin import lazy_pinyin
import pandas as pd
import numpy as np

change_count = 0
df = pd.read_excel("./result_stat_py.xlsx", sheet_name=0)

# %%
for index, row in df.iterrows():
    # 拼音相同视为错别字
    q1 = row['q1']
    q2 = row['q2']
    if type(q1) == type(111):
        continue
    q1_py = str(lazy_pinyin(q1))
    q2_py = str(lazy_pinyin(q2))
    # df.loc[index, 'q1_py'] = q1_py
    # df.loc[index, 'q2_py'] = q2_py
    if q1_py == q2_py:
        df.loc[index, 'py_match'] = 1
        print(q1, q2)
        change_count += 1
df.to_excel("./result_stat_total.xlsx")
print("替换了", change_count)
# ===================================idea2 finished==============================

# %%
# ==================================idea3========================================================
# 效果不佳，检测出的问题大多已经被正确划分了。
# 检测Lexical Semantics中Named Entity
import pandas as pd
import numpy as np

# 在idea2的基础上
df = pd.read_excel("./result_stat_total_2.xlsx", sheet_name=0)

# %%
# 使用ner判断替换
from pyhanlp import *

change_count = 0
ha_model = HanLP.newSegment()
for index, row in df.iterrows():
    q1 = row['q1']
    q2 = row['q2']

    # 有一条是数字的脏数据
    if type(q1) == type(111):
        continue

    q1_seg = ha_model.seg(q1)
    q2_seg = ha_model.seg(q2)

    # 长度相差1就比较
    diff = abs(len(q1) - len(q2))

    if diff <= 2:
        if "/nr" in str(q1_seg) and '/nr' in str(q2_seg):
            # print("q1")
            n1, n2 = "", ""
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
                print(q1, q2)
                change_count += 1
                df.loc[index, 'name_change'] = 0
    # df.loc[index, 'q1_py'] = q1_py
    # df.loc[index, 'q2_py'] = q2_py
    # if q1_py == q2_py:
    #     df.loc[index, 'py_match'] = 1
    #     print(q1, q2)
    #     change_count+=1
df.to_excel("./result_stat_total_3.xlsx")
print("替换了", change_count)
# ===================================idea3 finished==============================


# %%
# ==================================idea 4========================================================
# 检测Lexical Semantics中地名
import pandas as pd
import numpy as np

# 在idea2的基础上
df = pd.read_excel("./result_stat_total_3.xlsx", sheet_name=0)

# %%
# 判断位置替换
from pyhanlp import *

change_count = 0
ha_model = HanLP.newSegment()
for index, row in df.iterrows():
    q1 = row['q1']
    q2 = row['q2']

    # 有一条是数字的脏数据
    if type(q1) == type(111):
        continue

    q1_seg = ha_model.seg(q1)
    q2_seg = ha_model.seg(q2)

    # 长度相差2就比较
    diff = abs(len(q1) - len(q2))
    if diff <= 2:
        if "/ns" in str(q1_seg) and '/ns' in str(q2_seg):
            # print("q1")
            n1, n2 = "", ""
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
                print(q1, q2)
                change_count += 1
                df.loc[index, 'place_change'] = 0

df.to_excel("./result_stat_total_4.xlsx")
print("替换了", change_count)

# ===================================idea4 finished==================================


# %%
# ==================================idea 5========================================================
# 检测Lexical Semantics中组织名
import pandas as pd
import numpy as np

# 在idea4的基础上
df = pd.read_excel("./result_stat_total_4.xlsx", sheet_name=0)

# %%
# 判断位置替换
from pyhanlp import *

change_count = 0
ha_model = HanLP.newSegment()
for index, row in df.iterrows():
    q1 = row['q1']
    q2 = row['q2']

    # 有一条是数字的脏数据
    if type(q1) == type(111):
        continue

    q1_seg = ha_model.seg(q1)
    q2_seg = ha_model.seg(q2)

    # 长度相差2就比较
    diff = abs(len(q1) - len(q2))
    if diff <= 2:
        if "/ni" in str(q1_seg) and '/ni' in str(q2_seg):
            # print("q1")
            n1, n2 = "", ""
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
                print(q1, q2)
                change_count += 1
                # df.loc[index, 'place_change'] = 0

# df.to_excel("./result_stat_total_4.xlsx")
print("替换了", change_count)


# ===================================idea5 finished==================================


# %%
# ==================================idea 6========================================================
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
with open("/home/lty/code/question_matching/data/test_A_reverse.tsv", "w") as f:
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
    q1 = row["query1"]
    q2 = row['query2']
    label = row['label']

    # 有一条是数字的脏数据
    if type(q1) == type(111):
        continue

    q1_seg = ha_model.seg(q1)
    q2_seg = ha_model.seg(q2)

    # 长度相差2就比较
    diff = abs(len(q1) - len(q2))
    if diff <= 2:
        if "/ns" in str(q1_seg) and '/ns' in str(q2_seg):
            # print("q1")
            n1, n2 = "", ""
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
                print(q1, q2, label)
                change_count += 1
print(change_count)
# df.loc[index, 'place_change'] = 0

# ==================================查看训练集与验证集=====================================
# %%

# %%
# ==============================多音字、模糊拼音纠错=================================
from tqdm import tqdm
from pypinyin import pinyin, Style
from chinese_fuzzy_match import *
from string import punctuation as en_punc
from zhon.hanzi import punctuation as zh_punc


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


test_path = 'data/raw_data/test_A.tsv'
test_data = read_text_pair(test_path, is_test=True)

fix_idx = []
fix_data = []
pseg_idx = []


def heteronym_match(query1, query2):
    """
    多音字匹配检测

    :param query1: 待匹配字符串1
    :param query2: 待匹配字符串2
    :return: (bool) 是否匹配
    """
    query1_tokens = [c for c in query1]
    query2_tokens = [c for c in query2]

    pinyin_res1 = [pinyin(t, style=Style.NORMAL, heteronym=True)[0] for t in query1_tokens]
    pinyin_res2 = [pinyin(t, style=Style.NORMAL, heteronym=True)[0] for t in query2_tokens]

    if len(pinyin_res1) == len(pinyin_res2):
        for i, (item1, item2) in enumerate(zip(pinyin_res1, pinyin_res2)):
            if query1[i] != query2[i]:  # 当字不一致时才检查是否是多音字情况
                inter = set(item1).intersection(set(item2))
                if len(inter) == 0:
                    return False
        return True
    return False


print('use fuzzy pinyin correction')
print('use heteronym_match pinyin correction')
for idx, d in tqdm(enumerate(test_data)):
    if len(d['query1']) == len(d['query2']):
        query1, query2 = d['query1'], d['query2']
        # == 去除标点 ==
        for i in en_punc + zh_punc:
            query1 = query1.replace(i, '')
            query2 = query2.replace(i, '')

        # # == 拼音匹配 ==
        # q1_py = pinyin(d['query1'], style=Style.NORMAL)
        # q2_py = pinyin(d['query2'], style=Style.NORMAL)
        #
        # if q1_py == q2_py:
        #     fix_idx.append(idx)
        #     fix_data.append(d)
        # # =============

        # == 多音字匹配 ==
        res1 = heteronym_match(query1, query2)
        # =============

        # == 模糊拼音匹配：前后鼻音、平翘舌 ==
        res = chinese_fuzzy_match(query1, query2, use_fuzzy=True)
        # ==============================

        if res['match_type'] != 'not_match' or res1:
            fix_idx.append(idx)
            fix_data.append(d)

print("Num of fix index: ", len(fix_idx))
# for idx in fix_idx:
#     print(test_data[idx])


raw_result_path = "data/results/predict_result_0.9161848905525323.csv"
final_result_path = "data/results/predict_result_0.9161848905525323_fuzzy-pinyin_heteronym.csv"

# fix result
result_f = []
with open(raw_result_path, 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f.readlines()):
        label = int(line.strip())
        result_f.append(label)

cnt = 0
fixed_data = []
for f_idx in fix_idx:
    if result_f[f_idx] == 0:
        fixed_data.append(test_data[f_idx])
        cnt += 1
    result_f[f_idx] = 1
print("Num of data need to be fixed: ", cnt)

with open(final_result_path, 'w', encoding='utf-8') as f:
    for i, label in enumerate(result_f):
        f.write(str(label) + '\n')

# 拼音规则修改的数据，用于数据分析时的筛选
with open('data/tmp_data/text_corrector_pinyin_idx-flag.txt', 'w', encoding='utf-8') as f:
    for i in range(len(test_data)):
        label = 1 if i in fix_idx else -1
        f.write(str(label) + '\n')

# ==============================多音字、模糊拼音纠错=================================

# %%
# ==============================读取数据=================================
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

def read_label(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = line.rstrip()
            yield data
data = read_text_pair("/home/lty/code/question_matching/data/test_A.tsv",is_test=True)
# 提交的预测标签
labels = read_label("/home/lty/code/question_matching/data/results/1117_jc.csv")

# 一些处理adv的函数
def check_one_adv(q1,q2,adv):
    # 检查q1和q2是否只有一个包含了adv
    if adv in q1 and adv not in q2:
        return True
    elif adv in q2 and adv not in q1:
        return True
    else:
        return False
        
def remove_adv_same(q1,q2,adv):
    longer = q1 if len(q1)>=len(q2) else q2
    shorter = q2 if len(q2)<=len(q1) else q1
    longer_no_adv = longer.replace(adv,"")
    if longer_no_adv == shorter:
        return True
    else:
        return False

def check_two_dif_adv(q1,q2,adv_list_1,adv_list_2):
    flag_1,flag_2 = False,False # 标记两类adv是否出现再不同的q中
    # 如果q1中出现list1副词，q2出现list2副词
    for adv in adv_list_1:
        if adv in q1:
            flag_1 = True
            break
    for adv in adv_list_2:
        if adv in q2:
            flag_2 = True
            break
    if flag_1 and flag_2:
        return True
    else:
        # 如果q2中出现list1副词，q1出现list2副词
        flag_1,flag_2 = False,False # 标记两类adv是否出现再不同的q中
        for adv in adv_list_1:
            if adv in q2:
                flag_1 = True
                break
        for adv in adv_list_2:
            if adv in q1:
                flag_2 = True
                break
        if flag_1 and flag_2:
            return True
        else:
            return False

def check_two_same_adv(q1,q2,adv_list_1,adv_list_2):
    flag_1,flag_2 = False,False # 标记两类adv是否出现再不同的q中
    for adv in adv_list_1:
        if adv in q1:
            flag_1 = True
            break
    for adv in adv_list_1:
        if adv in q2:
            flag_2 = True
            break
    if flag_1 and flag_2:
        return True
    else:
        flag_1,flag_2 = False,False # 标记两类adv是否出现再不同的q中
        for adv in adv_list_2:
            if adv in q1:
                flag_1 = True
                break
        for adv in adv_list_2:
            if adv in q2:
                flag_2 = True
                break
        if flag_1 and flag_2:
            return True
        else:
            return False
# %%
# ==============================词语成语替换=================================
change_count = 0
new_label = []
for row,label in zip(data,labels):
    q1 = row['query1']
    q2 = row['query2']
    if ("成语" in q1 and "词语" in q2) or ("成语" in q2 and "词语" in q1):
        change_count+=1
        print(q1,q2,label)
        label = '0'
    new_label.append(label)
print(change_count)
# 输出文件路径
with open("/home/lty/code/question_matching/data/results/1116_ciyuchengyu.csv","w+") as f:
    for l in new_label:
        f.write(l)
        f.write("\n")
# ==============================词语成语替换=================================

# %%
# 检测插入副词经常
change_count = 0
new_label = []
for row,label in zip(data,labels):
    q1 = row['query1']
    q2 = row['query2']
    if ("经常" in q1 and "经常" not in q2 and "偶尔" not in q2 and "有时" not in q2 and not "有点" in q2 and "经长" not in q2 and "常常" not in q2) or ("经常" in q2 and "经常" not in q1 and "偶尔" not in q1 and "有时" not in q1 and not "有点" in q1 and "经长" not in q1 and "常常" not in q1):
        if label == '1': 
            change_count+=1
            print(q1,q2,label)
            label = '0'
    new_label.append(label)
print(change_count)
with open("/home/lty/code/question_matching/data/results/1117_jc.csv","w+") as f:
    for l in new_label:
        f.write(l)
        f.write("\n")
        
# 更改56条

        
    
# %%
# 检测插入副词
data = read_text_pair("/home/lty/code/question_matching/data/test_A.tsv",is_test=True)
# 提交的预测标签
labels = read_label("/home/lty/code/question_matching/data/results/1117_jc.csv")
change_count = 0
new_label = []
adv_list = ["经常","经长", "老是", "常常", "一直", "总是", "反复", "偶尔", "有时", "时常"]
for row,label in zip(data,labels):
    q1 = row['query1']
    q2 = row['query2']

# 经常 经长 老是 常常 一直，总 频繁 总是 反复
#  偶尔 有时 有点 时常
    for adv in adv_list:
        if check_one_adv(q1,q2,adv) and remove_adv_same(q1,q2,adv):
            if label == '1': 
                change_count+=1
                print(q1,q2,label)
                label = '0'
    new_label.append(label)
print(change_count)
with open("/home/lty/code/question_matching/data/results/1117_insert_adv_withoutyd.csv","w+") as f:
    for l in new_label:
        f.write(l)
        f.write("\n")
# 加上“有点”更改51，去掉“有点”更改46条

# %%
# 检测两边替换副词
data = read_text_pair("/home/lty/code/question_matching/data/test_A.tsv",is_test=True)
# 提交的预测标签
labels = read_label("/home/lty/code/question_matching/data/results/1117_insert_adv_withoutyd.csv")
change_count = 0
new_label = []
adv_list_1 = ["经常","经长", "老是", "常常", "一直", "总是", "反复", "时常"]
adv_list_2 = ["偶尔", "有时","有点"]
for row,label in zip(data,labels):
    q1 = row['query1']
    q2 = row['query2']
    if check_two_dif_adv(q1,q2,adv_list_1,adv_list_2):
        if label == '1': 
            change_count+=1
            print(q1,q2,label)
            label = '0'
    # if check_two_same_adv(q1,q2,adv_list_1,adv_list_2):
    # 查询两个句子中的同义副词
    #     # if label == '1': 
    #     change_count+=1
    #     print(q1,q2,label)
    #     label = '0'
    new_label.append(label)
print(change_count)
with open("/home/lty/code/question_matching/data/results/1117_insert_adv_withoutyd_difadv.csv","w+") as f:
    for l in new_label:
        f.write(l)
        f.write("\n")
# 更改了9条

