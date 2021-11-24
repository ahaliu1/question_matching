# coding: utf-8
"""
jx part: 3-针对Syntactic Structure的规则
这里实现了:
    (1)Symmetry识别;
    (2)Asymmetry识别;
    (3)NegativeAsymmetry识别;
PS:
执行该后处理脚本需要提前处理好
    1. 自定义近义词表. 用于替换一些近义词如: "男生,男孩,男孩子,男的"
    2. 一些基于星座和生肖的搭配如: "摩羯座男", "女水瓶座", "女牛"
"""
import json
import argparse
from tqdm import tqdm

import jieba
import Levenshtein
from LAC import LAC
import jieba.posseg as pseg

from src.utils import get_run_time

lac = LAC(mode='lac')

parser = argparse.ArgumentParser()
parser.add_argument("--test_path", type=str, required=True, help="test路径")
parser.add_argument("--test_unify_number_path", type=str, required=True, help="经过<统一数字>预处理的test路径")
parser.add_argument("--raw_result_path", type=str, required=True, help="上一步处理后的预测结果路径")
parser.add_argument("--final_result_path", type=str, required=True, help="本次处理预测结果输出路径")
args = parser.parse_args()

# ======================= 修改词表 =======================
# 载入自定义的近义词表
similar_words_path = "data/tmp_data/similar_words.txt"
similar_words_dict = {}
with open(similar_words_path, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        words = line.strip().split(',')
        for i in range(len(words)):
            similar_words_dict[words[i]] = words[0]  # key为被替换的词，value为替换后的词（同一行的第一个词）

# 载入lac自定义词表
lac.load_customization('data/lac_custom.txt', sep=None)


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


def find_k_idx(pseg_cut_res, k):
    k_idx = -1
    for idx, (w, f) in enumerate(pseg_cut_res):
        if 0 < idx < len(pseg_cut_res) - 1:
            if w == k and pseg_cut_res[idx - 1][1] == pseg_cut_res[idx + 1][1]:
                k_idx = idx
                break
    return k_idx


# ======================= Symmetry =======================
def check_Symmetry(q1, q2):
    """ Syntactic Structure - Symmetry """

    if len(q1) == len(q2):
        for k in ['和', '与', '又', '还是'] + ["乘以", "加", "乘"]:
            # for k in ['和', '与', '又', '还是']:  # 77
            # for k in ['和', '与', '又']: # 67
            if k in q1 and k in q2:
                res1 = lac_run_to_pseg_cut(q1)
                k_idx_1 = find_k_idx(res1, k)
                if k_idx_1 != -1:
                    res2 = lac_run_to_pseg_cut(q2)
                    k_idx_2 = find_k_idx(res2, k)
                    if res1[k_idx_1 - 1][0] == res2[k_idx_2 + 1][0] and res1[k_idx_1 + 1][0] == res2[k_idx_2 - 1][0]:
                        return True
    # 方法1：修改67个结果

    # #  ============== 11.11 方法2
    # if len(q1) == len(q2):
    #     res_exchange = check_exchange_replace(q1, q2)
    #     if res_exchange is None:
    #         return False
    #     for k in ['和', '与', '又', '还是']:
    #         if k in q1 and k in q2:
    #             idx_k_1 = q1.find(k)
    #             if res_exchange[0][1] < idx_k_1 < res_exchange[1][0]: # 188
    #                 return True
    return False


def check_Symmetry1111(q1, q2):
    """ Syntactic Structure - Symmetry """

    # if len(q1) == len(q2):
    #     for k in ['和', '与', '又', '还是']: # 77
    #     for k in ['和', '与', '又']: # 67
    #         if k in q1 and k in q2:
    #             res1 = [(w, f) for w, f in pseg.cut(q1)]
    #
    #             k_idx_1 = find_k_idx(res1, k)
    #             if k_idx_1 != -1:
    #                 res2 = [(w, f) for w, f in pseg.cut(q2)]
    #                 k_idx_2 = find_k_idx(res2, k)
    #                 if res1[k_idx_1 - 1][0] == res2[k_idx_2 + 1][0] and res1[k_idx_1 + 1][0] == res2[k_idx_2 - 1][0]:
    #                     return True
    # # 方法1：修改67个结果

    #  ============== 11.11 方法2
    if len(q1) == len(q2):
        res_exchange = check_exchange_replace(q1, q2)
        if res_exchange is None:
            return False
        for k in ['和', '与', '又', '还是'] + ["乘以", "加", "乘"]:
            if k in q1 and k in q2:
                idx_k_1 = q1.find(k)
                if res_exchange[0][1] < idx_k_1 < res_exchange[1][0]:
                    w1 = q1[res_exchange[0][0]: res_exchange[0][1] + 1]
                    w2 = q1[res_exchange[1][0]: res_exchange[1][1] + 1]
                    if w1.strip('.').isdigit() and w2.strip('.').isdigit():  # 若交换的两个都是数字,则需要用分词再检查一次(分词工具对于数字边界识别得更好)
                        return check_Symmetry(q1, q2)
                    return True

        # (补充1.)距离 & 距离相关的时间
        # 235(加了距离后的ss样本数
        w1 = q1[res_exchange[0][0]: res_exchange[0][1] + 1]
        w2 = q1[res_exchange[1][0]: res_exchange[1][1] + 1]
        if lac.run(w1)[1][0] == lac.run(w2)[1][0]:
            for k in ['距离', '多远', '公里', '米', '时间']:
                if '时间表' in q1 and '时间表' in q2:
                    return False
                if k in q1 and k in q2:
                    return True
        # (补充2.) 如:变黑变浓/变浓变黑
        # 241
        res_exchange = check_exchange_replace(q1, q2, forward=True)  # 向前扫描
        if res_exchange[0][1] + 1 == res_exchange[1][0] and \
                q1[res_exchange[0][0]] == q1[res_exchange[1][0]]:  # 两个交换的单词紧邻 & 它们首字符相同
            return True

    return False


def check_Symmetry1122(q1, q2):
    """ Syntactic Structure - Symmetry """

    #  ============== 11.11 方法2
    if len(q1) == len(q2):
        res_exchange = check_exchange_replace(q1, q2, forward=True, by_lac=True)
        if res_exchange is None:
            return False
        w1 = q1[res_exchange[0][0]: res_exchange[0][1] + 1]
        w2 = q1[res_exchange[1][0]: res_exchange[1][1] + 1]

        for k in ['和', '与', '及', '又', '还是', '跟'] + ["乘以", "加", "乘", "✖", "×", "+", "＋"] + ['炒', '爆', '煎', '晒']:
            if k in q1 and k in q2:
                idx_k_1 = q1.find(k, res_exchange[0][1] + 1, res_exchange[1][0])
                if res_exchange[0][1] + 1 == idx_k_1 == res_exchange[1][0] - len(k):
                    return True

                if res_exchange[0][1] < idx_k_1 < res_exchange[1][0]:

                    if w1.strip('.').isdigit() and w2.strip('.').isdigit():  # 若交换的两个都是数字,则需要用分词再检查一次(分词工具对于数字边界识别得更好)
                        return check_Symmetry(q1, q2)
                    return True

        # (补充1.)距离 & 距离相关的时间
        # 235(加了距离后的ss样本数
        if (lac.run(w1)[1][0] == lac.run(w2)[1][0]) or lac.run(w1)[1][0] == 'LOC' or lac.run(w2)[1][0] == 'LOC':
            dist_words = ['距离', '多远', '公里', '多少米', '多少千米', '几米']
            for k in dist_words:
                if '时间表' in q1 and '时间表' in q2:
                    return False
                if k in q1 and k in q2:
                    # if k == '时间' and ((k in w1 and k not in w2) or (k in w2 and k not in w1)):
                    #     # 时间不允许单独出现在某一交换词内: ct做一次多长时间   多长时间做一次ct
                    #     return False
                    return True

        # (补充2.) 如:变黑变浓/变浓变黑
        # 241
        res_exchange2 = check_exchange_replace(q1, q2, forward=True, by_lac=False)  # 向前扫描
        if res_exchange2[0][1] + 1 == res_exchange2[1][0] and \
                q1[res_exchange2[0][0]] == q1[res_exchange2[1][0]]:  # 两个交换的单词紧邻 & 它们首字符相同
            return True

        # (补充4.) 怎么和其他东西交换
        if "怎么" == w1 or "怎么" == w2:
            tmp_q1 = q1.replace(w1, '').replace(w2, '')
            tmp_q2 = q2.replace(w1, '').replace(w2, '')
            if tmp_q1 == tmp_q2:  # 除去交换词后剩余部分相同
                return True
    return False


# ======================= Asymmetry =======================
def check_Asymmetry(q1, q2):
    ignore_verbs = ["加", "乘", "乘以"]

    if len(q1) == len(q2):
        verb_idx_1 = -1
        res1 = []
        for idx, (w, f) in enumerate(pseg.cut(q1)):
            res1.append((w, f))
            if f == 'v' and w not in ignore_verbs:
                verb_idx_1 = idx

        if 0 < verb_idx_1 < len(res1) - 1:
            verb_idx_2 = -1
            res2 = []
            for idx, (w, f) in enumerate(pseg.cut(q2)):
                res2.append((w, f))
                if f == 'v' and w not in ignore_verbs:
                    verb_idx_2 = idx
            if 0 < verb_idx_2 < len(res2) - 1:
                if res1[verb_idx_1 - 1][0] == res2[verb_idx_2 + 1][0] and res1[verb_idx_1 + 1][0] == \
                        res2[verb_idx_2 - 1][0]:
                    return True

    return False


def check_Asymmetry1122(q1, q2):
    """ 判断是否属于：动词两侧的单词交换的情况 """

    ignore_verbs = ["加", "乘", "乘以"]
    allow_pos = [
        'v',  # 动词
        'p',  # 介词
        'r',  # 代词
        'c',  # 连词
        'u',  # 助词
    ]
    negation_word_pairs = [  # 否定词对
        ('有', '不'),
        ('只', '不'),
        ('得', '失'),
        ('想', '没'),
    ]

    if len(q1) == len(q2):
        res_exchange = check_exchange_replace(q1, q2, forward=True, by_lac=True)
        if res_exchange is None:
            return False

        # 交换词
        w1 = q1[res_exchange[0][0]: res_exchange[0][1] + 1]
        w2 = q1[res_exchange[1][0]: res_exchange[1][1] + 1]

        # 11-22 22:48 反义词识别
        if any(
                ((_pair[0] == w1[0] and _pair[1] == w2[0]) or (_pair[1] == w1[0] and _pair[0] == w2[0]))
                for _pair in negation_word_pairs
        ):
            return True

        # 11-21 20:40 补充"你""我"交换
        if (w1 == "你" and w2 == "我") or (w1 == "我" and w2 == "你"):
            return True

        # 保证交换词词性相同 （有效
        f1 = lac.run(w1)[1][0]
        f2 = lac.run(w2)[1][0]
        if f1 != f2 and f1[0] != f2[0]:
            return False

        # 11-21: 判断当两个交换词都是loc的情况
        if f1 == f2 == 'LOC':
            return True

        verb_idx_1 = -1
        res1 = []
        pseg_res = lac_run_to_pseg_cut(q1)
        for idx, (w, f) in enumerate(pseg_res):
            res1.append((w, f))

            # 符合中心词标准
            # 1. 中心词、'是'、'的'
            # 2. 不在忽略词表内
            # 3. 中心词在两个交换词之间
            if (f[0] in allow_pos or w == '是' or w == '的') and \
                    w not in ignore_verbs and \
                    q1.find(w, res_exchange[0][0] + len(w1), res_exchange[1][1] - len(w2) + 1) != -1:
                verb_idx_1 = idx

        if verb_idx_1 != -1:
            pivot_verb = pseg_res[verb_idx_1][0]
            verb_idx_2 = q2.find(pivot_verb, res_exchange[0][0] + len(w2), res_exchange[1][1] - len(w1) + 1)
            if verb_idx_2 != -1:
                return True
    return False


# ======================= NegativeAsymmetry =======================
# NegativeAsymmetry规则是寻找有主语谓语交换，且后面的名词或形容词相反的情况
def check_NegativeAsymmetry(q1, q2):
    if '比' in q1 and '比' in q2:
        pseg_res1 = [[w, f] for w, f in pseg.cut(q1)]
        pseg_res2 = [[w, f] for w, f in pseg.cut(q2)]

        # 同义词替换 - 预处理
        for i in range(len(pseg_res1)):
            ori_word = pseg_res1[i][0]
            if ori_word in similar_words_dict:
                tmp = pseg.lcut(similar_words_dict[ori_word])
                assert len(tmp) == 1
                pseg_res1[i][0] = tmp[0].word
                pseg_res1[i][1] = tmp[0].flag
        for i in range(len(pseg_res2)):
            ori_word = pseg_res2[i][0]
            if ori_word in similar_words_dict:
                tmp = pseg.lcut(similar_words_dict[ori_word])
                assert len(tmp) == 1
                pseg_res2[i][0] = tmp[0].word
                pseg_res2[i][1] = tmp[0].flag

        def get_bi_adj_idx(pseg_res):
            """ 从pos segment结果中获取'比'和adj的位置 """

            bi_idx, adj_idx = -1, -1

            if ['更', 'd'] in pseg_res:  # 优先找"更"后面的形容词
                tmp_idx = pseg_res.index(['更', 'd'])
                for i in range(tmp_idx, len(pseg_res)):
                    w, f = pseg_res[i]
                    if f == 'a' or f == 'm' or f == 'd':  # 找到的第一个形容词
                        adj_idx = i
                        break

            for i in range(len(pseg_res)):
                w, f = pseg_res[i]
                if w == '比' and f == 'p' and bi_idx == -1:
                    bi_idx = i
                if bi_idx != -1 and (f == 'a' or f == 'm' or f == 'd') and adj_idx == -1:  # 找到的第一个形容词
                    adj_idx = i
                if bi_idx != -1 and adj_idx != -1:
                    break
            return bi_idx, adj_idx

        def get_n_idx(pseg_res, bi_idx, adj_idx):
            """ 从pos segment结果中获取两个交换实体n的位置 """

            n1_idx, n2_idx = -1, -1
            if bi_idx == -1 or adj_idx == -1:
                return n1_idx, n2_idx

            for i in range(len(pseg_res)):
                if 0 <= i < bi_idx and 'n' in pseg_res[i][1] and n1_idx == -1:
                    n1_idx = i
                if bi_idx < i < adj_idx and 'n' in pseg_res[i][1] and n2_idx == -1:
                    n2_idx = i
                if n1_idx != -1 and n2_idx != -1:
                    break
            return n1_idx, n2_idx

        bi_idx1, adj_idx1 = get_bi_adj_idx(pseg_res1)
        bi_idx2, adj_idx2 = get_bi_adj_idx(pseg_res2)
        n1_idx1, n2_idx1 = get_n_idx(pseg_res1, bi_idx1, adj_idx1)
        n1_idx2, n2_idx2 = get_n_idx(pseg_res2, bi_idx2, adj_idx2)

        if 0 <= n1_idx1 < bi_idx1 < n2_idx1 < adj_idx1 and \
                0 <= n1_idx2 < bi_idx2 < n2_idx2 < adj_idx2 and \
                pseg_res1[n1_idx1][0] == pseg_res2[n2_idx2][0] and \
                pseg_res1[n2_idx1][0] == pseg_res2[n1_idx2][0] and \
                pseg_res1[adj_idx1][0] != pseg_res2[adj_idx2][0]:
            return True

    return False


def check_NegativeAsymmetry1114(q1, q2):
    if '比' in q1 and '比' in q2:
        res_exchange = check_exchange_replace(q1, q2)
        if res_exchange is None:
            return None

        tmp = lac.run(q1)
        pseg_res1 = [[tmp[0][i], tmp[1][i]] for i in range(len(tmp[0]))]
        tmp = lac.run(q2)
        pseg_res2 = [[tmp[0][i], tmp[1][i]] for i in range(len(tmp[0]))]

        # 同义词替换
        for i in range(len(pseg_res1)):
            ori_word = pseg_res1[i][0]
            if ori_word in similar_words_dict:
                tmp = pseg.lcut(similar_words_dict[ori_word])
                assert len(tmp) == 1
                pseg_res1[i][0] = tmp[0].word
                pseg_res1[i][1] = tmp[0].flag
        for i in range(len(pseg_res2)):
            ori_word = pseg_res2[i][0]
            if ori_word in similar_words_dict:
                tmp = pseg.lcut(similar_words_dict[ori_word])
                assert len(tmp) == 1
                pseg_res2[i][0] = tmp[0].word
                pseg_res2[i][1] = tmp[0].flag

        def get_bi_adj_idx(pseg_res):
            """ 获取"比"和形容词在句中的位置(pseg_idx) """

            bi_idx, adj_idx = -1, -1

            if ['更', 'd'] in pseg_res:  # 优先找"更"后面的形容词
                tmp_idx = pseg_res.index(['更', 'd'])
                for i in range(tmp_idx, len(pseg_res)):
                    w, f = pseg_res[i]
                    if f == 'a':  # 找到的第一个形容词
                        adj_idx = i
                        break

            for i in range(len(pseg_res)):
                w, f = pseg_res[i]
                if w == '比' and f == 'p' and bi_idx == -1:
                    bi_idx = i
                if bi_idx != -1 and (f == 'a') and adj_idx == -1:  # 找到的第一个形容词
                    adj_idx = i
                if bi_idx != -1 and adj_idx != -1:
                    break
            return bi_idx, adj_idx

        bi_idx1, adj_idx1 = get_bi_adj_idx(pseg_res1)
        bi_idx2, adj_idx2 = get_bi_adj_idx(pseg_res2)

        if 0 < bi_idx1 < adj_idx1 and 0 < bi_idx2 < adj_idx2:
            if pseg_res1[adj_idx1][0] == pseg_res2[adj_idx2][0]:  # 交换比较主语，形容词相同 -- label=N
                return "neg"
            else:  # 交换比较主语，形容词不相同 -- label=Y
                return "pos"

    return None


def check_NegativeAsymmetry1118(q1, q2):
    """ 检查中间连接词为“是”且后面的描述词反义的情况 """

    if '是' in q1 and '是' in q2:
        res = check_exchange_replace(q1, q2, forward=True)
        if res is None:
            return False
        shi_idx = q1[res[0][1] + 1: res[1][0]].find("是")
        if shi_idx == -1:
            return False

        idx = res[1][1] + 1
        if q1[idx:] != q2[idx:]:
            return True
    return False


# ======================= Passive =======================
def check_Passive(q1, q2):
    # todo 处理被动语态，被字开头
    return False


def check_exchange_replace(query1, query2, forward=False, by_lac=False):
    """ 检查交叉替换的情况 """
    if len(query1) != len(query2):
        return None

    N = len(query1)
    p1, p2 = 0, 0
    while p1 < N and p2 < N and query1[p1] == query2[p2]:
        p1 += 1
        p2 += 1
    if p1 >= N:
        return None  # 两个query相同

    # 初始化lst
    # lst 存放以query[0]开头的前缀在另一个query中的span位置
    lst1, lst2 = [], []  # item:[begin, end]
    start = 0
    while start < N:
        idx = query2.find(query1[p1], start)
        if idx != -1:
            save_p1 = p1
            save_idx = idx
            while p1 < N and idx < N and query1[p1] == query2[idx]:
                p1 += 1
                idx += 1
            lst1.append([save_idx, idx - 1])
            p1 = save_p1
            start = save_idx + 1
        else:
            break
    start = 0
    while start < N:
        idx = query1.find(query2[p2], start)
        if idx != -1:
            save_p2 = p2
            save_idx = idx
            while p2 < N and idx < N and query2[p2] == query1[idx]:
                p2 += 1
                idx += 1
            lst2.append([save_idx, idx - 1])
            p2 = save_p2
            start = save_idx + 1
        else:
            break

    ans = None
    for item1 in lst1:
        for item2 in lst2:
            len1 = item1[1] - item1[0]
            len2 = item2[1] - item2[0]
            if item1[1] == item2[1]:
                # if item1[1] == item2[1] and \
                #         query1[p1 + len1: item2[0]] == query2[p2 + len2: item1[0]]:  # 结束位置相同 & (交换的)中间部分相同
                if query1[p1 + len1 + 1: item2[0]] != query2[p2 + len2 + 1: item1[0]]:
                    continue
                ans = [[p1, p1 + len1], item2]
                break
        if ans:
            break

    # 向前扫描
    if ans and forward:
        p1 = ans[0][0]
        p2 = ans[1][0]
        while p1 - 1 >= 0 and p2 - 1 >= 0 and p2 - 1 > ans[0][1] and query1[p1 - 1] == query2[p2 - 1]:
            p1 -= 1
            p2 -= 1
        if by_lac:
            w1 = query1[p1: ans[0][1] + 1]
            w2 = query1[p2: ans[1][1] + 1]
            lac_res1 = lac.run(w1)
            lac_res2 = lac.run(w2)

            ans[0][0] = min(ans[0][0], ans[0][1] + 1 - len(lac_res1[0][-1]))
            ans[1][0] = min(ans[1][0], ans[1][1] + 1 - len(lac_res2[0][-1]))
            # ans[0][0] = ans[0][1] + 1 - len(lac_res1[0][-1])
            # ans[1][0] = ans[1][1] + 1 - len(lac_res2[0][-1])
        else:
            ans[0][0] = p1
            ans[1][0] = p2
    return ans


def lac_run_to_pseg_cut(text):
    """ lac run 结果转成 pseg cut形式 """
    lac_res = lac.run(text)
    res = [
        (lac_res[0][i], lac_res[1][i])  # word, flag
        for i in range(len(lac_res[0]))
    ]
    return res


def main():
    run_time = get_run_time()
    print("run time:", run_time)

    test_path = args.test_path
    test_data = read_text_pair(test_path, is_test=True)
    print("load test data from: ", test_path)

    test_unify_number_path = args.test_unify_number_path
    test_unify_number = read_text_pair(test_unify_number_path, True)
    print("use unify number data")

    label_Y_idx = set()
    label_N_idx = set()

    # ==================== Negative Asymmetry ====================
    na_neg_data_idx = []
    na_pos_data_idx = []
    for i, d in tqdm(enumerate(test_data), desc="#NegativeAsymmetry#"):
        tmp = check_NegativeAsymmetry(d['query1'], d['query2'])
        tmp2 = check_NegativeAsymmetry1118(d['query1'], d['query2'])
        if tmp or tmp2:
            na_pos_data_idx.append(i)  # ! pos
        else:
            pass
            tmp = check_NegativeAsymmetry1114(d['query1'], d['query2'])
            if tmp == "neg":
                na_neg_data_idx.append(i)  # ! neg
    print("Num Negative Asymmetry items: ", len(na_neg_data_idx) + len(na_pos_data_idx))

    label_Y_idx.update(na_pos_data_idx)
    label_N_idx.update(na_neg_data_idx)
    na_data = [test_data[i] for i in na_pos_data_idx + na_neg_data_idx]
    print("samples Negative Asymmetry:")
    for i in range(min(5, len(na_data))):
        print(na_data[i])

    # ==================== Symmetry ====================
    ss_data_idx2 = []
    for i, d in tqdm(enumerate(test_unify_number), desc="#check_Symmetry1122#"):
        if check_Symmetry1122(d['query1'], d['query2']) and i not in label_N_idx:
            ss_data_idx2.append(i)
    print("Num Symmetry items: ", len(ss_data_idx2))

    label_Y_idx.update(ss_data_idx2)
    ss_data = [test_data[i] for i in ss_data_idx2]
    print("samples Symmetry:")
    for i in range(min(5, len(ss_data))):
        print(ss_data[i])

    # ==================== Asymmetry ====================
    as_data_idx2 = []
    for i, d in tqdm(enumerate(test_unify_number), desc="#check_Asymmetry1122#"):
        if check_Asymmetry1122(d['query1'], d['query2']) and i not in label_Y_idx:
            as_data_idx2.append(i)
    label_N_idx.update(as_data_idx2)
    as_data = [test_data[i] for i in as_data_idx2]
    print("samples Asymmetry:")
    for i in range(min(5, len(as_data))):
        print(as_data[i])

    # ==================== output ====================
    data_y = [test_data[i] for i in label_Y_idx]
    data_n = [test_data[i] for i in label_N_idx]

    raw_result_path = args.raw_result_path
    final_result_path = args.final_result_path

    result_f = []
    with open(raw_result_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f.readlines()):
            label = int(line.strip())
            result_f.append(label)

    #  检查冲突
    assert len(label_Y_idx.intersection(label_N_idx)) == 0

    cnt = 0
    data_to_fix = []
    for f_idx in label_Y_idx:
        if result_f[f_idx] == 0:
            cnt += 1
            data_to_fix.append({'data': test_data[f_idx], 'pred': 1})
        result_f[f_idx] = 1
    for f_idx in label_N_idx:
        if result_f[f_idx] == 1:
            cnt += 1
            data_to_fix.append({'data': test_data[f_idx], 'pred': 0})
        result_f[f_idx] = 0

    print("Num of data need to be fixed: ", cnt)

    with open(final_result_path, 'w', encoding='utf-8') as f:
        for label in result_f:
            f.write(str(label) + '\n')
    # ====================
    flag = 3
    with open(f'data/tmp_data/test_B-syntactic_structure-flag_{run_time}.txt', 'w', encoding='utf-8') as f:
        for i in range(len(test_data)):
            label = 0  # 未改动
            if i in label_Y_idx:
                label = flag
            elif i in label_N_idx:
                label = -flag
            f.write(str(label) + '\n')

    print('finish')


if __name__ == '__main__':
    main()
