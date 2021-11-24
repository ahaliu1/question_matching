"""
jx part: 2-针对Lexical Semantics的规则
这里实现了:
    (1)数字识别;
"""
import argparse
from tqdm import tqdm

import pandas as pd
import numpy as np

import Levenshtein
from LAC import LAC
import jieba.analyse

# from src.utils import get_run_time

parser = argparse.ArgumentParser()
parser.add_argument("--test_path", type=str, required=True, help="test路径")
parser.add_argument("--raw_result_path", type=str, required=True, help="上一步处理后的预测结果路径")
parser.add_argument("--final_result_path", type=str, required=True, help="本次处理预测结果输出路径")
args = parser.parse_args()

lac = LAC(mode='rank')


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


# ==================== 数字识别 ====================
# idea1
def check_number(query1, query2, max_op_times=2):
    """ 检查query1和query2是否是由修改数字得到的 —— 标签为N """
    editops = Levenshtein.editops(query1, query2)
    cnt = 0
    if 0 < len(editops) <= max_op_times:  # 只检测修改长度<=max_op_times的问题对
        for op_seq in editops:
            op, pos1, pos2 = op_seq
            if op == 'delete':
                if query1[pos1].isdigit():
                    cnt += 1
            elif op == 'insert':
                if query2[pos2].isdigit():
                    cnt += 1
            elif op == 'replace':
                if query1[pos1].isdigit() and query2[pos2].isdigit():
                    cnt += 1
        return cnt == len(editops)
    return False


# idea2
def check_number_by_weight(query1, query2, max_op_times=2):
    """ 检查query1和query2是否是由修改数字得到的 —— 标签为N """

    def is_important_char(query, c_pos):
        """
        基于词权重，判断query中c_pos位置的字符是否是重要字符

        :param query:
        :param c_pos:
        :return:
        """
        res = jieba.analyse.extract_tags(query, topK=10, withWeight=True, allowPOS=())
        for word, weight in res:
            word_idx = query.index(word)
            if word_idx <= c_pos < word_idx + len(word):
                return True
        return False

    editops = Levenshtein.editops(query1, query2)
    cnt_digit = 0
    cnt_important = 0
    if 0 < len(editops) <= max_op_times:  # 只检测修改长度为max_op_times的问题对
        for op_seq in editops:
            op, pos1, pos2 = op_seq
            if op == 'delete':
                if query1[pos1].isdigit():
                    cnt_digit += 1
                if is_important_char(query1, pos1):
                    cnt_important += 1
            elif op == 'insert':
                if query2[pos2].isdigit():
                    cnt_digit += 1
                if is_important_char(query2, pos2):
                    cnt_important += 1
            elif op == 'replace':
                if query1[pos1].isdigit() and query2[pos2].isdigit():
                    cnt_digit += 1
                if is_important_char(query1, pos1) or is_important_char(query2, pos2):
                    cnt_important += 1
        return cnt_digit == len(editops) and cnt_important > 0
    return False


# ==================== 实体替换识别 ====================
def check_entity_replace(query1, query2, max_op_times=2, entity_type='LOC'):
    """ 检查实体替换 """

    def pos_tokens(query):
        tokens = [t for t in query]
        pos_lst = [''] * len(tokens)

        lac_res = lac.run(query)
        offset = 0
        for li in range(len(lac_res[0])):
            word, flag = lac_res[0][li], lac_res[1][li]
            for wi, w in enumerate(word):
                pos_lst[offset + wi] = flag
            offset += len(word)
        return pos_lst

    pos_lst1 = pos_tokens(query1)
    pos_lst2 = pos_tokens(query2)

    editops = Levenshtein.editops(query1, query2)
    cnt = 0
    if 0 < len(editops) <= max_op_times:  # 只检测修改长度<=max_op_times的问题对
        for op_seq in editops:
            op, pos1, pos2 = op_seq
            if op == 'delete':
                if pos_lst1[pos1] == entity_type:
                    cnt += 1
            elif op == 'insert':
                if pos_lst2[pos2] == entity_type:
                    cnt += 1
            elif op == 'replace':
                if pos_lst1[pos1] == entity_type and pos_lst2[pos2] == entity_type:
                    cnt += 1
        return cnt == len(editops)
    return False


# ==================== single change识别 ====================
# 针对 Lexical Semantics - Word & Phrase的insert和replace
def get_single_change(query1, query2):
    """ 检查query1和query2是否只进行了一个下标位置的修改，一个词换成另一个也算 """
    query1 = query1.lower()
    query2 = query2.lower()

    p1 = p2 = 0
    p3 = len(query1) - 1
    p4 = len(query2) - 1

    while p1 < len(query1) and p2 < len(query2) and query1[p1] == query2[p2]:
        p1 += 1
        p2 += 1
    while p3 >= p1 and p4 >= p2 and query1[p3] == query2[p4]:
        p3 -= 1
        p4 -= 1

    # 分词工具帮助确定边界
    def fn(query, pp1, pp2):
        pseg_res = lac.run(query)
        sep_idx = [0]

        idx = 0
        flag1 = flag2 = True
        for item in pseg_res[0]:
            idx += len(item)
            if flag1 and idx > pp1:
                pp1 = sep_idx[-1]
                flag1 = False
            if flag2 and idx > pp2:
                pp2 = idx - 1  # pp2指向实体最后一个位置
                flag2 = False
            sep_idx.append(idx)

        return pp1, pp2

    # 修改边界
    p1, p3 = fn(query1, p1, p3)
    p2, p4 = fn(query2, p2, p4)

    # 这样得到的输出不一定是只有一个实体，所以还需要在外部判断
    # eg.
    # >>> get_single_change('济南到济宁高速费多少','济宁到济南高速费多少')
    # >>> ['济南到济宁', '济宁到济南']
    return [query1[p1: p3 + 1], query2[p2: p4 + 1]]


def check_single_change(query1, query2):
    res = get_single_change(query1, query2)

    pseg_res1 = lac.run(res[0])
    pseg_res2 = lac.run(res[1])
    if len(pseg_res1) > 1 or len(pseg_res2) > 1:
        return None

    # todo: 判断词义相似性

    return res


# ==================== insert 识别 ====================
def check_insert(query1, query2):
    """ 检查<query2>是否由<query1>insert得到，若是则返回insert word"""
    if len(query1) == len(query2):
        return None
    if len(query1) > len(query2):
        query1, query2 = query2, query1
    if query2.startswith(query1):
        _insert_word = query2[len(query1):]
        return _insert_word
    elif query2.endswith(query1):
        _insert_word = query2[: -len(query1)]
        return _insert_word
    return None


def check_insert_adj(query1, query2):
    """ 检查adj插入 """

    def pos_rank_tokens(query):
        tokens = [t for t in query]
        pos_lst = [''] * len(tokens)
        rank_lst = [0] * len(tokens)

        lac_res = lac.run(query)  # !lac mode=rank
        offset = 0
        for li in range(len(lac_res[0])):
            word, flag, rank = lac_res[0][li], lac_res[1][li], lac_res[2][li]
            for wi, w in enumerate(word):
                pos_lst[offset + wi] = flag
                rank_lst[offset + wi] = rank
            offset += len(word)
        return pos_lst, rank_lst

    # 去除标点
    from string import punctuation as en_punc
    from zhon.hanzi import punctuation as zh_punc
    for i in en_punc + zh_punc:
        query1 = query1.replace(i, '')
        query2 = query2.replace(i, '')

    if len(query1) == len(query2):
        return {}

    if len(query1) > len(query2):
        query1, query2 = query2, query1

    insert_words = get_insert_words(query1, query2)
    if len(insert_words.keys()) == 1:
        insert_pos = list(insert_words.keys())[0]
        insert_word = list(insert_words.values())[0]

        ## add 11-17 21:50
        if insert_pos == 0 and '急' == insert_word:
            return False
        if insert_word in ["好", "您好", "你好"]:
            return False

        lac_res = lac.run(insert_word)

        lac_front = lac.run(query1[:insert_pos])
        lac_back = lac.run(query1[insert_pos:])

        query2_pos_lst, query2_rank_lst = pos_rank_tokens(query2)

        # 要求仅为插入一个形容词的情况,且形容词后面是一个名词
        if len(lac_res[0]) == 1 and lac_res[1][0][0] == 'a' and query2_rank_lst[insert_pos] == 3:
            if insert_pos >= len(query1):  # 在query1末尾插入
                return False

            tmp = insert_word + lac_back[0][0]
            tmp_res = lac.run(tmp)
            if lac_back[1][0][0] == 'n' and len(tmp_res[0]) > 1:  # 判断一下这个形容词和紧接着下一个名词能否合并
                return True
    return False


def get_insert_words(query1, query2):
    """ 获取insert词，返回 位置-单词dict """
    if len(query1) > len(query2):
        query1, query2 = query2, query1
    pos_to_word = {}  # pos是query1中的插入位置，word是插入词
    res = Levenshtein.editops(query1, query2)
    for op, pos1, pos2 in res:
        if op == 'insert':
            pos_to_word[pos1] = pos_to_word.get(pos1, "") + query2[pos2]  # 合并相同位置的insert词
        else:
            return {}
    return pos_to_word


def main():
    # run_time = get_run_time()
    # print("run time:", run_time)

    test_path = args.test_path
    test_data = read_text_pair(test_path, is_test=True)
    print("load test data from: ", test_path)

    label_Y_idx = set()
    label_N_idx = set()

    # ========== 数字识别 ============
    for i, d in enumerate(test_data):
        if check_number(d['query1'], d['query2'], 999):
            label_N_idx.add(i)
    # ========== 数字识别 ============

    print("Num of label_Y_idx items: ", len(label_Y_idx))
    print("Num of label_N_idx items: ", len(label_N_idx))

    # ================= output =================
    raw_result_path = args.raw_result_path
    final_result_path = args.final_result_path

    # fix result
    result_f = []
    with open(raw_result_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f.readlines()):
            label = int(line.strip())
            result_f.append(label)

    assert len(label_Y_idx.intersection(label_N_idx)) == 0  # 检查冲突

    cnt = 0
    for f_idx in label_Y_idx:
        if result_f[f_idx] == 0:
            cnt += 1
        result_f[f_idx] = 1
    for f_idx in label_N_idx:
        if result_f[f_idx] == 1:
            cnt += 1
        result_f[f_idx] = 0

    print("Num of data need to be fixed: ", cnt)

    with open(final_result_path, 'w', encoding='utf-8') as f:
        for i, label in enumerate(result_f):
            f.write(str(label) + '\n')
    #
    # flag = 2
    # with open(f'data/tmp_data/test_B-postop_insert_idx-flag_{run_time}.txt', 'w', encoding='utf-8') as f:
    #     for i in range(len(test_data)):
    #         label = 0  # 未改动
    #         if i in label_Y_idx:
    #             label = flag
    #         elif i in label_N_idx:
    #             label = -flag
    #         f.write(str(label) + '\n')


if __name__ == '__main__':
    main()
