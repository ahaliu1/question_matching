# coding: utf-8
import json
import jieba
import jieba.posseg as pseg
import Levenshtein
from LAC import LAC

lac = LAC(mode='lac')


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


def check_Symmetry(q1, q2):
    """ Syntactic Structure - Symmetry """

    if len(q1) == len(q2):
        for k in ['和', '与', '又', '还是'] + ["乘以", "加", "乘"]:
            # for k in ['和', '与', '又', '还是']:  # 77
            # for k in ['和', '与', '又']: # 67
            if k in q1 and k in q2:
                res1 = [(w, f) for w, f in pseg.cut(q1)]

                k_idx_1 = find_k_idx(res1, k)
                if k_idx_1 != -1:
                    res2 = [(w, f) for w, f in pseg.cut(q2)]
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
                    w2 = q2[res_exchange[1][0]: res_exchange[1][1] + 1]
                    if w1.isdigit() and w2.isdigit():  # 若交换的两个都是数字,则需要用分词再检查一次(分词工具对于数字边界识别得更好)
                        return check_Symmetry(q1, q2)
                    return True

        # (补充1.)距离 & 距离相关的时间
        # 235(加了距离后的ss样本数
        w1 = q1[res_exchange[0][0]: res_exchange[0][1] + 1]
        w2 = q1[res_exchange[1][0]: res_exchange[1][1] + 1]
        if lac.run(w1)[1][0] == 'LOC' and lac.run(w2)[1][0] == 'LOC':  # 两个都是地名
            for k in ['距离', '多远', '公里', '米', '时间']:
                if k in q1 and k in q2:
                    return True
        # (补充2.) 如:变黑变浓/变浓变黑
        # 241
        res_exchange = check_exchange_replace(q1, q2, forward=True)  # 向前扫描
        if res_exchange[0][1] + 1 == res_exchange[1][0] and \
                q1[res_exchange[0][0]] == q1[res_exchange[1][0]]:  # 两个交换的单词紧邻 & 它们首字符相同
            return True

    return False


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


# ps: 标记了1111的函数使用11月11日新增的check_exchange_replace函数(帮助确定交换实体的位置)
def check_Asymmetry1111(q1, q2):
    """ 通过词性标注工具,将源文本分词,然后获得交换词 """

    ignore_verbs = ["加", "乘", "乘以"]

    if len(q1) == len(q2):
        res_exchange = check_exchange_replace(q1, q2)
        if res_exchange is None:
            return False

        w1 = q1[res_exchange[0][0]: res_exchange[0][1] + 1]
        w2 = q1[res_exchange[1][0]: res_exchange[1][1] + 1]

        verb_idx_1 = -1
        res1 = []
        pseg_res = lac.run(q1)
        for idx, (w, f) in enumerate(zip(pseg_res[0], pseg_res[1])):
            res1.append((w, f))
            if f == 'v' and w not in ignore_verbs and \
                    q1.find(w, res_exchange[0][0] + len(w1), res_exchange[1][1] - len(w2) + 1) != -1:
                verb_idx_1 = idx

        if verb_idx_1 != -1:
            verb_idx_2 = -1
            res2 = []
            pseg_res2 = lac.run(q2)
            for idx, (w, f) in enumerate(zip(pseg_res2[0], pseg_res2[1])):
                res2.append((w, f))
                if f == 'v' and w not in ignore_verbs and \
                        q2.find(w, res_exchange[0][0] + len(w2), res_exchange[1][1] - len(w1) + 1) != -1:
                    verb_idx_2 = idx
            if verb_idx_2 != -1:
                return True
    return False


def check_exchange_replace(query1, query2, forward=False):
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
        while p1 >= 0 and p2 >= 0 and p2 > ans[0][1] and query1[p1] == query2[p2]:
            p1 -= 1
            p2 -= 1
        ans[0][0] = p1 + 1
        ans[1][0] = p2 + 1
    return ans


if __name__ == '__main__':
    """
        后处理说明: 
        由于是比较强的规则,目前把syntactic_structure.py处理放在后处理的最后一步. 
        顺序:模型输出 -> 1. pinyin+模糊拼音+多音字 -> 2. 数字识别 -> 3. Syntactic Structure
        
    """

    # ====================
    test_path = "data/raw_data/test_A.tsv"
    test_data = read_text_pair(test_path, is_test=True)

    # 对test文件进行预处理: 将中文数字转成阿拉伯数字,便于分词
    test_unify_number_path = "data/tmp_data/test_unify_number.tsv"
    test_unify_number = read_text_pair(test_unify_number_path, True)

    label_Y_idx = []  # 标签为1的数据idx
    label_N_idx = []  # 标签为0的数据idx

    # --Symmetry
    # ss_data_idx = []
    # for i, d in enumerate(test_unify_number):
    #     if check_Symmetry(d['query1'], d['query2']):
    #         ss_data_idx.append(i)
    # print("Num Symmetry items: ", len(ss_data_idx))

    # 11.11
    ss_data_idx2 = []
    for i, d in enumerate(test_unify_number):
        if check_Symmetry1111(d['query1'], d['query2']):
            ss_data_idx2.append(i)
    print("Num Symmetry items: ", len(ss_data_idx2))

    label_Y_idx.extend(ss_data_idx2)
    ss_data = [test_data[i] for i in ss_data_idx2]
    print("samples Symmetry:")
    for i in range(min(5, len(ss_data))):
        print(ss_data[i])

    # --Asymmetry
    # as_data_idx = []
    # for i, d in enumerate(test_unify_number):
    #     if check_Asymmetry(d['query1'], d['query2']):
    #         as_data_idx.append(i)
    # print("Num Asymmetry items: ", len(as_data_idx))

    # --Asymmetry
    as_data_idx2 = []
    for i, d in enumerate(test_unify_number):
        if check_Asymmetry1111(d['query1'], d['query2']) and i not in label_Y_idx:  # 简单检查:有交换就属于N
            as_data_idx2.append(i)
    label_N_idx.extend(as_data_idx2)
    as_data = [test_data[i] for i in as_data_idx2]
    print("samples Asymmetry:")
    for i in range(min(5, len(as_data))):
        print(as_data[i])

    # output
    raw_result_path = "data/results/predict_result_0.9161848905525323_fuzzy-pinyin-heteronym_postop-num999.csv"
    final_result_path = "data/results/predict_result_0.91618_fuzzy-pinyin-heteronym_postop-num999_ss1111.csv"

    result_f = []
    with open(raw_result_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f.readlines()):
            label = int(line.strip())
            result_f.append(label)

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
        for label in result_f:
            f.write(str(label) + '\n')
    # ====================

    print('finish')
