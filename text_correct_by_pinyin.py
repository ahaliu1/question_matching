"""
jx part: 1-基于拼音匹配的规则后处理
这里实现了:
    (1)多音字匹配 - heteronym_match, 基于pypinyin;
    (2)模糊拼音匹配 - fuzzy_match, 基于[chinese_fuzzy_match](https://github.com/ibbd-dev/python3-tools/blob/master/ibbd_python3_tools/chinese_fuzzy_match.py);
    (3)普通拼音匹配 - pinyin_match, 基于pypinyin;
"""
import argparse
from tqdm import tqdm

from pypinyin import pinyin, Style

from chinese_fuzzy_match import *
from src.utils import get_run_time

from string import punctuation as en_punc
from zhon.hanzi import punctuation as zh_punc

parser = argparse.ArgumentParser()
parser.add_argument("--test_path", type=str, required=True, help="test路径")
parser.add_argument("--raw_result_path", type=str, required=True, help="上一步处理后的预测结果路径")
parser.add_argument("--final_result_path", type=str, required=True, help="本次处理预测结果输出路径")
parser.add_argument("--do_fuzzy", action='store_true', help="模糊拼音匹配")
parser.add_argument("--do_heteronym", action='store_true', help="多音字匹配")
parser.add_argument("--do_pinyin", action='store_true', help="普通拼音匹配")
args = parser.parse_args()


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


def heteronym_match(query1, query2):
    """ 多音字匹配 """
    query1_tokens = [c for c in query1]
    query2_tokens = [c for c in query2]

    pinyin_res1 = [pinyin(t, style=Style.NORMAL, heteronym=True)[0] for t in query1_tokens]
    pinyin_res2 = [pinyin(t, style=Style.NORMAL, heteronym=True)[0] for t in query2_tokens]

    if len(pinyin_res1) == len(pinyin_res2):
        for i, (item1, item2) in enumerate(zip(pinyin_res1, pinyin_res2)):
            if query1[i] != query2[i]:  # 当字不一致时才检查是否时多音字情况
                inter = set(item1).intersection(set(item2))
                if len(inter) == 0:
                    return False
        return True
    return False


def pinyin_match(query1, query2):
    """ 普通pinyin纠正 """
    if len(query1) != len(query2):
        return False

    q1_py = pinyin(query1, style=Style.NORMAL)
    q2_py = pinyin(query2, style=Style.NORMAL)

    if q1_py == q2_py:
        return True
    return False


def fuzzy_match(query1, query2):
    """ 模糊pinyin匹配：前后鼻音、平翘舌 """
    if len(query1) != len(query2):
        return False

    res = chinese_fuzzy_match(query1, query2, use_fuzzy=True)
    if res['match_type'] != 'not_match':
        return True
    return False


def main():
    run_time = get_run_time()
    print("run time:", run_time)

    test_path = args.test_path
    test_data = read_text_pair(test_path, is_test=True)
    print("load test data from: ", test_path)

    label_Y_idx = set()
    label_N_idx = set()

    # ================= 模糊拼音 =================
    if args.do_fuzzy:
        fuzzy_py_idx = []
        for idx, d in tqdm(enumerate(test_data), desc="#fuzzy_match#"):
            query1, query2 = d['query1'], d['query2']
            if len(query1) != len(query2):
                continue
            # 去除标点
            for i in en_punc + zh_punc:
                query1 = query1.replace(i, '')
                query2 = query2.replace(i, '')
            if fuzzy_match(query1, query2):
                fuzzy_py_idx.append(idx)
        label_Y_idx.update(fuzzy_py_idx)
        print("Num of fuzzy pinyin: ", len(fuzzy_py_idx))

    # ================= 多音字 =================
    if args.do_heteronym:
        heteronym_idx = []
        for idx, d in tqdm(enumerate(test_data), desc="#heteronym_match#"):
            query1, query2 = d['query1'], d['query2']
            if len(query1) != len(query2):
                continue
            if heteronym_match(query1, query2):
                heteronym_idx.append(idx)
        label_Y_idx.update(heteronym_idx)
        print("Num of heteronym", len(heteronym_idx))

    if args.do_pinyin:
        # ================= 普通pinyin =================
        pinyin_idx = []
        for idx, d in tqdm(enumerate(test_data), desc="#pinyin_match#"):
            query1, query2 = d['query1'], d['query2']
            if len(query1) != len(query2):
                continue

            if pinyin_match(query1, query2):
                pinyin_idx.append(idx)
        label_Y_idx.update(pinyin_idx)
        print("Num of pinyin", len(pinyin_idx))

    # ================= output =================
    raw_result_path = args.raw_result_path
    final_result_path = args.final_result_path

    # fix result
    result_f = []
    with open(raw_result_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f.readlines()):
            label = int(line.strip())
            result_f.append(label)

    cnt = 0  # 需要修改的标签数
    data_Y = []
    for idx in label_Y_idx:
        if result_f[idx] == 0:
            data_Y.append(test_data[idx])
            cnt += 1
        result_f[idx] = 1
    print("Num of data need to be fixed: ", cnt)

    with open(final_result_path, 'w', encoding='utf-8') as f:
        for i, label in enumerate(result_f):
            f.write(str(label) + '\n')

    flag = 1
    with open(f'data/tmp_data/test_B-text_corrector_pinyin_idx-flag_{run_time}.txt', 'w', encoding='utf-8') as f:
        for i in range(len(test_data)):
            label = 0  # 未改动
            if i in label_Y_idx:
                label = flag
            elif i in label_N_idx:
                label = -flag
            f.write(str(label) + '\n')


if __name__ == '__main__':
    main()
