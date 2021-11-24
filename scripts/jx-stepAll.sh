#!/bin/bash

TEST_PATH="data/raw_data/test_B_1118.tsv"  # 原始的testB文件
STEP1_IN="data/results/torch_B_ltypre-2.csv"  # 模型预测+lty预处理结果

STEP1_OUT="data/results/torch_B_ltypre-2_1124-2145_fuzzy-heteronym.csv"
STEP2_OUT="data/results/torch_B_ltypre-2_1124-2145_fuzzy-heteronym_num999.csv"
STEP3_OUT="data/results/torch_B_ltypre-2_1124-2145_fuzzy-heteronym_num999_na-ss1122-as1122.csv"
STEP4_OUT="data/results/result.csv"

echo "#jx part start#"

echo "#step1 start#"
python text_correct_by_pinyin.py \
  --test_path=$TEST_PATH \
  --raw_result_path=$STEP1_IN \
  --final_result_path=$STEP1_OUT \
  --do_fuzzy \
  --do_heteronym \
  "$@"

echo "#step2 start#"
python jx_postpreprocess.py \
  --test_path=$TEST_PATH \
  --raw_result_path=$STEP1_OUT \
  --final_result_path=$STEP2_OUT \
  "$@"

echo "#step3 start#"
python syntactic_structure.py \
  --test_path=$TEST_PATH \
  --raw_result_path=$STEP2_OUT \
  --final_result_path=$STEP3_OUT \
  "$@"

echo "#step4 start#"
python text_correct_by_pinyin.py \
  --test_path=$TEST_PATH \
  --raw_result_path=$STEP3_OUT \
  --final_result_path=$STEP4_OUT \
  --do_pinyin \
  "$@"

echo "final submit file path: "$STEP4_OUT
echo "#jx part finished#"
