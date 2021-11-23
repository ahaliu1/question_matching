# 2021千言-问题匹配鲁棒性评测B榜第六名方案
任务地址：https://www.datafountain.cn/competitions/516
## baseline/
比赛官方提供的基于paddlepaddle的ERNIE baseline。

## attach/
在paddle的baseline上使用FGM对抗攻击embedding以增强鲁棒性。

## share_data/
比赛中一些用到的数据
```
.
├── afqmc_public
│   ├── dev.json
│   ├── test.json
│   └── train.json
├── ATEC
│   ├── atec_nlp_sim_train_add.csv
│   └── atec_nlp_sim_train.csv
├── aug
│   ├── aug_loc_37624.tsv
│   └── aug_similar_110389.tsv
├── CCKS_2018_3
│   ├── task3_dev.txt
│   ├── task3_sample_submission.csv
│   ├── task3_test_data_expand
│   │   ├── Readme
│   │   └── test_with_id.txt
│   └── task3_train.txt
├── train_dataset
│   ├── BQ
│   │   ├── dev
│   │   ├── test
│   │   └── train
│   ├── LCQMC
│   │   ├── dev
│   │   ├── test
│   │   └── train
│   └── OPPO
│       ├── dev
│       └── train
├── train_en_ch.txt
└── train_en.txt
```

训练集：BQ+LCQMC+OPPO+CCKS_2018_3+ATEC+afqmc_public

train_en.txt;train_en_ch.txt将训练集query1翻译成英文后再翻译回中文所做的数据增强，在misspelling单项上有10%的提升。

aug/ 地名增强数据及基于大词林的近义词增强。

## torch_baseline
大佬提供的pytorch版本RENIE模型，相比于paddle模型做了一些改进，最终提交结果是基于这个模型的输出

https://zhuanlan.zhihu.com/p/427995338

## postprocess.py

大量的后处理规则
* 模糊拼音纠错
* 语法纠错
* 词法纠错
...

## preprocess.py

预处理及数据增强代码
