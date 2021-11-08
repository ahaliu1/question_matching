###
 # @Descripttion: 
 # @version: 
 # @Author: Tingyu Liu
 # @Date: 2021-10-25 16:41:40
 # @LastEditors: Tingyu Liu
 # @LastEditTime: 2021-10-25 17:12:14
### 
python -u -m paddle.distributed.launch --gpus "2" train.py \
       --train_set ../data/train.txt \
       --dev_set ../data/dev.txt \
       --device gpu \
       --eval_step 100 \
       --save_dir ./checkpoints \
       --train_batch_size 128 \
       --learning_rate 2E-5 \
       --rdrop_coef 0.0 \
       --epochs 8 \
       --fgm_on \
       --adv_eps 1.0