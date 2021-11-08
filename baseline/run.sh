###
 # @Descripttion: 
 # @version: 
 # @Author: Tingyu Liu
 # @Date: 2021-09-23 09:47:03
 # @LastEditors: Tingyu Liu
 # @LastEditTime: 2021-10-15 20:19:02
### 
# $unset CUDA_VISIBLE_DEVICES
python -u -m paddle.distributed.launch --gpus "1" train.py \
       --train_set ../data/train_reverse.txt \
       --dev_set ../data/dev.txt \
       --device gpu \
       --eval_step 100 \
       --save_dir ./checkpoints \
       --train_batch_size 128 \
       --learning_rate 2E-5 \
       --rdrop_coef 0.0 \
       --epochs 8
