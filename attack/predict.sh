# python -u \
#     predict.py \
#     --device gpu \
#     --params_path "./checkpoints/model_15300/model_state.pdparams" \
#     --batch_size 128 \
#     --input_file "../data/test_A.tsv" \
#     --result_file "14_predict_result"

/home/lty/code/question_matching/attack/checkpoints/model_36300/model_state.pdparams

python -u \
    predict.py \
    --device gpu \
    --params_path "./checkpoints/model_36300/model_state.pdparams" \
    --batch_size 128 \
    --input_file "../data/test_A.tsv" \
    --result_file "15_predict_result"