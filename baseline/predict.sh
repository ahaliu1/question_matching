python -u \
    predict.py \
    --device gpu \
    --params_path "./7_checkpoints/model_20100/model_state.pdparams" \
    --batch_size 128 \
    --input_file "../data/test_A_reverse.tsv" \
    --result_file "13_predict_result"