#!/bin/bash
if [[ $# -ne 2 ]]; then
  echo "train.sh <batch_size> <gpu>"
  exit 1
fi
BERT_PATH="mt_dnn_models/mt_dnn_large.pt"
for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99
do
    echo "!!!!BERT_PATH:${BERT_PATH}"
    prefix="mt-dnn-toxic"
    BATCH_SIZE=$1
    gpu=$2
    echo "export CUDA_VISIBLE_DEVICES=${gpu}"
    export CUDA_VISIBLE_DEVICES=${gpu}
    tstr=$(date +"%FT%H%M")

    train_datasets="toxic"
    test_datasets="toxic"
    MODEL_ROOT="checkpoints"
    DATA_DIR="data/mt_dnn"

    answer_opt=0
    optim="adamax"
    grad_clipping=0
    global_grad_clipping=1

    model_dir="checkpoints/${prefix}_${optim}_answer_opt${answer_opt}_gc${grad_clipping}_ggc${global_grad_clipping}_${tstr}"
    log_file="${model_dir}/log.log"

    mv data/mt_dnn/toxic_train${i}.json data/mt_dnn/toxic_train.json
    model_dir="checkpoints/${prefix}_${optim}_answer_opt${answer_opt}_gc${grad_clipping}_ggc${global_grad_clipping}_${tstr}"
    log_file="${model_dir}/log.log"
    python train.py --data_dir ${DATA_DIR} --init_checkpoint ${BERT_PATH} --batch_size ${BATCH_SIZE} --output_dir ${model_dir} --log_file ${log_file} --answer_opt ${answer_opt} --optimizer ${optim} --train_datasets ${train_datasets} --test_datasets ${test_datasets} --grad_clipping ${grad_clipping} --global_grad_clipping ${global_grad_clipping} --epochs 1 --batch_size_eval 4
    mv data/mt_dnn/toxic_train.json data/mt_dnn/toxic_train${i}.json
    python scripts/strip_model.py --checkpoint "${model_dir}/model_0.pt" --fout "checkpoint/model_${i}_stripped.pt"
    BERT_PATH="checkpoint/model_${i}_stripped.pt"
done
