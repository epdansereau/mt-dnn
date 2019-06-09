#!/bin/bash
if [[ $# -ne 3 ]]; then
  echo "train.sh <batch_size> <gpu> <bert_path>"
  exit 1
fi
prefix="mt-dnn-sst"
BATCH_SIZE=$1
gpu=$2
echo "export CUDA_VISIBLE_DEVICES=${gpu}"
export CUDA_VISIBLE_DEVICES=${gpu}
tstr=$(date +"%FT%H%M")

train_datasets="sst"
test_datasets="sst"
MODEL_ROOT="checkpoints"
BERT_PATH=$3
DATA_DIR="data/mt_dnn"

answer_opt=0
optim="adamax"
grad_clipping=0
global_grad_clipping=1
inference=1

model_dir="checkpoints/${prefix}_${optim}_answer_opt${answer_opt}_gc${grad_clipping}_ggc${global_grad_clipping}_${tstr}"
log_file="${model_dir}/log.log"
python train.py --data_dir ${DATA_DIR} --init_checkpoint ${BERT_PATH} --batch_size ${BATCH_SIZE} --output_dir ${model_dir} --log_file ${log_file} --answer_opt ${answer_opt} --optimizer ${optim} --train_datasets ${train_datasets} --test_datasets ${test_datasets} --grad_clipping ${grad_clipping} --global_grad_clipping ${global_grad_clipping} --epochs 1 --inference ${inference}