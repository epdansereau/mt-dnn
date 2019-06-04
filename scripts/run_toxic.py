from datetime import datetime
import os

prefix="mt-dnn-toxic"
BATCH_SIZE=4
gpu=0
print('CUDA_VISIBLE_DEVICES=',gpu)
CUDA_VISIBLE_DEVICES=gpu
tstr= datetime.now().strftime('%Y%m%d%H%M%S')

train_datasets="toxic"
test_datasets="toxic"
MODEL_ROOT="checkpoints"
BERT_PATH="mt_dnn_models/mt_dnn_large.pt"
DATA_DIR="data/mt_dnn"

answer_opt=0
optim="adamax"
grad_clipping=0
global_grad_clipping=1

model_dir=os.path.join("checkpoints",prefix+tstr)
log_file=os.path.join(model_dir,"log.log")
python train.py --data_dir ${DATA_DIR} --init_checkpoint ${BERT_PATH} --batch_size ${BATCH_SIZE} --output_dir ${model_dir} --log_file ${log_file} --answer_opt ${answer_opt} --optimizer ${optim} --train_datasets ${train_datasets} --test_datasets ${test_datasets} --grad_clipping ${grad_clipping} --global_grad_clipping ${global_grad_clipping}
