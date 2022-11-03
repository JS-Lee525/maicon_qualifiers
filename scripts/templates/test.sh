#!/usr/bin/env bash

# basic
run_base_dir=./temp/result/exp/00001/result_test
phase=test
load_pretrained_model="./runs/exp/00001/checkpoint/epoch=00049_val_loss=0.2245.ckpt"

# data
num_class=2
datadir=/home/jwchoi/Downloads/LEVIR-CD-256
dataset_mode=changeformer
patch_size=256
batch_size=16
batch_size_inference=2

# model
metric=f1_iou

# trainer
callbacks=result
loggers=csv
wandb_project=maicon

# RUN
python test.py --gpu_ids 0 --run_base_dir ${run_base_dir} --phase ${phase} --load_pretrained_model ${load_pretrained_model} --num_class ${num_class} --datadir ${datadir} --dataset_mode ${dataset_mode} --patch_size ${patch_size} --batch_size ${batch_size} --batch_size_inference ${batch_size_inference} --metric ${metric} --callbacks ${callbacks} --loggers ${loggers} --wandb_project ${wandb_project}
