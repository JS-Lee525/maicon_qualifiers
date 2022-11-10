#!/usr/bin/env bash

# basic
run_base_dir=/content/drive/MyDrive/maicon_qualifiers/result/bit_transformer001
phase=test
load_pretrained_model="./runs/exp/00001/checkpoint/epoch=00049_val_loss=0.2245.ckpt"

# data
num_class=3
datadir=/workspace/maicon_qualifiers/temp/data/01_data
dataset_mode=maicon_patch_v0
patch_size=256
batch_size=24
batch_size_inference=1

# model
## metric=f1_iou

# trainer
callbacks=result

# RUN
python test.py --gpu_ids 0 --run_base_dir ${run_base_dir} --phase ${phase} --load_pretrained_model ${load_pretrained_model} --num_class ${num_class} --datadir ${datadir} --dataset_mode ${dataset_mode} --patch_size ${patch_size} --batch_size ${batch_size} --batch_size_inference ${batch_size_inference} --callbacks ${callbacks} # --metric ${metric} --loggers ${loggers} --wandb_project ${wandb_project}
