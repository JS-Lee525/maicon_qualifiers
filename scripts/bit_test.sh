#!/usr/bin/env bash

# basic
run_base_dir=/content/drive/MyDrive/maicon_qualifiers/result/bit_transformer001_e039
phase=test
load_pretrained_model="/content/drive/MyDrive/maicon_qualifiers/runs/exp/00001/fold0/checkpoint/bit_epoch=00039_val_mIoU=0.5916_val_loss=0.0415.ckpt"

# data
num_class=3
datadir=/content/drive/MyDrive/maicon
dataset_mode=maicon_image_v0
patch_size=512
batch_size=24
batch_size_inference=1
patch_resize_factor=2
# model
## metric=f1_iou

# trainer
callbacks=result

# RUN
python test.py --gpu_ids 0 --run_base_dir ${run_base_dir} --patch_resize_factor ${patch_resize_factor} --phase ${phase} --load_pretrained_model ${load_pretrained_model} --num_class ${num_class} --datadir ${datadir} --dataset_mode ${dataset_mode} --patch_size ${patch_size} --batch_size ${batch_size} --batch_size_inference ${batch_size_inference} --callbacks ${callbacks} # --metric ${metric} --loggers ${loggers} --wandb_project ${wandb_project}
