#!/usr/bin/env bash

# basic
exp_name=changer
exp_number=1
seed_determinism=2147483647
fold=0

# data
num_class=3
datadir=/workspace/maicon_qualifiers/temp/data/01_data
dataset_mode=maicon_patch_v0
patch_size=256
patch_overlap=0

# model
model=cd_base
loss=bce
metric=f1_iou

# training
batch_size=16
batch_size_inference=1
max_epochs=100
lr=0.001
lr_policy=linear
optimizer=adamw

# trainer
callbacks=lr_ckpt_metricvalid
check_val_every_n_epoch=1
checkpoint_every_n_epochs=1
checkpoint_filename="epoch={epoch:05d}_val_mIoU={metric/val_mIoU:.4f}_val_loss={loss/val_loss:.4f}"
checkpoint_monitor=metric/val_mIoU
checkpoint_monitor_mode=max
loggers=wandb
wandb_project=maicon

# network
net_module=build_opencd
net_config=config/opencd/changer/changer_s101_512x512_40k_levircd_ch3.py
load_pretrained_network=temp/pretrained/ChangerEx_s101-512x512_40k_levircd_weight.pt

# RUN
python train.py --gpu_ids 0 --train_only --seed_determinism ${seed_determinism} --exp_name ${exp_name} --exp_number ${exp_number} --num_class ${num_class} --datadir ${datadir} --dataset_mode ${dataset_mode} --patch_size ${patch_size} --patch_overlap ${patch_overlap} --model ${model} --loss ${loss} --metric ${metric} --batch_size ${batch_size} --batch_size_inference ${batch_size_inference} --max_epochs ${max_epochs} --lr ${lr} --lr_policy ${lr_policy} --optimizer ${optimizer} --callbacks ${callbacks} --check_val_every_n_epoch ${check_val_every_n_epoch} --checkpoint_every_n_epochs ${checkpoint_every_n_epochs} --checkpoint_filename ${checkpoint_filename} --checkpoint_monitor ${checkpoint_monitor} --checkpoint_monitor_mode ${checkpoint_monitor_mode} --loggers ${loggers} --wandb_project ${wandb_project} --net_module ${net_module} --net_config ${net_config} --load_pretrained_network ${load_pretrained_network}
