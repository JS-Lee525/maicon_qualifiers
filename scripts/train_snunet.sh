#!/usr/bin/env bash

# basic
exp_name=exp
exp_number=1

# data
num_class=3
datadir=./temp/data/01_data
dataset_mode=maicon_image_v0
patch_size=512
patch_resize_factor=2
patch_overlap=0
split=./temp/maicon-split.pkl

# model
model=cd_base
loss=bce
metric=f1_iou

# training
batch_size=24
batch_size_inference=1
max_epochs=100
lr=0.001
lr_policy=linear
optimizer=adamw
fold=0

# trainer
callbacks=lr_ckpt_metricvalid
check_val_every_n_epoch=1
checkpoint_every_n_epochs=1
checkpoint_filename="epoch={epoch:05d}_val_mIoU={metric/val_mIoU:.4f}_val_loss={loss/val_loss:.4f}"
checkpoint_monitor=metric/val_mIoU
checkpoint_monitor_mode=max
loggers=tb
wandb_project=maicon

# network
net_module=gencd.models.networks
net_config=config/snunet_ch3.yaml
load_pretrained_network=temp/pretrained/snunet-32-weight.pt


#Valid TEST
# max_data=100


# RUN
python train.py \
--seed_determinism 2147483647 --gpu_ids 0 \
--exp_name ${exp_name} --exp_number ${exp_number} \
--num_class ${num_class} --datadir ${datadir} --dataset_mode ${dataset_mode} --patch_size ${patch_size} --patch_overlap ${patch_overlap} --dataset_split ${split} --patch_resize_factor ${patch_resize_factor} \
--model ${model} --loss ${loss} --metric ${metric} \
--batch_size ${batch_size} --batch_size_inference ${batch_size_inference} --max_epochs ${max_epochs} --lr ${lr} --lr_policy ${lr_policy} --optimizer ${optimizer} --fold ${fold} \
--callbacks ${callbacks} --check_val_every_n_epoch ${check_val_every_n_epoch} --checkpoint_every_n_epochs ${checkpoint_every_n_epochs} --checkpoint_filename ${checkpoint_filename} --checkpoint_monitor ${checkpoint_monitor} --checkpoint_monitor_mode ${checkpoint_monitor_mode} --loggers ${loggers} --wandb_project ${wandb_project} --checkpoint_nooverwrite --train_only \
--net_module ${net_module} --net_config ${net_config} --load_pretrained_network ${load_pretrained_network}