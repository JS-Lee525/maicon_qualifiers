#!/usr/bin/env bash

# basic
exp_name=BIT
exp_number=3

# data
num_class=3
datadir=/content/drive/MyDrive/maicon
dataset_mode=maicon_image_v1
patch_size=512
split=./maicon-split-clean.pkl
patch_resize_factor=2
# model
model=cd_base
loss=bce
metric=f1_iou

# training
batch_size=24
batch_size_inference=1
max_epochs=200
lr=0.001
lr_policy=linear
optimizer=adamw
fold=3

# trainer
callbacks=lr_ckpt_metricvalid
check_val_every_n_epoch=1
checkpoint_every_n_epochs=1
checkpoint_filename="bit_fold2_cosine_epoch={epoch:05d}_val_mIoU={metric/val_mIoU:.4f}_val_loss={loss/val_loss:.4f}"
checkpoint_monitor=metric/val_mIoU
checkpoint_monitor_mode=max
loggers=wandb

# network
net_module=gencd.models.networks
net_config=/content/drive/MyDrive/maicon_qualifiers/config/BIT/bit-vitaev2.yaml
load_pretrained_network=/content/drive/MyDrive/maicon_prep-HoJoon/gencd/models/networks/weights/bit-rsp-vitaev2-s-cdd-weight.pt


#Valid TEST
# max_data=100

patch_overlap=0
# RUN
python train.py \
--seed_determinism 2147483647 --gpu_ids 0 --save_weights_only False\
--exp_name ${exp_name} --exp_number ${exp_number} \
--num_class ${num_class} --datadir ${datadir} --dataset_mode ${dataset_mode} --patch_size ${patch_size} --patch_overlap ${patch_overlap} --dataset_split ${split} \
--model ${model} --loss ${loss} --metric ${metric} \
--batch_size ${batch_size} --batch_size_inference ${batch_size_inference} --max_epochs ${max_epochs} --lr ${lr} --lr_policy ${lr_policy} --optimizer ${optimizer} --fold ${fold} \
--callbacks ${callbacks} --check_val_every_n_epoch ${check_val_every_n_epoch} --checkpoint_every_n_epochs ${checkpoint_every_n_epochs} --checkpoint_filename ${checkpoint_filename} --checkpoint_monitor ${checkpoint_monitor} --checkpoint_monitor_mode ${checkpoint_monitor_mode} --loggers ${loggers} --checkpoint_nooverwrite --train_only \
--net_module ${net_module} --net_config ${net_config} --load_pretrained_network ${load_pretrained_network} --mixed_precision --patch_resize_factor ${patch_resize_factor}