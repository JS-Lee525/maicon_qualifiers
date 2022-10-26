# GenCD
## Quickstart
### Train
```bash
python train.py --exp_name pre --exp_number 1 --gpu_ids 0 --datadir $S2LOOKINGDATASET_PATH --dataset_mode s2_v0 --patch_size 256 --model cd_base --net_config ./config/snunet_baseline.yaml --load_pretrained_network $WEIGHT_PATH/snunet-32-weight.pt --max_epochs 100 --check_val_every_n_epoch 1 --checkpoint_every_n_epochs 1 --batch_size 8 --batch_size_inference 2 --lr 0.001 --lr_policy step --optimizer adam --loss dicefocal --loggers wandb --callbacks lr_ckpt --checkpoint_filename epoch={epoch:05d}_val_loss={loss/val_loss:.4f}
```
### Test
```bash
python test.py --gpu_ids 0 --run_base_dir $SAVE_PATH --phase val --datadir $S2LOOKINGDATASET_PATH --dataset_mode s2_v0 --patch_size 256 --batch_size 8 --batch_size_inference 2 --load_pretrained_model $TRAINED_WEIGHT_PATH --callbacks result
```
