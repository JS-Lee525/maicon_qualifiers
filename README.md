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

## Using different networks
```bash
--net_module $MODULE_NAME --net_config $CONFIG_PATH --load_pretrained_network $WEIGHT_PATH
```
- `net_module`: 모듈을 직접 입력 (git clone한 경우 경로를 환경변수에 추가해주어야 모듈로 검색됨) or 아무 입력 없을 경우 gencd.models.networks을 사용.
- `net_config`: network class 이름과 나머지 config을 담고 있는 YAML 파일이어야함. dictionary 형태이므로 network는 key arguments들로만 정의될 수 있어야함. e.g. `config/snunet_baseline.yaml` 참고
```
net_config.yaml
{
    net_name: SNUNet_ECAM
    config:
        in_ch: 3
        out_ch: 2
}
```
- `load_pretrained_network`: pretrained weight 파일 (.pt) 경로

## Dataset
`gencd.data.s2looking_dataset`과 `gencd.data.s2_v0_dataset` 참고. 데이터셋이 바뀌거나 다른 transform 방법으로 customize할 경우에 gencd.data 아래에 새로운 class를 정의해서 사용하면 됨.
### Args
- `dataset_mode`: 데이터셋 class "XxxYyyDataset"이 xxx_yyy_dataset.py에 정의되어야함 (파일명에서 언더바를 parsing한 단어로 찾음)
- `datadir`: 데이터셋 base 경로. 데이터셋 구조에 따라 Class 내 prepare_data 메소드로 하위 경로 등 개별로 정의함.
### Methods
- `prepare_data`: datadir과 phase를 이용해 데이터 경로들을 읽음. Case의 고유번호인 self.keys를 정의해야함.
- `read_data`: index를 input으로 받아 `prepare_data`에서 읽은 경로들로부터 image, image2, mask, metadata를 받아 return. 이 때, image, image2 shape은 HWC, mask는 HW1 형태의 array로. mask가 없는 test dataset의 경우 None을 return 하도록.
- `prepare_transforms`: phase=='train'의 경우 crop 및 augmentation, 그 외의 경우 원본 이미지를 그대로 ToTensor만 취함.

## Logging
현재 csv, tensorboard, wandb 모두 있으나 wandb로 통일하면 좋을 것 같음. `--loggers wandb`으로 사용 가능.
- `--wandb_project`: wandb상 프로젝트명. 
- `--wandb_name`: 프로젝트 아래에 표기되는 run 이름. 입력 없을 경우, training folder이름으로 들어감.
