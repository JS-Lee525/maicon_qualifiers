import json
import os
import numpy as np
from PIL import Image
import SimpleITK as sitk
import wandb

import torch
from torchvision.transforms import ToPILImage, ToTensor
from torchvision.utils import make_grid
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

class MyTrainer(pl.Trainer):
    '''Pytorch Lightning Trainer with custom callbacks, logger, args, etc.
    '''
    
    def __init__(self, pl_module: pl.LightningDataModule, **kwargs):
        opt = pl_module.opt
        self.opt = opt

        nkwargs = {}
        nkwargs['default_root_dir'] = opt.run_base_dir
        if opt.gpu_ids:
            nkwargs['accelerator'] = 'gpu'
            nkwargs['devices'] = opt.gpu_ids
        else:
            nkwargs['accelerator'] = 'cpu'
        
        nkwargs['logger'] = self.define_loggers(opt)
        nkwargs['callbacks'] = self.define_callbacks(opt)
        # mixed precision is default
        nkwargs['precision'] = 32 if opt.single_precision else 16
        
        nkwargs['max_epochs'] = opt.max_epochs   
        #nkwargs['val_check_interval'] = opt.val_check_interval
        nkwargs['check_val_every_n_epoch'] = opt.check_val_every_n_epoch
        nkwargs['log_every_n_steps'] = opt.log_every_n_steps
        nkwargs['detect_anomaly'] = opt.detect_anomaly
        
        # if any, update additional kwargs, but pl_module's opt has higher priority than additional kwargs
        for k,v in kwargs.items():
            if not k in nkwargs.keys():
                nkwargs[k] = v
        
        super().__init__(**nkwargs)
                
    def define_callbacks(self, opt):
        L = []
        # ModelCheckpoint
        if 'ckpt' in opt.callbacks.lower():
            if opt.checkpoint_nooverwrite:
                save_top_k = -1
            else:
                save_top_k = 1
            cb_checkpoint = ModelCheckpoint(
                dirpath=os.path.join(opt.save_dir, 'checkpoint'),
                every_n_epochs=opt.checkpoint_every_n_epochs,
                monitor=opt.checkpoint_monitor,
                mode=opt.checkpoint_monitor_mode,
                filename=opt.checkpoint_filename,
                auto_insert_metric_name=False,
                save_weights_only=not opt.save_fullmodel,
                save_top_k=(-1 if opt.checkpoint_nooverwrite else 1),
            )
            L.append(cb_checkpoint)
        # LearingRateMonitor
        if 'lr' in opt.callbacks.lower():
            cb_lrmonitor = LearningRateMonitor(logging_interval='epoch')
            L.append(cb_lrmonitor)
            
        return L
    
    def define_loggers(self, opt):
        save_dir_split = opt.save_dir.split(os.sep)
            
        save_dir_split = ['.','.'] + save_dir_split
        log_version = save_dir_split.pop()
        log_name = save_dir_split.pop()
        log_dir = os.path.join(*save_dir_split)
        
        L = []
        if (opt.loggers):
            if 'csv' in opt.loggers.lower():        # CSV Logger       
                L.append(CSVLogger(log_dir, name=log_name, version=log_version))
            if 'tb' in opt.loggers.lower():
                L.append(TensorBoardLogger(log_dir, name=log_name, version=log_version, sub_dir='tensorboard'))
            if 'wandb' in opt.loggers.lower():
                L.append(WandbLogger(save_dir=opt.save_dir, project='maicon', name=os.path.join(log_name, log_version)))
        
        if len(L)==0:
            L = False
        
        return L