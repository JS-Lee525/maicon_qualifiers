from imageio import imwrite
import json
import os
import numpy as np
from PIL import Image
import SimpleITK as sitk
from typing import Optional, Sequence, List, Tuple, Union
import wandb

import torch
from torchvision.transforms import ToPILImage, ToTensor
from torchvision.utils import make_grid
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger

class MyTrainer(pl.Trainer):
    '''Pytorch Lightning Trainer with custom callbacks, logger, args, etc.
    '''
    
    def __init__(self, pl_module: pl.LightningDataModule, **kwargs):
        opt = pl_module.opt
        self.opt = opt
        self.inference = opt.inference

        nkwargs = {}
        nkwargs['default_root_dir'] = opt.run_base_dir
        if opt.gpu_ids:
            nkwargs['accelerator'] = 'gpu'
            nkwargs['devices'] = opt.gpu_ids
        else:
            nkwargs['accelerator'] = 'cpu'
        
        nkwargs['logger'] = self.define_loggers(opt)
        nkwargs['log_every_n_steps'] = opt.log_every_n_steps
        nkwargs['callbacks'] = self.define_callbacks(opt)
        # mixed precision is default
        nkwargs['precision'] = 16 if opt.mixed_precision else 32
        
        if not self.inference:
            nkwargs['max_epochs'] = opt.max_epochs   
            nkwargs['check_val_every_n_epoch'] = opt.check_val_every_n_epoch
            nkwargs['detect_anomaly'] = opt.detect_anomaly
        
        # if any, update additional kwargs, but pl_module's opt has higher priority than additional kwargs
        for k,v in kwargs.items():
            if not k in nkwargs.keys():
                nkwargs[k] = v
        
        super().__init__(**nkwargs)
                
    def define_callbacks(self, opt):
        L = []
        if not opt.callbacks:
            return []
    
        callbacks = opt.callbacks.lower().split('_')
    
        # ModelCheckpoint
        if 'ckpt' in callbacks:
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
        if 'lr' in callbacks:
            cb_lrmonitor = LearningRateMonitor(logging_interval='epoch')
            L.append(cb_lrmonitor)
            
        # Results
        if 'result' in callbacks:
            cb_result = ResultsCallback(opt.result_dir)
            L.append(cb_result)
            
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
                L.append(WandbLogger(save_dir=opt.save_dir, project=opt.wandb_project, name=opt.wandb_name if opt.wandb_name else os.path.join(log_name, log_version)))
        
        if len(L)==0:
            L = False
        
        return L
    
# save results callback
class ResultsCallback(pl.Callback):
    '''Save results to SAVE_DIR
    '''
    def __init__(
        self,
        result_dir: Union[List[str], str] = './results',
    ):
        super().__init__()
        if not isinstance(result_dir, list):
            result_dir = [result_dir]
        self.result_dir = result_dir
    
    # Callback method
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):       
        keys = batch['metadata']['key']
        outs = outputs.detach().cpu().numpy()
        
        for i in range(outs.shape[0]):
            n_img = outs[i].astype(np.float16)
            fn = keys[i]           
            
            newpath = os.path.join(self.result_dir[dataloader_idx], f'{fn}.npy')
            
            np.save(newpath, n_img)
    
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        keys = batch['metadata']['key']
        outs = outputs.detach().cpu().numpy()
        
        for i in range(outs.shape[0]):
            n_img = outs[i].astype(np.float16)
            fn = keys[i]           
            
            newpath = os.path.join(self.result_dir[dataloader_idx], f'{fn}.npy')
            
            np.save(newpath, n_img)