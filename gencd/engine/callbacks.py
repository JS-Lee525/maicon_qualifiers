import copy
import os
import numpy as np
from PIL import Image
from typing import Optional, Sequence, List, Tuple, Union
import wandb

import torch
import pytorch_lightning as pl

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
        #outs = outputs.detach().cpu().numpy()
        outs = pl_module.outputs.detach().cpu().numpy()
        
        for i in range(outs.shape[0]):
            n_img = outs[i].astype(np.float16)
            fn = keys[i]           
            
            newpath = os.path.join(self.result_dir[dataloader_idx], f'{fn}.npy')
            
            np.save(newpath, n_img)
    
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        keys = batch['metadata']['key']
        #outs = outputs.detach().cpu().numpy()
        outs = pl_module.outputs.detach().cpu().numpy()
        
        for i in range(outs.shape[0]):
            n_img = outs[i].astype(np.float16)
            fn = keys[i]           
            
            newpath = os.path.join(self.result_dir[dataloader_idx], f'{fn}.npy')
            
            np.save(newpath, n_img)
            
            
# metrics callback
# for logging best validation metric (best means when monitor is best. e.g. val_loss)
class MetricsBestValidCallback(pl.Callback):
    def __init__(
        self,
        metric: Union[List[str], str] = 'metric/val_mIOU',
        monitor: str = 'loss/val_loss',
        monitor_mode: str = 'min',
    ):
        super().__init__()
        if not isinstance(metric, list):
            metric = [metric]
        self.metric = metric
        self.monitor = monitor
        self.monitor_mode = monitor_mode
        self.all_metrics = {}
    
    def on_validation_epoch_end(self, trainer, pl_module):
        each_me = copy.deepcopy(trainer.callback_metrics)
        for k,v in each_me.items():
            if k in self.all_metrics.keys():
                self.all_metrics[k].append(v.item())
            else:
                self.all_metrics[k] = [v.item()]                
                
        # last validation epoch
        every_n_epoch = pl_module.hparams['opt'].check_val_every_n_epoch
        if pl_module.current_epoch == every_n_epoch*(trainer.max_epochs//every_n_epoch) - 1:
            self.log_best()
                
    def log_best(self):
        monitored = self.all_metrics[self.monitor]
        if self.monitor_mode == 'min':
            idx = np.argmin(np.array(monitored))
        else:
            idx = np.argmax(np.array(monitored))
        
        for k in self.metric:
            self.log(k+'_best', self.all_metrics[k][idx])