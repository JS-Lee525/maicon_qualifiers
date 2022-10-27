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