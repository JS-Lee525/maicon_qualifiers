from collections import OrderedDict
import json
import os
import numpy as np
import pickle
from sklearn.model_selection import KFold
from typing import Optional
from torch.utils.data import DataLoader
import torchvision.transforms
import pytorch_lightning as pl

from . import find_dataset_using_name

class MyDataModule(pl.LightningDataModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        
    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            dataset_class = find_dataset_using_name(self.opt.dataset_mode)
            self.ds_train = dataset_class(self.opt, phase='train')
            print(f"train dataset [{type(self.ds_train).__name__}] was created")  
            self.ds_valid = dataset_class(self.opt, phase='val')
            print(f"valid dataset [{type(self.ds_valid).__name__}] was created")  

        if stage == "test" or stage is None:            
            dataset_class = find_dataset_using_name(self.opt.dataset_mode)
            self.ds_test = dataset_class(self.opt, phase='test')
            print(f"test datasets [{type(self.ds_test).__name__}] were created")            
            
    def train_dataloader(self):
        DL = DataLoader(
            self.ds_train, 
            batch_size=self.opt.batch_size,
            #drop_last=self.opt.batch_drop_last,
            shuffle=True,
            num_workers=int(self.opt.num_workers),
            pin_memory=True,
        )
        return DL
    
    def val_dataloader(self):
        if len(self.ds_valid) > 0:
            DL = DataLoader(
                self.ds_valid, 
                batch_size=self.opt.batch_size_inference,
                #drop_last=self.opt.batch_drop_last,
                shuffle=False,
                num_workers=int(self.opt.num_workers),
                pin_memory=True,
            )
            return DL
        else:
            return None
        
    def test_dataloader(self):
        if len(self.ds_test) > 0:
            DL = DataLoader(
                self.ds_test, 
                batch_size=self.opt.batch_size_inference,
                shuffle=False,
                num_workers=int(self.opt.num_workers),
                pin_memory=True,
            )
            return DL
        else:
            return None