import argparse
from collections import OrderedDict
import itertools
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import pytorch_lightning as pl

from monai.inferers import sliding_window_inference
from monai.metrics import MeanIoU

from .losses import define_loss
from .networks.utils import define_network, load_pretrained_net
from .utils import get_scheduler, define_optimizer

class CDBaseModel(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.save_hyperparameters()
        '''
        self.inference = opt.inference
        
        if not self.inference:
            self.result_names = ['mask', 'predS']
        else:
            self.result_names = ['testout']
        '''
        # define networks
        self.netC = define_network(opt.net_config, opt.net_module)
        if opt.load_pretrained:
            self.netC = load_pretrained_net(self.netC, opt.load_pretrained)
            
        # define loss functions
        self.criterion = define_loss(opt.loss)
        
        # define metric
        self.metric = MeanIoU()
        
    ### predefined methods
    
    def configure_optimizers(self):
        optimizer_C = define_optimizer(self.netC.parameters(), self.hparams['opt'])
        optimizers = [optimizer_C]
        schedulers = [get_scheduler(optimizer, self.hparams['opt']) for optimizer in optimizers]
        
        return optimizers, schedulers
    
    def forward(self, x1, x2):
        output = self.netC(x1, x2)
        if isinstance(output, tuple):
            output = output[0]
        return output
            
    def training_step(self, batch, batch_idx):
        self.set_input(batch)
        self.predC = self.forward(self.image, self.image2)
        
        loss = self.criterion(self.predC, self.mask)
        self.log(f'loss/train_loss', loss, batch_size=self.current_batch_size, on_step=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self.forward_test(batch)
        
        loss = self.criterion(outputs, self.mask)
        
        bin_outputs = F.one_hot(outputs.argmax(1), num_classes=int(outputs.shape[1])).permute(0,3,1,2)
        self.metric(bin_outputs, self.mask)
        
        self.log(f'loss/val_loss', loss, batch_size=self.current_batch_size, on_step=True, on_epoch=True)
                    
        return loss 
    
    def validation_epoch_end(self, outputs):
        mean_val_metric = self.metric.aggregate().item()
        self.metric.reset()
        self.log(f'metric/val_mIOU', mean_val_metric)
    
    ### custom methods
    
    def set_input(self, batch):
        self.image = batch['image']
        self.image2 = batch['image2']
        self.current_batch_size = batch['image'].shape[0]
        
        if 'mask' in batch.keys():
            self.mask = F.one_hot(batch['mask'].long(), num_classes=self.hparams['opt'].num_class).permute(0,3,1,2)
    
    def forward_cat(self, xs):
        nch = int(xs.shape[1])
        x1 = xs[:,:nch//2]
        x2 = xs[:,nch//2:]
        return self.forward(x1, x2)
    
    def forward_test(self, batch):
        self.set_input(batch)
        images = torch.cat((self.image, self.image2), dim=1)
        roi_size = (self.hparams['opt'].patch_size, self.hparams['opt'].patch_size)
        sw_batch_size = self.hparams['opt'].batch_size
        
        outputs = sliding_window_inference(images, roi_size, sw_batch_size, self.forward_cat, overlap=0.5, mode='gaussian')
        
        return outputs
    