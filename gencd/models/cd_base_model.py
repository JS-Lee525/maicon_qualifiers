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
from monai.networks import one_hot

from .losses import define_loss
from .networks.utils import define_network, load_pretrained_net
from .utils import get_scheduler, define_optimizer, define_metrics

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
        self.net_names = ['netC']
        self.netC = define_network(opt.net_config, opt.net_module)
        if opt.load_pretrained_model is None and opt.load_pretrained_network:
            self.netC = load_pretrained_net(self.netC, opt.load_pretrained_network)
            
        # define loss functions
        self.criterion = define_loss(opt.loss)
        
        # define metric
        self.metrics = define_metrics(opt.metric)                                                           
        
    ### predefined methods
    
    def configure_optimizers(self):
        optimizer_C = define_optimizer(self.netC.parameters(), self.hparams['opt'])
        optimizers = [optimizer_C]
        schedulers = [get_scheduler(optimizer, self.hparams['opt']) for optimizer in optimizers]
        
        return optimizers, schedulers
    
    def forward(self, x1, x2):
        output = self.netC(x1, x2)
        if isinstance(output, tuple) or isinstance(output, list):
            output = output[-1]
        return output
            
    def training_step(self, batch, batch_idx):
        self.set_input(batch)
        self.predC = self.forward(self.image, self.image2)
        
        loss = self.criterion(self.predC, self.mask)
        self.log(f'loss/train_loss', loss, batch_size=self.current_batch_size, on_step=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self._step_test(batch)
        
        loss = self.criterion(outputs, self.mask)
        
        bin_outputs = one_hot(outputs.argmax(1, keepdim=True), self.hparams['opt'].num_class)
        for k in self.metrics.keys():
            self.metrics[k](bin_outputs, self.mask)
        
        self.log(f'loss/val_loss', loss, batch_size=self.current_batch_size, on_step=True, on_epoch=True)
                    
        return loss
    
    def validation_epoch_end(self, outputs):
        for k in self.metrics.keys():
            mean_metric = self.metrics[k].aggregate()
            if isinstance(mean_metric, list):
                mean_metric = mean_metric[0]
            mean_metric = mean_metric.item()
            self.metrics[k].reset()
            self.log(f'metric/val_{k}', mean_metric)
    
    def predict_step(self, batch, batch_idx):
        return self._step_test(batch)
    
    def test_step(self, batch, batch_idx):
        outputs = self._step_test(batch)
        loss = None
        
        if 'mask' in batch.keys():
            loss = self.criterion(outputs, self.mask)
            self.log(f'loss/test_loss', loss, batch_size=self.current_batch_size, on_step=True, on_epoch=True)
            bin_outputs = one_hot(outputs.argmax(1, keepdim=True), self.hparams['opt'].num_class)
            for k in self.metrics.keys():
                self.metrics[k](bin_outputs, self.mask)
                    
        return outputs

    def test_epoch_end(self, outputs):
        for k in self.metrics.keys():
            mean_metric = self.metrics[k].aggregate()
            if isinstance(mean_metric, list):
                mean_metric = mean_metric[0]
            mean_metric = mean_metric.item()
            self.metrics[k].reset()
            self.log(f'metric/test_{k}', mean_metric)
        
    
    ### custom methods
    
    def set_input(self, batch):
        self.image = batch['image']
        self.image2 = batch['image2']
        self.current_batch_size = batch['image'].shape[0]
        
        if 'mask' in batch.keys():
            self.mask = one_hot(batch['mask'], self.hparams['opt'].num_class)
    
    def forward_test(self, xs):
        nch = int(xs.shape[1])
        x1 = xs[:,:nch//2]
        x2 = xs[:,nch//2:]
        out = self.forward(x1, x2)
        return out
    
    def _step_test(self, batch):
        self.set_input(batch)
        images = torch.cat((self.image, self.image2), dim=1)
        roi_size = (self.hparams['opt'].patch_size, self.hparams['opt'].patch_size)
        sw_batch_size = self.hparams['opt'].batch_size
        
        outputs = sliding_window_inference(images, roi_size, sw_batch_size, self.forward_test, overlap=0.5, mode='gaussian')
        
        return outputs
    
    def load_pretrained(self, path):
        self.load_pretrained_nets(path, nets=self.net_names)
    
    def load_pretrained_nets(self, path, nets=[]):
        '''For loading state_dict of part of the model.
        Loading full model should be done by "load_from_checkpoint" (Lightning)
        '''
        
        device = next(self.parameters()).device
        
        # load from checkpoint or state_dict
        print(f'trying to load pretrained from {path}')
        try:
            state_dict = torch.load(path, map_location=device)['state_dict']
        except:
            state_dict = torch.load(path, map_location=device)
        
        if len(nets)==0:
            self.load_state_dict(state_dict)
        
        all_keys_match = True
        for name in nets:
            if hasattr(self, name):
                net = getattr(self, name)
                new_weights = net.state_dict()
                
                # first check if pretrained has all keys
                keys_match = True
                for k in new_weights.keys():
                    if not f'{name}.{k}' in state_dict.keys():
                        keys_match = False
                        all_keys_match = False
                        print(f"not loading {name} because keys don't match")
                        break
                if keys_match:
                    for k in new_weights.keys():
                        new_weights[k] = state_dict[f'{name}.{k}']
                    net.load_state_dict(new_weights)
                        
        if all_keys_match:
            print('<All keys matched successfully>')