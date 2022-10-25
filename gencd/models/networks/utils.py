import copy
import importlib
import functools
import numpy as np
import os
from typing import Optional, Sequence, Tuple, Union
import yaml

import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler

def define_network(net_config, net_module=None):
    if net_module is None:
        net_module = 'gencd.models.networks'
    netlib = importlib.import_module(net_module)
    
    with open(net_config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    net_name = cfg['net_name']
    config = cfg['config']
    
    netcls = None
    if not hasattr(netlib, net_name):
        print(f'In {net_module}, there should be a class that matches {net_name}.')        
    else:
        netcls = getattr(netlib, net_name)
    
    if netcls:
        net = netcls(**config)       
        return net
    return None


def load_pretrained_net(net, path):
    '''allow partial loading
    '''

    device = next(net.parameters()).device

    # load from checkpoint or state_dict
    print(f'trying to load pretrained from {path}')
    try:
        state_dict = torch.load(path, map_location=device)['state_dict']
    except:
        state_dict = torch.load(path, map_location=device)

    all_okay = True

    new_weights = net.state_dict()

    # partial loading. check key and shape
    for k in new_weights.keys():
        if not k in state_dict.keys():
            print(f'{k} is missing in pretrained')
            all_okay = False
        else:
            if new_weights[k].shape != state_dict[k].shape:
                print(f'skip {k}, required shape: {new_weights[k].shape}, pretrained shape: {state_dict[k].shape}')
                all_okay = False
            else:       
                new_weights[k] = state_dict[k]
    
    try:
        net.load_state_dict(new_weights)
        if all_okay:
            print('<All weights loaded successfully>')
    except:
        print(f'cannot load {path}. using intial net.')
        pass
    
    return net


def init_weights(net, init_type='xavier'):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'xavier':
                init.xavier_normal_(m.weight)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    
    net.apply(init_func)
    return net