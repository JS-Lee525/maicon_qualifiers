import re
import torch.nn as nn
from monai.losses import DiceLoss, DiceCELoss #, DiceFocalLoss
from .dice import DiceFocalLoss

def define_loss(loss_name):
    eps = 1e-5
    
    if loss_name.lower() == 'ce':
        loss = nn.CrossEntropyLoss()
    elif loss_name.lower() == 'dice':
        loss = DiceLoss(softmax=True, include_background=False, smooth_nr=eps, smooth_dr=eps,)
    elif loss_name.lower() == 'dicece':
        loss = DiceCELoss(softmax=True, include_background=False, smooth_nr=eps, smooth_dr=eps,)
    elif loss_name.lower() == 'dicefocal':
        loss = DiceFocalLoss(softmax=True, include_background=False, smooth_nr=eps, smooth_dr=eps,)
    else:
        raise NotImplementedError(f'loss name [{loss_name}] is not recognized')        
    return loss