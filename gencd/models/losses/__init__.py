import re
import torch.nn as nn
from monai.losses import DiceLoss, DiceCELoss, DiceFocalLoss

def define_loss(loss_name):
    if loss_name.lower() == 'ce':
        loss = nn.CrossEntropyLoss()
    elif loss_name.lower() == 'dice':
        loss = DiceLoss()
    elif loss_name.lower() == 'dicece':
        loss = DiceCELoss()
    elif loss_name.lower() == 'dicefocal':
        loss = DiceFocalLoss()
    else:
        raise NotImplementedError(f'loss name [{loss_name}] is not recognized')        
    return loss