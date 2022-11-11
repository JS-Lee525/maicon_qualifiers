import re
import torch
import torch.nn as nn
from monai.losses import DiceLoss, DiceCELoss, FocalLoss, TverskyLoss #, DiceFocalLoss
from .dice import DiceFocalLoss
from .boundary import BoundaryBCEWithLogitsLoss, BoundaryLoss

def define_loss(loss_name):
    eps = 1e-5
    
    if loss_name.lower() == 'ce':
        loss = nn.CrossEntropyLoss()
    elif loss_name.lower() == 'bce':
        loss = nn.BCEWithLogitsLoss()
    elif loss_name.lower() == 'bdbce':
        loss = BoundaryBCEWithLogitsLoss()        
    elif bool(re.match(r"bce(_[0-9]+)+", loss_name.lower())):
        pos_weight = torch.tensor([float(x) for x in loss_name.lower().split('_')[1:]])[:,None,None]
        loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight) 
    elif loss_name.lower() == 'focal':
        loss = FocalLoss(include_background=False, gamma=2, weight=0.5)
    elif loss_name.lower() == 'dice':
        loss = DiceLoss(softmax=True, include_background=False, smooth_nr=eps, smooth_dr=eps,)
    elif loss_name.lower() == 'dicece':
        loss = DiceCELoss(softmax=True, include_background=False, smooth_nr=eps, smooth_dr=eps,)
    elif loss_name.lower() == 'dicefocal':
        loss = DiceFocalLoss(softmax=True, gamma=2, focal_weight=0.5, include_background=False, smooth_nr=eps, smooth_dr=eps,)       
    elif loss_name.lower() == 'focal_bg':
        loss = FocalLoss(include_background=True, gamma=2, weight=0.5)
    elif loss_name.lower() == 'dice_bg':
        loss = DiceLoss(softmax=True, include_background=True, smooth_nr=eps, smooth_dr=eps,)
    elif loss_name.lower() == 'dicece_bg':
        loss = DiceCELoss(softmax=True, include_background=True, smooth_nr=eps, smooth_dr=eps,)
    elif loss_name.lower() == 'dicefocal_bg':
        loss = DiceFocalLoss(softmax=True, gamma=2, focal_weight=0.5, include_background=True, smooth_nr=eps, smooth_dr=eps,)
    elif loss_name.lower() == 'tversky':
        loss = TverskyLoss(softmax=True, include_background=True, smooth_nr=eps, smooth_dr=eps,)
    else:
        raise NotImplementedError(f'loss name [{loss_name}] is not recognized')        
    return loss