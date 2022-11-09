import glob
import importlib
import numpy as np
import os
from PIL import Image

import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from .levir_dataset import LevirDataset
from .transforms import ComposeList

class LEVIRv0normDataset(LevirDataset):    
    ## override this to define self.transform
    def prepare_transforms(self):
        self.transform = None
        if self.phase == 'train':
            self.transform = A.Compose([                
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=30, p=0.5),        
                
                A.CropNonEmptyMaskIfExists(self.opt.patch_size, self.opt.patch_size, p=1),
                #A.RandomCrop(self.opt.patch_size, self.opt.patch_size, p=1),
                #A.OneOf([
                #    A.CropNonEmptyMaskIfExists(self.opt.patch_size, self.opt.patch_size, p=1),
                #    A.RandomCrop(self.opt.patch_size, self.opt.patch_size, p=1),
                #], p=1.0),                
                
                A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5), max_pixel_value=1.0, p=1.0),
                ToTensorV2(p=1.0, transpose_mask=True),                
            ], additional_targets={'image2': 'image'})            
        else:
            self.transform = A.Compose([
                A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5), max_pixel_value=1.0, p=1.0),
                ToTensorV2(p=1.0, transpose_mask=True),
            ], additional_targets={'image2': 'image'})    
    
