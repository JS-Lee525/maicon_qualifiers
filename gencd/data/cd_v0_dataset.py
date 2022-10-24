import glob
import importlib
import numpy as np
import os
from PIL import Image

import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from .cd_base_dataset import CDBaseDataset

class CDv0Dataset(CDBaseDataset):
    ## override this to define self.keys and paths
    def prepare_data(self):
        basedir = os.path.join(self.opt.datadir, self.phase)
        self.image_dir = os.path.join(basedir, 'Image1')
        self.image2_dir = os.path.join(basedir, 'Image2')
        self.mask_dir = os.path.join(basedir, 'label')        
        
        self.image_paths = sorted(glob.glob(os.path.join(self.image_dir, f'*.{self.opt.file_extension}')))
        self.image2_paths = sorted(glob.glob(os.path.join(self.image2_dir, f'*.{self.opt.file_extension}')))
        self.mask_paths = sorted(glob.glob(os.path.join(self.mask_dir, f'*.{self.opt.file_extension}')))
        
        print(f'image: {len(self.image_paths)}\nimage2: {len(self.image2_paths)}\nmask: {len(self.mask_paths)}')
        assert len(self.image_paths)==len(self.image2_paths)
        if len(self.mask_paths)>0:
            assert len(self.image_paths)==len(self.mask_paths)
        
        self.keys = [os.path.basename(x).split(f'.{self.opt.file_extension}')[0] for x in self.image_paths]
    
    ## override this to read data by index. must return image, image2, mask or image, image2, None.
    def read_data(self, index):        
        image_path = self.image_paths[index]
        image2_path = self.image2_paths[index]
        image = np.array(Image.open(image_path).convert('RGB')).astype(np.float32)/255.
        image2 = np.array(Image.open(image2_path).convert('RGB')).astype(np.float32)/255.
        
        if len(self.mask_paths)>0:
            mask_path = self.mask_paths[index]
            mask = (np.array(Image.open(mask_path).convert('L'))/255.>0.5).astype(int)
        else:
            mask = None
        
        return image, image2, mask
    
    ## override this to define self.transform
    def prepare_transforms(self):
        self.transform = None
        if self.phase == 'train':
            transforms1 = A.Compose([
                A.RandomBrightnessContrast(p=0.8),    
                A.RandomGamma(p=0.8),
            ])            
            transforms2 = A.Compose([
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=30, p=0.5),
                A.RandomCrop(self.opt.patch_size, self.opt.patch_size, p=1.0),
                ToTensorV2(p=1.0, transpose_mask=True),
            ], additional_targets={'image2': 'image'})   
            
            self.transform = ComposeList([transforms1, transforms2])
        else:
            self.transform = A.Compose([ToTensorV2(p=1.0, transpose_mask=True)], additional_targets={'image2': 'image'})    
    
class ComposeList:
    def __init__(self, list_compose):
        self.list_compose = list_compose
    
    def __call__(self, **kwargs):
        for x in self.list_compose:
            kwargs = x(**kwargs)
        return kwargs