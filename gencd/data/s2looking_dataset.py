import glob
import importlib
import numpy as np
import os
from PIL import Image

import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from .base_dataset import BaseDataset

class S2LookingDataset(BaseDataset):       
    ## override this to define self.keys and paths
    def prepare_data(self):
        basedir = os.path.join(self.opt.datadir, self.phase)
        self.image_dir = os.path.join(basedir, 'Image1')
        self.image2_dir = os.path.join(basedir, 'Image2')
        self.mask_dir = os.path.join(basedir, 'label')        
        
        self.image_paths = sorted(glob.glob(os.path.join(self.image_dir, '*.png')))
        self.image2_paths = sorted(glob.glob(os.path.join(self.image2_dir, '*.png')))
        self.mask_paths = sorted(glob.glob(os.path.join(self.mask_dir, '*.png')))
        
        print(f'image: {len(self.image_paths)}\nimage2: {len(self.image2_paths)}\nmask: {len(self.mask_paths)}')
        assert len(self.image_paths)==len(self.image2_paths)
        if len(self.mask_paths)>0:
            assert len(self.image_paths)==len(self.mask_paths)
        
        self.keys = [os.path.basename(x).split('.')[0] for x in self.image_paths]
    
    ## override this to read data by index. must return image, image2, mask or image, image2, None.
    def read_data(self, index):        
        image_path = self.image_paths[index]
        image2_path = self.image2_paths[index]
        image = np.array(Image.open(image_path).convert('RGB')).astype(np.float32)/255.
        image2 = np.array(Image.open(image2_path).convert('RGB')).astype(np.float32)/255.
        key = self.keys[index]
        
        metadata = {'key': key, 'image_path': image_path, 'image2_path': image2_path}
        
        if len(self.mask_paths)>0:
            mask_path = self.mask_paths[index]
            mask = (np.array(Image.open(mask_path).convert('L'))/255.>0.5).astype(np.uint8)
            mask = np.expand_dims(mask, axis=-1)
            metadata['mask_path'] = mask_path
        else:
            mask = None
        
        return image, image2, mask, metadata
