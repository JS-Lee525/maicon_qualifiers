import glob
import importlib
import numpy as np
import os
import pickle
from PIL import Image

import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from .base_dataset import BaseDataset

'''
DATADIR should be ...../01_data/
'''

class MaiconImageDataset(BaseDataset):       
    ## override this to define self.keys and paths
    def prepare_data(self):
        if self.phase in ['train', 'val']:
            basedir = os.path.join(self.opt.datadir, 'train')
        elif self.phase == 'test':
            basedir = os.path.join(self.opt.datadir, 'test')
        
        self.image_dir = os.path.join(basedir, 'x')
        self.mask_dir = os.path.join(basedir, 'y')        
        
        self.image_paths = sorted(glob.glob(os.path.join(self.image_dir, '*.png')))
        self.mask_paths = sorted(glob.glob(os.path.join(self.mask_dir, '*.png')))
        
        _size = min(len(self.image_paths), self.opt.max_dataset_size)
        self.image_paths = self.image_paths[:_size]
        self.mask_paths = self.mask_paths[:_size]
        
        
        
        self.keys = [os.path.basename(x).split('.')[0] for x in self.image_paths]
        
        if self.phase in ['train', 'val']:
            cfold = int(self.opt.fold[0]) if (isinstance(self.opt.fold, list)) else self.opt.fold
            with open(self.opt.dataset_split, 'rb') as f:
                self.ds_split = pickle.load(f)        
            if self.phase == 'train':
                self.image_paths = [self.image_paths[i] for i,x in enumerate(self.keys) if x in self.ds_split[cfold]['train']]
                self.mask_paths = [self.mask_paths[i] for i,x in enumerate(self.keys) if x in self.ds_split[cfold]['train']]
                self.keys = [x for x in self.keys if x in self.ds_split[cfold]['train']]
            elif self.phase == 'val':
                self.image_paths = [self.image_paths[i] for i,x in enumerate(self.keys) if x in self.ds_split[cfold]['valid']]
                self.mask_paths = [self.mask_paths[i] for i,x in enumerate(self.keys) if x in self.ds_split[cfold]['valid']]
                self.keys = [x for x in self.keys if x in self.ds_split[cfold]['valid']]
                         
        print(f'image: {len(self.image_paths)}\nmask: {len(self.mask_paths)}')
                    
        print(f'keys: {len(self.keys)}')
    
    ## override this to read data by index. must return image, image2, mask or image, image2, None.
    def read_data(self, index):        
        image_path = self.image_paths[index]
        image = np.array(Image.open(image_path).convert('RGB')).astype(np.float32)/255.

        h, w = image.shape[:2]
        image1 = image[:, :w//2]
        image2 = image[:, w//2:w]

        key = self.keys[index]

        metadata = {'key': key, 'image_path': image_path,}

        if len(self.mask_paths)>0:
            mask_path = self.mask_paths[index]
            mask = (np.array(Image.open(mask_path).convert('L'))).astype(np.uint8)
            h, w = mask.shape[:2]
            mask1 = mask[:, :w//2]
            mask2 = mask[:, w//2:]

            catmask = np.zeros((h, w//2, self.opt.num_class), dtype=np.float32)
            catmask[...,0][mask2==1] = 1 # 신축
            catmask[...,2][mask2==3] = 1 # 갱신
            catmask[...,1][mask1==2] = 1 # 소멸

            mask = catmask
            metadata['mask_path'] = mask_path
        else:
            mask = None

        return image1, image2, mask, metadata
