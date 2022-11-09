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
Multi-class.. ignore overlapping 1, 2 labels
'''

class MaiconMCPatchDataset(BaseDataset):       
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
        
        print(f'image: {len(self.image_paths)}\nmask: {len(self.mask_paths)}')
        
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
                
                
        if self.phase == 'train':
            self.keys = []
            self.padding = []
            
            crop_size = np.array([self.opt.patch_size,]*2)
            sliding_window_size = (crop_size*(1-self.opt.patch_overlap)).astype(int)
            
            for idx, x in enumerate(self.image_paths):
                im = Image.open(x)
                w, h = im.size
                
                # padding for sliding window
                old_sh = np.array((h, w//2))
                new_sh = (np.ceil(np.divide(old_sh - crop_size, sliding_window_size)) * sliding_window_size).astype(int) + crop_size
                pads = new_sh - old_sh
                pads = np.array([[a//2, a-a//2] for a in pads.astype(int)])          
                
                # sliding window indexes
                slices = np.divide(new_sh - crop_size, sliding_window_size).astype(int) + 1
                slices[slices<1] = 1
                slices = np.stack([mg.ravel() for mg in np.meshgrid(*[np.arange(s) for s in slices])], axis=-1) * sliding_window_size
                slices = np.concatenate([slices, slices+crop_size], axis=-1)
                
                for s in slices:
                    self.keys.append([idx, s])
                self.padding.append(pads)                
            
        print(f'keys: {len(self.keys)}')
    
    ## override this to read data by index. must return image, image2, mask or image, image2, None.
    def read_data(self, index):        
        if self.phase == 'train':
            idx, slices = self.keys[index]
            image_path = self.image_paths[idx]
            image = np.array(Image.open(image_path).convert('RGB')).astype(np.float32)/255.
            h, w = image.shape[:2]
            image1 = image[:, :w//2]
            image2 = image[:, w//2:w]   
            
            metadata = {'image_path': image_path,}
            
            pads = self.padding[idx]
            s1, s2, e1, e2 = slices
            slice_func = (slice(s1, e1), slice(s2, e2), slice(None),)
            
            image1 = np.pad(image1, tuple([tuple(p) for p in pads]) + ((0,0),), 'constant', constant_values=0)
            image2 = np.pad(image2, tuple([tuple(p) for p in pads]) + ((0,0),), 'constant', constant_values=0)
            image1 = image1[slice_func]
            image2 = image2[slice_func]
            
            if len(self.mask_paths)>0:
                mask_path = self.mask_paths[idx]
                mask = (np.array(Image.open(mask_path).convert('L'))).astype(np.uint8)
                h, w = mask.shape[:2]
                mask1 = mask[:, :w//2]
                mask2 = mask[:, w//2:]
                
                catmask = np.zeros((h, w//2), dtype=np.uint8)
                catmask[mask1==2] = 2 # 소멸
                catmask[mask2==1] = 1 # 신축
                catmask[mask2==3] = 3 # 갱신
                
                mask = np.expand_dims(catmask, axis=-1)
                
                mask = np.pad(mask, tuple([tuple(p) for p in pads]) + ((0,0),), 'constant', constant_values=0)
                mask = mask[slice_func]
                metadata['mask_path'] = mask_path
            else:
                mask = None
        
        else:
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
                
                catmask = np.zeros((h, w//2), dtype=np.uint8)
                catmask[mask1==2] = 2
                catmask[mask2==1] = 1
                catmask[mask2==3] = 3
                
                mask = np.expand_dims(catmask, axis=-1)
                metadata['mask_path'] = mask_path
            else:
                mask = None

        return image1, image2, mask, metadata
