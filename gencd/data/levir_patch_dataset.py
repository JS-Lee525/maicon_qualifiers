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

class LevirPatchDataset(BaseDataset):
    '''Run all possible patches in training phase
    '''
    
    ## override this to define self.keys and paths
    def prepare_data(self):
        if self.phase == 'train':
            basedir = os.path.join(self.opt.datadir, 'train')
        elif self.phase == 'val':
            basedir = os.path.join(self.opt.datadir, 'val')
        elif self.phase == 'test':
            basedir = os.path.join(self.opt.datadir, 'test')
        
        self.image_dir = os.path.join(basedir, 'A')
        self.image2_dir = os.path.join(basedir, 'B')
        self.mask_dir = os.path.join(basedir, 'label')        
        
        self.image_paths = sorted(glob.glob(os.path.join(self.image_dir, '*.png')))
        self.image2_paths = sorted(glob.glob(os.path.join(self.image2_dir, '*.png')))
        self.mask_paths = sorted(glob.glob(os.path.join(self.mask_dir, '*.png')))
        
        print(f'image: {len(self.image_paths)}\nimage2: {len(self.image2_paths)}\nmask: {len(self.mask_paths)}')
        assert len(self.image_paths)==len(self.image2_paths)
        if len(self.mask_paths)>0:
            assert len(self.image_paths)==len(self.mask_paths)
        
        if self.phase == 'train':
            self.keys = []
            self.padding = []
            
            crop_size = np.array([self.opt.patch_size,]*2)
            sliding_window_size = (crop_size*(1-self.opt.patch_overlap)).astype(int)
            
            for idx, x in enumerate(self.image_paths):
                im = Image.open(x)
                
                # padding for sliding window
                old_sh = np.array(im.size[::-1])
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
        else:
            self.keys = [os.path.basename(x).split('.')[0] for x in self.image_paths]
    
        print(f'keys: {len(self.keys)}')
        
    ## override this to read data by index. must return image, image2, mask or image, image2, None.
    def read_data(self, index):        
        if self.phase == 'train':
            idx, slices = self.keys[index]
            image_path = self.image_paths[idx]
            image2_path = self.image2_paths[idx]
            image = np.array(Image.open(image_path).convert('RGB')).astype(np.float32)/255.
            image2 = np.array(Image.open(image2_path).convert('RGB')).astype(np.float32)/255.            
            
            metadata = {'image_path': image_path, 'image2_path': image2_path}
            
            pads = self.padding[idx]
            s1, s2, e1, e2 = slices
            slice_func = (slice(s1, e1), slice(s2, e2), slice(None),)
            
            image = np.pad(image, tuple([tuple(p) for p in pads]) + ((0,0),), 'constant', constant_values=0)
            image2 = np.pad(image2, tuple([tuple(p) for p in pads]) + ((0,0),), 'constant', constant_values=0)
            image = image[slice_func]
            image2 = image2[slice_func]
            
            if len(self.mask_paths)>0:
                mask_path = self.mask_paths[idx]
                mask = (np.array(Image.open(mask_path).convert('L'))/255.>0.5).astype(np.uint8)
                mask = np.expand_dims(mask, axis=-1)
                mask = np.pad(mask, tuple([tuple(p) for p in pads]) + ((0,0),), 'constant', constant_values=0)
                mask = mask[slice_func]
                metadata['mask_path'] = mask_path
            else:
                mask = None
        else:
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
