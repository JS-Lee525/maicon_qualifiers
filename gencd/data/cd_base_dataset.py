import glob
import importlib
import numpy as np
import os
from PIL import Image

import torch
from torch.utils.data import Dataset

class CDBaseDataset(Dataset):
    def __init__(self, opt, phase='train'):
        super().__init__()
        self.opt = opt
        self.phase = phase
        self.prepare_data()
        self.prepare_transforms()
        
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        image, image2, mask = self.read_data(index)
        
        input_items = {
            'image': image, 'image2': image2
        }
        if mask is not None:
            input_items['mask'] = mask
                
        return_items = self.transform(**input_items)
        
        return return_items
    
    ## override this to define self.keys, paths, and etc.
    def prepare_data(self):
        pass
    
    ## override this to read data by index. must return image, image2, mask or image, image2, None.
    def read_data(self, index):
        return None, None, None
    
    ## override this to define self.transform
    def prepare_transforms(self):
        pass