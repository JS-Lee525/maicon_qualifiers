import numpy as np
import torch

def convert_multilabel_logit_to_mask(x):
    x = torch.sigmoid(torch.from_numpy(x.astype(np.float32))).detach().cpu().numpy() > 0.5
    
    h, w = x.shape[-2:]
    mask = np.zeros((h, w*2), dtype=np.uint8)
    
    # 소멸
    mask[:,:w][x[1]==1] = 2
    # 신축
    mask[:,w:][x[0]==1] = 1
    # 갱신
    mask[:,w:][x[2]==1] = 3
    
    return mask
    
def suppress_nonmax(x):
    ncols = x.shape[0]
    a = x.argmax(0)
        
    out = np.zeros( (a.size,ncols), dtype=np.uint8)
    out[np.arange(a.size),a.ravel()] = 1
    out.shape = a.shape + (ncols,)
    out = out.transpose(2,0,1)    
    
    x[out==0] = 0
    return x

def convert_multilabel_logit_to_mask_v2(x):
    x = torch.sigmoid(torch.from_numpy(x.astype(np.float32))).detach().cpu().numpy()
    
    x = suppress_nonmax(x)
    x = x > 0.5
    
    h, w = x.shape[-2:]
    mask = np.zeros((h, w*2), dtype=np.uint8)
    
    # 소멸
    mask[:,:w][x[1]==1] = 2
    # 신축
    mask[:,w:][x[0]==1] = 1
    # 갱신
    mask[:,w:][x[2]==1] = 3
    
    return mask
    
def convert_multiclass_logit_to_mask(x):
    x = x.argmax(0)

    h, w = x.shape[-2:]
    mask = np.zeros((h, w*2), dtype=np.uint8)
    
    # 소멸
    mask[:,:w][x==2] = 2
    # 신축
    mask[:,w:][x==1] = 1
    # 갱신
    mask[:,w:][x==3] = 3
    
    return mask