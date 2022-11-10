import glob
from imageio import imread, imwrite
import numpy as np
import os
import shutil

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

if __name__ == '__main__':

    phase = 'test'

    # 0=multilabel, 1=multiclass
    result_dirs = [
        ['./temp/result/changer/00001/fold0/e009', 0],
        ['./temp/result/changerMC/00001/fold0/e007', 1],
    ]

    datadir_base = '/workspace/maicon_qualifiers/temp/data/01_data'

    dir_image = os.path.join(datadir_base, phase, 'x')

    for result_dir in result_dirs:
        result_npys = sorted(glob.glob(os.path.join(result_dir[0], '*.npy')))
        submit_dir = os.path.join(result_dir[0], 'submit2')
        os.makedirs(submit_dir, exist_ok=True)
        for x in tqdm(result_npys):
            tkey = os.path.basename(x).split('.npy')[0]
            tx = np.load(x)

            tresult = convert_multilabel_logit_to_mask_v2(tx) if result_dir[1] == 0 else \
            convert_multiclass_logit_to_mask(tx)

            imwrite(os.path.join(submit_dir, f'{tkey}.png'), tresult)

    submit_base = '/workspace/maicon_qualifiers/temp/result/changerMC/00001/fold0/e007'
    submit_dir = os.path.join(submit_base, 'submit')
    submit_zip = os.path.join(submit_base, 'submit_changerMC_00001_fold0_e007.zip')

    predfiles = sorted(glob.glob(os.path.join(submit_dir, '*.png')))

    shutil.make_archive(submit_zip.split('.zip')[0], 'zip', submit_dir)

    submitcommand = "curl -X POST 'https://api.aiconnect.kr/assignment/251/results' \
    --header 'Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJhdG9rZW4iLCJpc3MiOiJtbmMtYWljb25uZWN0LXVzZXIiLCJleHAiOjE2NjgzMzQ1OTksImlhdCI6MTY2NzcyOTc5OSwiZW1haWwiOiJkamMwMTA1QGdtYWlsLmNvbSJ9.9yvp0EBXTUENB2iCQG65xukvlFY-oXgwv5q7Nu6_MeE' \
    --form 'file=@"
    submitcommand += submit_zip
    submitcommand += "'"

    print(submitcommand)