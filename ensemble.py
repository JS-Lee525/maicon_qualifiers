import copy
import numpy as np
import os
import shutil
from tqdm.autonotebook import tqdm
import torch

from gencd.data.datamodule import MyDataModule
from gencd.engine.trainer import MyTrainer
from gencd.models import create_model
from gencd.options import TrainOptions, TestOptions, EnsembleOptions
from gencd.utils.misc import find_files

if __name__ == '__main__':
    eopt = EnsembleOptions().parse()
    assert eopt.load_pretrained_model
    
    # run individual tests
    main_dir = eopt.result_dir
    save_dirs = []
    for i, pretrained in enumerate(eopt.load_pretrained_model):
        opt = copy.deepcopy(eopt)
        opt.load_pretrained_model = pretrained
        
        save_dir = os.path.join(main_dir, f'{i:02d}')
        os.makedirs(save_dir, exist_ok=True)
        save_dirs.append(save_dir)
        opt.result_dir = opt.save_dir = opt.run_base_dir = save_dir
    
        ### update from pretrained model checkpoint
        ckpt = torch.load(opt.load_pretrained_model, map_location=torch.device('cpu'))
        old_opt = ckpt['hyper_parameters']['opt']
        del ckpt

        model_args = TrainOptions.get_model_specific_args()
        for k,v in old_opt.__dict__.items():
            if (k in model_args) and (not k in opt.__dict__.keys()):
                opt.__dict__[k] = v

        ### prepare
        dm = MyDataModule(opt)
        model = create_model(opt)
        model.load_pretrained(opt.load_pretrained_model)
        trainer = MyTrainer(dm)

        ### check phase
        if opt.phase and opt.phase=='val':
            dm.setup('fit')
            trainer.predict(model, dataloaders=dm.val_dataloader())
        else:
            trainer.test(model, dm)    
    
    # gather results
    files = [find_files('', x, suffix='.npy') for x in save_dirs]
    for i in range(len(files[0])):
        for j in range(len(files)):
            if j == 0:
                npysum = np.load(files[j][i])
            else:
                npysum += np.load(files[j][i])
            if j == len(files)-1:
                npysum /= len(files)
                np.save(os.path.join(main_dir, os.path.basename(files[0][i])), npysum.astype(np.float16))
                
    # save or delete individual results
    if not eopt.save_temp_results:
        for x in save_dirs:
            shutil.rmtree(x)
            print(f'clean up {x}')
                
        
    
    
    