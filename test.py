import copy
import os
import shutil
from tqdm.autonotebook import tqdm
import torch

from gencd.data.datamodule import MyDataModule
from gencd.engine.trainer import MyTrainer
from gencd.models import create_model
from gencd.options import TrainOptions, TestOptions

if __name__ == '__main__':
    opt = TestOptions().parse()
        
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
    model.load_pretrained(opt.load_pretrained_model[0])
    trainer = MyTrainer(dm)

    ### check phase
    if opt.phase and opt.phase=='val':
        dm.setup('fit')
        trainer.predict(model, dataloaders=dm.val_dataloader())
    else:
        trainer.test(model, dm)    
    
