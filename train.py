import copy
import os
import shutil
from tqdm.autonotebook import tqdm
import wandb

from monai.utils import set_determinism

from gencd.data.datamodule import MyDataModule
from gencd.engine.trainer import MyTrainer
from gencd.models import create_model
from gencd.options import TrainOptions

if __name__ == '__main__':
    topt = TrainOptions().parse()

    if topt.seed_determinism:
        if topt.seed_determinism < 0:
            set_determinism()
        else:
            set_determinism(seed=topt.seed_determinism)
    
    if topt.fold and (not -1 in topt.fold):
        for f in topt.fold:
            opt = copy.deepcopy(topt)
            opt.fold = f            
            savedir = os.path.join(opt.save_dir, f'fold{f}')
            os.makedirs(savedir, exist_ok=True)
            opt.save_dir = savedir        
            opt.wandb_name = os.sep.join(savedir.split(os.sep)[-3:])
            
            dm = MyDataModule(opt)
            model = create_model(opt)
            trainer = MyTrainer(dm)

            if opt.resume_from_checkpoint:
                trainer.fit(model, dm, ckpt_path=opt.resume_from_checkpoint)
            else:
                trainer.fit(model, dm)

            if not opt.train_only:
                # validation
                #trainer.predict(model, dataloaders=dm.val_dataloader(), ckpt_path='best')

                # test
                trainer.test(model, dm, ckpt_path='best')
                
            wandb.finish()            
            
    else:    
        opt = copy.deepcopy(topt)
        dm = MyDataModule(opt)
        model = create_model(opt)
        trainer = MyTrainer(dm)

        if opt.resume_from_checkpoint:
            trainer.fit(model, dm, ckpt_path=opt.resume_from_checkpoint)
        else:
            trainer.fit(model, dm)

        if not opt.train_only:
            # validation
            #trainer.predict(model, dataloaders=dm.val_dataloader(), ckpt_path='best')

            # test
            trainer.test(model, dm, ckpt_path='best')
