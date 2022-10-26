import copy
import os
import shutil
from tqdm.autonotebook import tqdm

from gencd.data.datamodule import MyDataModule
from gencd.engine.trainer import MyTrainer
from gencd.models import create_model
from gencd.options import TrainOptions

if __name__ == '__main__':
    opt = TrainOptions().parse()

    dm = MyDataModule(opt)
    model = create_model(opt)
    trainer = MyTrainer(dm)
    trainer.fit(model, dm)
    
    if not opt.train_only:
        # validation
        #trainer.predict(model, dataloaders=dm.val_dataloader(), ckpt_path='best')
        
        # test
        trainer.test(model, dm, ckpt_path='best')
