import importlib
import sys
import pytorch_lightning as pl

def find_model_using_name(model_name):
    """Import the module "[model_name]_model.py". case-insensitive.
    """
    model_filename = f"gencd.models.{model_name}_model"
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower(): #\
           #and issubclass(cls, pl.LightningModule):
            model = cls

    if model is None:
        print(f'In {model_filename}.py, there should be a subclass of pl.LightningModule with class name that matches {target_model_name} in lowercase.')
        sys.exit(0)

    return model

def create_model(opt):
    model = find_model_using_name(opt.model)
    instance = model(opt)
    print(f'model [{type(instance).__name__}] was created')
    return instance