import importlib
from torch.utils.data import Dataset

def find_dataset_using_name(dataset_name):
    """Import the module "[dataset_name]_dataset.py". case-insensitive.
    """
    dataset_filename = "gencd.data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() and issubclass(cls, Dataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError(f'In {dataset_filename}.py, there should be a subclass of BaseDataset with class name that matches {target_dataset_name} in lowercase.')
    return dataset