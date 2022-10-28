import importlib
import os

from utils.registry import TRAINER_REGISTRY

folder = os.path.dirname(os.path.abspath(__file__))
filenames = [
    os.path.splitext(os.path.basename(v))[0]
    for v in os.listdir(folder)
    if v.endswith('_train.py')
]
_import_modules = [importlib.import_module(f'train.{f}') for f in filenames]


def get_trainer(opt, logger=None):
    return TRAINER_REGISTRY.get(opt['train']['type'])(opt, logger)
