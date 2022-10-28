import importlib
import os

from utils.registry import MODEL_REGISTRY

folder = os.path.dirname(os.path.abspath(__file__))
filenames = [
    os.path.splitext(os.path.basename(v))[0]
    for v in os.listdir(folder)
    if os.path.basename(v) not in ['__init__.py'] and v.endswith('.py')
]
_import_modules = [importlib.import_module(f'model.{f}') for f in filenames]


def get_model(opt, logger=None):
    model_type = opt.pop('type')
    return MODEL_REGISTRY.get(model_type)(**opt)
