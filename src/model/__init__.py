import importlib
import os

from utils.logger import LM
from utils.registry import MODEL_REGISTRY

folder = os.path.dirname(os.path.abspath(__file__))
filenames = [
    os.path.splitext(os.path.basename(v))[0]
    for v in os.listdir(folder)
    if os.path.basename(v) not in ['__init__.py'] and v.endswith('.py')
]
_import_modules = [importlib.import_module(f'model.{f}') for f in filenames]


def get_model(opt):
    logger = LM('root')
    model_type = opt.pop('type')
    model = MODEL_REGISTRY.get(model_type)(**opt)

    params_size = 0
    for p in model.get_parameters():
        params_size += p.size

    logger.info(
        f'[model] Loading model {model_type}\n'
        + f'Net structure: \n{model}\n'
        + f'Net Parameters: {params_size}'
    )
    return model
