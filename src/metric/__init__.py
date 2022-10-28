import importlib
import os

from utils.registry import METRIC_REGISTRY

folder = os.path.dirname(os.path.abspath(__file__))
filenames = [
    os.path.splitext(os.path.basename(v))[0]
    for v in os.listdir(folder)
    if v.endswith('_metric.py')
]
_import_modules = [importlib.import_module(f'metric.{f}') for f in filenames]


def get_metric(type, **kwds):
    return METRIC_REGISTRY.get(type)(**kwds)
