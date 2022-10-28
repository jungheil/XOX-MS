import importlib
import os
import random

import mindspore.dataset as de
import mindspore.dataset.vision as vs
from utils.registry import DATASET_REGISTRY

from dataset.argument import RandomNoise

__all__ = ['get_dataset']


folder = os.path.dirname(os.path.abspath(__file__))
filenames = [
    os.path.splitext(os.path.basename(v))[0]
    for v in os.listdir(folder)
    if v.endswith('_dataset.py')
]
_import_modules = [importlib.import_module(f'dataset.{f}') for f in filenames]


def get_dataset(opt, is_train=True):
    dataset = DATASET_REGISTRY.get(opt['type'])(opt)
    ds = de.GeneratorDataset(
        dataset,
        column_names=["img", "seg"],
        num_parallel_workers=opt['num_parallel_workers'],
    )

    if is_train:
        trans = []
        agms = []
        if opt['agm']:
            # TODO Random Rotation
            # if opt['agm'].get('rot'):
            #     trans.append(vs.RandomRotation(5))
            if opt['agm'].get('crop'):
                trans.append(
                    vs.RandomResizedCrop(
                        opt['size'], scale=(0.9, 1.0), ratio=(0.75, 1.333)
                    )
                )
            else:
                trans.append(vs.Resize([opt['size'], opt['size']]))
            if opt['agm'].get('hflip'):
                trans.append(vs.RandomHorizontalFlip(prob=0.5))
            if opt['agm'].get('vflip'):
                trans.append(vs.RandomVerticalFlip(prob=0.5))
            if opt['agm'].get('color'):
                agms.append(
                    vs.RandomColorAdjust(
                        (0.8, 1.2), (0.6, 1.4), (0.5, 1.5), (-0.05, 0.05)
                    )
                )
            if opt['agm'].get('blur'):
                agms.append(
                    vs.GaussianBlur(
                        (1, 1) if random.random() < 0.75 else (5, 5), random.random()
                    )
                )
            if opt['agm'].get('noise'):
                agms.append(RandomNoise())
            trans = de.transforms.Compose(trans)
            agms = de.transforms.Compose(agms)

            ds = ds.map(
                operations=trans,
                input_columns=['img', 'seg'],
                num_parallel_workers=opt['num_parallel_workers'],
            )
            ds = ds.map(
                operations=agms,
                input_columns=['img'],
                num_parallel_workers=opt['num_parallel_workers'],
            )
        # ds = ds.map(operations=vs.Normalize(mean=opt['mean'], std=opt['std']), input_columns=['img'], num_parallel_workers=opt['num_parallel_workers'])
        ds = ds.map(
            operations=[vs.HWC2CHW()],
            input_columns=['img'],
            num_parallel_workers=opt['num_parallel_workers'],
        )
        ds = ds.map(
            operations=[vs.HWC2CHW()],
            input_columns=['seg'],
            num_parallel_workers=opt['num_parallel_workers'],
        )
    else:
        ds = ds.map(
            operations=vs.Resize([opt['size'], opt['size']]),
            input_columns=['img', 'seg'],
            num_parallel_workers=opt['num_parallel_workers'],
        )
        # ds = ds.map(operations=vs.Normalize(mean=opt['mean'], std=opt['std']), input_columns=['img'], num_parallel_workers=opt['num_parallel_workers'])
        ds = ds.map(
            operations=[vs.HWC2CHW()],
            input_columns=['img'],
            num_parallel_workers=opt['num_parallel_workers'],
        )
        ds = ds.map(
            operations=[vs.HWC2CHW()],
            input_columns=['seg'],
            num_parallel_workers=opt['num_parallel_workers'],
        )

    ds = ds.batch(opt['batch_size'], drop_remainder=True)
    return ds
