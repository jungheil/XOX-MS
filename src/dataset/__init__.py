import importlib
import os
import random

import mindspore.dataset as de
import mindspore.dataset.vision as vs
from mindspore.dataset.vision import Inter
from utils.registry import DATASET_REGISTRY

from dataset.argument import (Affine, RandomCustom, RandomNoise, RescaleCrop,
                              Rot90)

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
            if opt['agm'].get('rot90'):
                trans.append(
                    RandomCustom(Rot90, process_prob=0.5, rand_params={'k': (1, 3)})
                )
            # if opt['agm'].get('rot'):
            #     trans.append(RandomCustom(vs.Rotate,rand_params={'degrees':(-5.,5.)},resample=Inter.BICUBIC))
            if opt['agm'].get('affine'):
                trans.append(
                    RandomCustom(
                        Affine,
                        rand_params={
                            'degrees': (-5.0, 5.0),
                            'shift_x': (0.9, 1.0),
                            'shift_y': (0.9, 1.0),
                            'shear_x': (0,20),
                            'shear_y': (0,20),
                        },
                    )
                )
            if opt['agm'].get('crop'):
                trans.append(
                    RandomCustom(
                        RescaleCrop,
                        size=opt['size'],
                        rand_params={'scale': (0.8, 1.2), 'ratio': (0.75, 1.333)},
                        interpolation=Inter.BICUBIC,
                    )
                )
            else:
                trans.append(vs.Resize([opt['size'], opt['size']]))
            if opt['agm'].get('hflip'):
                trans.append(RandomCustom(vs.HorizontalFlip, process_prob=0.5))
            if opt['agm'].get('vflip'):
                trans.append(RandomCustom(vs.VerticalFlip, process_prob=0.5))
            if opt['agm'].get('color'):
                agms.append(vs.RandomColorAdjust((0.85, 1.15), (0.85, 1.15)))
            if opt['agm'].get('blur'):
                agms.append(
                    vs.GaussianBlur(
                        (1, 1) if random.random() < 0.9 else (3, 3), random.random()
                    )
                )
            if opt['agm'].get('noise'):
                agms.append(RandomNoise())
            # TODO DATA ARGUMENTATION
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
