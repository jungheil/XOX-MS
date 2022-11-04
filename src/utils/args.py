import argparse
import random
from collections import OrderedDict

import yaml


def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def parse_options(root_path, is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--opt', type=str, required=True, help='Path to option YAML file.'
    )
    parser.add_argument('--auto_resume', default=False, action='store_true')

    args = parser.parse_args()

    # parse yml to dict
    with open(args.opt, mode='r') as f:
        opt = yaml.load(f, Loader=ordered_yaml()[0])

    # random seed
    seed = opt.get('seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['seed'] = seed

    opt['auto_resume'] = args.auto_resume
    opt['opt_path'] = args.opt
    
    if is_train:
        opt['phase'] = 'train'
    else:
        opt['phase'] = 'eval'
        opt['datasets']['common']['shuffle']=False
        
    opt['output']['phase'] = opt['phase']
        
    opt['root_path'] = root_path
    opt['output']['name'] = opt['name']
    opt['output_dir'] = 'log'
    if opt['output'].get('save_img') == 'inf':
        opt['output']['save_img'] = float('inf')

    # datasets common option
    ds_common = opt['datasets'].get('common')
    if ds_common:
        for k in opt['datasets']:
            if k == 'common':
                continue
            ds = opt['datasets'][k]
            for p in ds_common:
                if p not in ds:
                    ds[p] = ds_common[p]
    for k in opt['datasets']:
        if k == 'common':
            continue
        mean = opt['datasets'][k].get('mean')
        if isinstance(mean, (int, float)):
            opt['datasets'][k]['mean'] = [mean] * opt['datasets'][k]['slides']
        std = opt['datasets'][k].get('std')
        if isinstance(std, (int, float)):
            opt['datasets'][k]['std'] = [std] * opt['datasets'][k]['slides']
            
    if is_train and not opt['datasets']['val']:
        opt['output']['eval_freq'] = 0
        
    return opt
