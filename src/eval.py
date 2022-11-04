import os
import time
from shutil import copyfile

import mindspore as ms
from mindspore import context
from mindspore.common import set_seed

from dataset import get_dataset
from train import get_trainer
from utils.args import parse_options
from utils.logger import LM


def init_work(opt):
    opt["work_id"] = opt["name"] + time.strftime(
        r"_%Y-%m-%d-%H-%M-%S", time.localtime()
    )
    os.makedirs(os.path.join(root_path, opt["output_dir"], "test", opt["work_id"]))
    os.makedirs(
        os.path.join(root_path, opt["output_dir"], "test", opt["work_id"], "img")
    )
    logger = LM(
        "root",
        log_file_path=os.path.join(
            root_path, opt["output_dir"], "test", opt["work_id"]
        ),
    )

    copyfile(
        opt['opt_path'],
        os.path.join(
            root_path,
            opt["output_dir"],
            "test",
            opt["work_id"],
            os.path.basename(opt['opt_path']),
        ),
    )

    set_seed(opt["seed"])
    context.set_context(mode=context.GRAPH_MODE, device_target=opt["device"])
    # context.set_context(mode=context.PYNATIVE_MODE, device_target=opt['device'])
    context.set_context(enable_graph_kernel=False)
    msg = r"""
==================================================
╭━╮╭━┳━━━┳━╮╭━╮╱╱╭━╮╭━┳━━━╮
╰╮╰╯╭┫╭━╮┣╮╰╯╭╯╱╱┃┃╰╯┃┃╭━╮┃
╱╰╮╭╯┃┃╱┃┃╰╮╭╯╱╱╱┃╭╮╭╮┃╰━━╮
╱╭╯╰╮┃┃╱┃┃╭╯╰╮╭━━┫┃┃┃┃┣━━╮┃
╭╯╭╮╰┫╰━╯┣╯╭╮╰╋━━┫┃┃┃┃┃╰━╯┃
╰━╯╰━┻━━━┻━╯╰━╯╱╱╰╯╰╯╰┻━━━╯
"""
    msg += (
        f'\nMindspore Version: {ms.__version__}\n'
        + f'Device Target: {ms.get_context("device_target")}\n'
        # + f'Max Memory: {ms.get_context("max_device_memory")}\n'
        + f'Mindspore Mode: {ms.get_context("mode")}\n\n'
        + f'Task Name: {opt["name"]}\n'
        + f'Seed: {opt["seed"]}\n'
        + '==================================================\n'
    )

    logger.info(msg)


def eval_pipeline(opt):

    logger = LM("root")
    ds_name = [i for i in opt["datasets"] if i.startswith('val_')]
    val_dss = [get_dataset(opt["datasets"][i], False) for i in ds_name]

    train = get_trainer(opt, logger)
    train.load_ckpt(
        opt["eval"]["load_ckpt"],
    )
    
    logger.info('[eval] Evaluation begins.')
    for i, ds in enumerate(val_dss):
        logger.info(f'[eval] validation {ds_name[i]}.')
        train.eval(ds)
    logger.info('[eval] Evaluation is end.')



if __name__ == "__main__":
    root_path = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))
    opt = parse_options(root_path, False)
    init_work(opt)
    eval_pipeline(opt)
