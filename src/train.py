import glob
import os
import time
from shutil import copyfile

import cv2
import mindspore as ms
import numpy as np
from mindspore import context, nn
from mindspore.common import set_seed
from zmq import device

from dataset import get_dataset
from train import get_trainer
from utils.args import parse_options
from utils.logger import LM


def init_work(opt):
    opt["work_id"] = opt["name"] + time.strftime(
        r"_%Y-%m-%d-%H-%M-%S", time.localtime()
    )
    os.makedirs(os.path.join(root_path, opt["output_dir"], "train", opt["work_id"]))
    os.makedirs(
        os.path.join(root_path, opt["output_dir"], "train", opt["work_id"], "img")
    )
    os.makedirs(
        os.path.join(root_path, opt["output_dir"], "train", opt["work_id"], "ckpt")
    )
    logger = LM(
        "root",
        log_file_path=os.path.join(
            root_path, opt["output_dir"], "train", opt["work_id"]
        ),
    )

    copyfile(
        opt['opt_path'],
        os.path.join(
            root_path,
            opt["output_dir"],
            "train",
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

#TODO copy log file
def auto_resume(opt):
    log = glob.glob(
        os.path.join(root_path, opt["output_dir"], "train", opt["name"] + "_*")
    )
    assert len(log), 'There is no log to load.'
    last_dir = sorted(log)[-2]
    ckpt = glob.glob(
        os.path.join(
            root_path,
            opt["output_dir"],
            "train",
            last_dir,
            "ckpt",
            opt["name"] + "_[0-9]*_[0-9]*.ckpt",
        )
    )
    assert len(ckpt), 'There is no ckpt to load.'
    ckpt_path = sorted(ckpt)[-1]
    epoch = int(os.path.basename(ckpt_path).split("_")[-2])
    iter = int(os.path.basename(ckpt_path).split("_")[-1][:-5])
    return ckpt_path, epoch, iter


def train_pipeline(opt):

    logger = LM("root")
    train_ds = get_dataset(opt["datasets"]["train"], True)
    val_ds = get_dataset(opt["datasets"]["val"], True)
    
    ckpt,epoch,iter=None,0,0
    if opt['auto_resume']:
        try:
            ckpt, epoch, iter = auto_resume(opt)
        except Exception as e:
            logger.warning(f'Auto resume failed! {repr(e)}')

    if ckpt:
        train = get_trainer(opt, logger, resume_epoch=epoch, resume_iter=iter)
        train.load_ckpt(ckpt)
    else:
        train = get_trainer(opt, logger)

    if opt["train"].get("load_ckpt"):
        train.load_ckpt(
            opt["train"].get("load_ckpt"),
            filter_prefix=[
                "network.conv1.conv.0",
                "moment1.conv1.conv.0",
                "moment2.conv1.conv.0",
                "network.outconv1",
                "moment1.outconv1",
                "moment2.outconv1",
            ],
        )

    train.train(train_ds, val_ds)


if __name__ == "__main__":
    root_path = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))
    opt = parse_options(root_path)
    init_work(opt)
    train_pipeline(opt)
