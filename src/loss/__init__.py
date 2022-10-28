from utils.registry import LOSS_REGISTRY

from loss.loss import *


def get_loss(opt):
    loss = LOSS_REGISTRY.get(opt['loss']['type'])()
    return loss
