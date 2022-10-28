from copy import deepcopy

import mindspore as ms
import numpy as np
from loss import get_loss
from mindspore import load_checkpoint, load_param_into_net, nn
from mindspore import ops as P
from model import get_model
from utils.registry import TRAINER_REGISTRY

from train.base_train import BaseTrain


@TRAINER_REGISTRY
class SegTrain(BaseTrain):
    def __init__(self, opt, logger):
        super(SegTrain, self).__init__(opt, logger)
        self.opt = opt

        self.loss_fn = get_loss(opt["train"])
        scheduler_opt = deepcopy(opt["train"]["scheduler"])
        scheduler_type = scheduler_opt.pop("type")
        self.scheduler = self.get_schedule(scheduler_type, **scheduler_opt)
        self.optim = self.get_optimizer(
            opt["train"]["optim"]["type"],
            self.net.trainable_params(),
            lr=self.scheduler,
        )
        # self.loss_net = nn.WithLossCell(self.net, self.loss_fn)
        # self.train_net = nn.TrainOneStepCell(self.loss_net, self.optim)
        self.train_model = ms.Model(
            network=self.net,
            loss_fn=self.loss_fn,
            optimizer=self.optim,
            metrics=self.metric,
        )

    def post_process(self, out):
        out = P.Sigmoid()(out)
        out = out.asnumpy().squeeze() * 255
        out = out.transpose((1, 2, 0))
        return out
