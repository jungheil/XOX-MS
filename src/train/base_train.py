import copy
import os

from metric import get_metric
from mindspore import load_checkpoint, load_param_into_net, nn
from model import get_model
from utils.common import init_weights

from train.callback import StepLossTimeMonitor
from train.schedule import *


class BaseTrain:
    """Base model."""

    def __init__(self, opt, logger, resume_epoch=0, resume_iter=0):
        self.opt = opt
        self.logger = logger
        self.resume_epoch = resume_epoch
        self.resume_iter = resume_iter

        self.is_train = opt["phase"] == "train"

        self.net = get_model(opt["model"])
        self.metric = {}

        if self.is_train:
            self.max_epoch = opt["train"]["max_epoch"]
            init_weights(self.net, opt['train']['init_weights'])
            val = opt["train"].get("val")
            if val:
                for m in val:
                    val[m] = val[m] if val[m] else {}
                    p = copy.copy(val[m])
                    t = p.pop('type')
                    self.metric[m] = get_metric(t, **p)
        else:
            val = opt["eval"]['val']
            for m in val:
                val[m] = val[m] if val[m] else {}
                p = copy.copy(val[m])
                t = p.pop('type')
                self.metric[m] = get_metric(t, **p)

        self.train_model = None
        self.train_cb = None

    def create_cb(self):
        self.train_cb = [
            StepLossTimeMonitor(self.logger, self.opt["output"], self.post_process)
        ]

    def train(self, train_ds, val_ds):
        self.create_cb()


        self.logger.info('[train] Training started.')
        self.logger.info(
            f'[train] Start epoch: {self.resume_epoch}; Final epoch: {self.max_epoch}\n'
            + f'LossFunction: {self.opt["train"]["loss"]["type"]}\n'
            + f'Optimazer: {self.opt["train"]["optim"]["type"]}\n'
        )

        self.train_model.fit(
            self.max_epoch,
            train_ds,
            val_ds,
            callbacks=self.train_cb,
            dataset_sink_mode=False,
            valid_dataset_sink_mode=False,
            initial_epoch=self.resume_epoch,
            valid_frequency=self.opt['output']['eval_freq'],
        )

    def eval(self, val_ds):
        self.create_cb()
        
        self.train_model.eval(
            val_ds, callbacks=self.train_cb, dataset_sink_mode=False
        )

    def get_optimizer(self, type, params, lr, **kwds):
        if type == "SGD":
            optimizer = nn.SGD(params=params, learning_rate=lr, **kwds)
        elif type == "Adagrad":
            optimizer = nn.Adagrad(params=params, learning_rate=lr, **kwds)
        elif type == "RMSProp":
            optimizer = nn.RMSProp(params=params, learning_rate=lr, **kwds)
        elif type == "Adam":
            optimizer = nn.Adam(params=params, learning_rate=lr, **kwds)
        elif type == "AdamW":
            optimizer = nn.AdamWeightDecay(params=params, learning_rate=lr, **kwds)
        else:
            raise NotImplementedError(f"optimizer {type} is not supperted yet.")
        return optimizer

    def get_schedule(self, type, **kwds):
        warmup_step = kwds.get("warmup")
        try:
            kwds.pop("warmup")
        except:
            pass

        if type == "None":
            schedule = ConstantLR(kwds["lr"])
        elif type == "MultiStepLR":
            lr = kwds["lr"]
            milestones = kwds["milestones"]
            decay = kwds["decay"]
            schedule = MultiStepLR(lr, milestones, decay)
        elif type == "CosineDecayLR":
            schedule = nn.CosineDecayLR(**kwds)
        else:
            raise NotImplementedError(f"schedule {type} is not supperted yet.")

        if warmup_step:
            schedule = PluginWarmUpLR(warmup_step, schedule)
        return PluginResumeLR(self.resume_iter, schedule)

    def load_ckpt(self, path, **kwds):
        self.logger.info(f'[load ckpt] Loading {os.path.basename(path)}.')
        load_param_into_net(self.net, load_checkpoint(path, **kwds))

    def post_process(self, img):
        return img


class EmptyWithEvalCell(nn.Cell):

    def __init__(self, network):
        super(EmptyWithEvalCell, self).__init__(auto_prefix=False)
        self.network = network

    def construct(self, data, label):
        output = self.network(data)
        return output, label