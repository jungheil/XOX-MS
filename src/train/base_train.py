from loss import get_loss
from metric import get_metric
from mindspore import load_checkpoint, load_param_into_net, nn
from model import get_model
from utils.logger import LM

from train.callback import StepLossTimeMonitor
from train.schedule import *


class BaseTrain:
    """Base model."""

    def __init__(self, opt, logger):
        self.opt = opt
        self.is_train = opt["phase"] == "train"
        self.max_epoch = opt["train"]["max_epoch"]
        self.logger = logger
        self.resume_epoch = 0
        self.net = get_model(opt["model"])
        self.train_model = None
        self.train_cb = None

        self.metric = {}
        val = opt["train"].get("val")
        if val:
            for m in val:
                val[m] = val[m] if val[m] else {}
                self.metric[m] = get_metric(m, **val[m])

    def create_cb(self):
        self.train_cb = [
            StepLossTimeMonitor(self.logger, self.opt["output"], self.post_process)
        ]

    def train(self, train_ds, val_ds):
        self.create_cb()

        params_size = 0
        for p in self.train_model.predict_network.get_parameters():
            params_size += p.size

        self.logger.info('Training started.')
        self.logger.info(
            f'Net Structure: \n{self.train_model.predict_network}\n'
            + f'Net Parameters: {params_size}\n'
            + f'Start epoch: {self.resume_epoch}; Final epoch: {self.max_epoch}\n'
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

        warmup_step = kwds.get("warmup")
        if warmup_step:
            schedule = PluginWarmUpLR(warmup_step, schedule)
        return schedule

    def load_ckpt(self, path, **kwds):
        load_param_into_net(self.net, load_checkpoint(path, **kwds))

    def post_process(self, img):
        return img

    def resume(self, epoch):
        self.resume_epoch = epoch