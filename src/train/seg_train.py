from copy import deepcopy

import mindspore as ms
import numpy as np
from loss import get_loss
from matplotlib import pyplot as PLT
from mindspore import nn
from mindspore import ops as P
from mindspore.common import dtype as mstype
from utils.registry import TRAINER_REGISTRY

from train.base_train import BaseTrain


@TRAINER_REGISTRY
class SegTrain(BaseTrain):
    def __init__(self, opt, logger, resume_epoch=0, resume_iter=0):
        super(SegTrain, self).__init__(opt, logger, resume_epoch, resume_iter)
        self.opt = opt

        self.loss_fn = get_loss(opt["train"])
        scheduler_opt = deepcopy(opt["train"]["scheduler"])
        scheduler_type = scheduler_opt.pop("type")
        self.scheduler = self.get_schedule(scheduler_type, **scheduler_opt)
        train_param = [
            'outconv1.weight',
            'outconv2.weight',
            'outconv3.weight',
            'outconv4.weight',
            'conv1.conv.0.weight',
            'ca1.fc.0.weight',
            'ca1.fc.2.weight',
            'ca2.fc.0.weight',
            'ca2.fc.2.weight',
            'ca3.fc.0.weight',
            'ca3.fc.2.weight',
            'ca4.fc.0.weight',
            'ca4.fc.2.weight',
        ]
        for param in self.net.trainable_params():
            if param.name not in train_param:
                param.requires_grad = False
        self.optim = self.get_optimizer(
            opt["train"]["optim"]["type"],
            self.net.trainable_params(),
            lr=self.scheduler,
        )
        # self.loss_net = SegWithLossCell(self.net, self.loss_fn)
        # self.train_net = nn.TrainOneStepCell(self.loss_net, self.optim)
        # self.eval_net = SegWithEvalCell(self.net, self.loss_fn)
        self.train_model = ms.Model(
            network=self.net,
            loss_fn=self.loss_fn,
            optimizer=self.optim,
            metrics=self.metric,
        )

    def post_process(self, out):
        out = out[0]
        seg = P.Argmax(axis=1)(out).asnumpy().squeeze()
        out = np.zeros((seg.shape[0], seg.shape[1], 3))

        # out[seg==0] = np.array(255,0,0)
        out[seg == 1] = np.array([0, 255, 0])
        out[seg == 2] = np.array([0, 0, 255])
        return out


# class SegWithLossCell(nn.Cell):
#     def __init__(self, backbone, loss_fn):
#         """输入有两个，前向网络backbone和损失函数loss_fn"""
#         super(SegWithLossCell, self).__init__(auto_prefix=False)
#         self._backbone = backbone
#         self._loss_fn = loss_fn

#         self.softmax = P.Softmax(axis=1)
#         self.sigmoid = P.Sigmoid()

#         self.one_hot = P.OneHot(axis=1)
#         self.on_value, self.off_value = ms.Tensor(1.0, mstype.float32), ms.Tensor(
#             0.0, mstype.float32
#         )

#     def construct(self, data, labels):
#         output = self._backbone(data)
#         c = output.shape[1]
#         if c == 1:
#             output = self.sigmoid(output)
#         else:
#             output = self.softmax(output)

#             labels = self.cast(labels, mstype.int32)
#             labels = self.one_hot(labels, c, self.on_value, self.off_value)

#         return self._loss_fn(output, labels)


# class SegWithEvalCell(nn.Cell):
#     def __init__(self, backbone, loss_fn):
#         """输入有两个，前向网络backbone和损失函数loss_fn"""
#         super(SegWithEvalCell, self).__init__(auto_prefix=False)
#         self._backbone = backbone
#         self._loss_fn = loss_fn

#         self.argmax = P.Argmax(axis=1)
#         self.sigmoid = P.Sigmoid()

#         self.one_hot = P.OneHot(axis=1)
#         self.on_value, self.off_value = ms.Tensor(1.0, mstype.float32), ms.Tensor(
#             0.0, mstype.float32
#         )

#     def construct(self, data, labels):
#         output = self._backbone(data)

#         if c == 1:
#             output = self.sigmoid(output)
#         else:
#             output = self.argmax(output)
#             output = self.cast(output, mstype.float32)

#             labels = self.cast(labels, mstype.int32)
#             labels = self.one_hot(labels, c, self.on_value, self.off_value)

#         return output, labels
