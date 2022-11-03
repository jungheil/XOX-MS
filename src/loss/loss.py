from cmath import log
from inspect import Parameter
from math import gamma
import mindspore as ms
from mindspore import nn
from mindspore import ops as P
from mindspore.common import dtype as mstype
from mindspore.nn import LossBase
from utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY
class SegCrossEntropy(LossBase):
    def __init__(self):
        super(SegCrossEntropy, self).__init__()
        self.cast = P.Cast()
        self.reshape = P.Reshape()
        self.sum = P.ReduceSum()

        self.one_hot = P.OneHot(axis=1)
        self.on_value, self.off_value = ms.Tensor(1.0, mstype.float32), ms.Tensor(
            0.0, mstype.float32
        )

        self.ce = nn.CrossEntropyLoss(reduction='none', label_smoothing=0.1)

        self.transpose = P.Transpose()
        self.softmax = P.Softmax(axis=1)
        self.argmax = P.Argmax(axis=1)
        self.not_equal = P.NotEqual()
        self.bor = P.BitwiseOr()
        self.mul = P.Mul()
        self.div = P.RealDiv()
        self.greater = P.Greater()
        self.max = P.Maximum()
        self.reduce_max = P.ReduceMax()

    def construct(self, logits, labels):
        smooth = 1e-5
        c = logits.shape[1]

        labels = self.cast(labels, mstype.int32)

        logits = self.softmax(logits)
        pred = self.argmax(logits)
        mask_p = self.not_equal(pred, 0)
        mask_l = self.not_equal(labels, 0)
        mask_p = self.cast(mask_p, mstype.float32)
        mask_l = self.cast(mask_l, mstype.float32)
        mask = self.max(mask_p, mask_l)

        labels = self.one_hot(labels, c, self.on_value, self.off_value)
        labels = labels[:, 1:, :, :]
        logits = logits[:, 1:, :, :]
        loss = self.ce(logits, labels)
        loss = self.mul(loss, mask)
        loss = self.div(self.sum(loss), self.sum(mask))

        logits = self.transpose(logits, (0, 2, 3, 1)).view((-1, c - 1))
        labels = self.transpose(labels, (0, 2, 3, 1)).view((-1, c - 1))

        intersection = logits * labels
        dice = (2.0 * self.sum(intersection) + smooth) / (
            self.sum(logits) + self.sum(labels) + smooth
        )
        dice = 1 - dice
        return loss + dice


# TODO mask
class FocalLoss(LossBase):
    def __init__(
        self, gamma=2, label_smoothing=0.0, reduction='mean', mask=True
    ) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction='none', label_smoothing=label_smoothing)
        self.gamma = gamma
        self.reduction = reduction
        self.mask = mask
        self.sum = P.ReduceSum()
        self.mean = P.ReduceMean()
        self.exp = P.Exp()
        self.cast = P.Cast()
        self.ne = P.NotEqual()

    def construct(self, logits, labels):
        # if self.mask:
        mask = labels[:,1:,...].sum(axis=1)
        mask = self.ne(mask, 0)
        mask = self.cast(mask, mstype.float32)
        
        loss = self.ce(logits, labels)
        pt = self.exp(-loss)
        weight = (1 - pt) ** self.gamma
        loss = weight * loss

        if self.reduction == 'none':
            pass
        elif self.reduction == 'sum':
            if self.mask:
                loss = self.sum(loss*mask)
            else:
                loss = self.sum(loss)
        elif self.reduction == 'mean':
            if self.mask:
                loss = self.sum(loss*mask) / self.sum(mask)
            else:
                loss = self.mean(loss)
        else:
            raise NotImplementedError
        return loss


class DiceLoss(LossBase):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = 1e-5
        self.sum = P.ReduceSum()
        self.transpose = P.Transpose()

    def construct(self, logits, labels):
        c = logits.shape[1]
        logits = self.transpose(logits, (0, 2, 3, 1)).view((-1, c))
        labels = self.transpose(labels, (0, 2, 3, 1)).view((-1, c))
        intersection = logits * labels
        dice = (2.0 * self.sum(intersection) + self.smooth) / (
            self.sum(logits) + self.sum(labels) + self.smooth
        )
        dice = 1 - dice
        return dice


class MSSSIMLoss(LossBase):
    def __init__(self,ignore_indiex=0):
        super().__init__()
        self.msssim = nn.MSSSIM(max_val=1.0, k1=0.01**2, k2=0.03**2)
        # self.msssim.set_grad(False)
        self.mean = P.ReduceMean()
        # self.ignore_indiex = ignore_indiex

    def construct(self, logits, labels):
        logits = logits[:,1:,...]
        labels = labels[:,1:,...]
        loss = self.msssim(logits, labels)
        # loss = self.msssim(logits, labels)
        loss = 1 - self.mean(loss)
        return loss


@LOSS_REGISTRY
class SegHybridLoss(LossBase):
    def __init__(self):
        super().__init__()
        self.fl = FocalLoss(reduction='mean')
        self.ssim = MSSSIMLoss()
        self.dice = DiceLoss()

        self.softmax = P.Softmax(axis=1)
        self.sigmoid = P.Sigmoid()

        self.one_hot = P.OneHot(axis=1)
        self.on_value, self.off_value = ms.Tensor(1.0, mstype.float32), ms.Tensor(
            0.0, mstype.float32
        )

    def construct(self, logits, labels):
        c = logits.shape[1]
        if c == 1:
            logits = self.sigmoid(logits)
        else:
            logits = self.softmax(logits)
            labels = self.cast(labels, mstype.int32)
            labels = self.one_hot(labels, c, self.on_value, self.off_value)

        return (
            self.fl(logits, labels)
            + self.ssim(logits, labels)
            + self.dice(logits, labels)
        )
        
        
@LOSS_REGISTRY
class DSHybridLoss(LossBase):
    def __init__(self):
        super().__init__()
        self.fl = FocalLoss(reduction='mean')
        # self.fl = nn.FocalLoss()
        self.ssim = MSSSIMLoss()
        # self.dice = DiceLoss()
        self.dice = nn.MultiClassDiceLoss(ignore_indiex=0, activation=None)

        self.softmax = P.Softmax(axis=1)
        self.sigmoid = P.Sigmoid()

        self.one_hot = P.OneHot(axis=1)
        self.on_value, self.off_value = ms.Tensor(1.0, mstype.float32), ms.Tensor(
            0.0, mstype.float32
        )
        
    def construct(self, logits, labels):
        loss = ms.Tensor(0,ms.float32)
        
        # l = len(logits)
        # c = logits[0].shape[1]
        
        labels = self.cast(labels, mstype.int32)
        labels = self.one_hot(labels, 3, self.on_value, self.off_value)
        
        weight = [0.4,0.3,0.2,0.1]
        
        for i in range(4):
            out = logits[i]

            out = self.softmax(out)

            loss += (
                self.fl(out, labels) +  self.dice(out, labels)
                + self.ssim(out, labels)
            )*weight[i]
        return loss