import mindspore as ms
from mindspore import nn
from mindspore import ops as P
from mindspore.common import dtype as mstype
from mindspore.common.initializer import One
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


@LOSS_REGISTRY
class BCEDiceLoss(LossBase):
    '''BCEDiceLoss'''

    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.bceloss = ops.BinaryCrossEntropy()
        self.sigmoid = ops.Sigmoid()
        self.reduceSum = ops.ReduceSum(keep_dims=False)

    def construct(self, predict, target):
        '''construct'''
        bce = self.bceloss(predict, target)
        smooth = 1e-5
        num = target.shape[0]
        predict = predict.view(num, -1)
        target = target.view(num, -1)
        intersection = predict * target
        dice = (2.0 * self.reduceSum(intersection, 1) + smooth) / (
            self.reduceSum(predict, 1) + self.reduceSum(target, 1) + smooth
        )
        dice = 1 - dice / num
        return 0.5 * bce + dice
