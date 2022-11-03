import mindspore as ms
import mindspore.ops as P
import numpy as np
from mindspore import nn
from mindspore.common import dtype as mstype
from utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY
class SegDiceMetric(nn.Metric):
    """DiceMetric"""

    def __init__(
        self,smooth=1e-5
    ):
        super(SegDiceMetric, self).__init__()
        self.dice_sum = 0.0
        self.count = 0.0
        self.clear()
        self.sigmoid = P.Sigmoid()
        self.reducesum = P.ReduceSum()
        self.smooth = smooth
        self.argmax = P.Argmax(axis=1)
        self.one_hot = P.OneHot(axis=1)
        self.cast = P.Cast()
        self.on_value, self.off_value = ms.Tensor(1.0, mstype.float32), ms.Tensor(
            0.0, mstype.float32
        )
        self.dice = nn.Dice

    def clear(self):
        """Resets the internal evaluation result to initial state."""
        self.dice_sum = 0.0
        self.count = 0.0

    def update(self, output, target):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : 'NumpyArray' or list of `NumpyArray`
            The labels of the data.
        preds : 'NumpyArray' or list of `NumpyArray`
            Predicted values.
        """
        
        output=output[0]

        # c = output.shape[1]

        if 3 == 1:
            output = self.sigmoid(output)
        else:
            output = self.argmax(output)
            output = self.one_hot(output, 3, self.on_value, self.off_value)
            

            target = self.cast(target, mstype.int32)
            target = self.one_hot(target, 3, self.on_value, self.off_value)
            
        intersection = self.reducesum(output * target)
        # TODO 分母少了平方
        dice = (2.0 * intersection + self.smooth) / (
            self.reducesum(output) + self.reducesum(target) + self.smooth
        )
        self.dice_sum += dice * output.shape[0]

        self.count += output.shape[0]

    def eval(self):
        return self.dice_sum / self.count


# @METRIC_REGISTRY
# class SegIOUMetric(nn.Metric):
#     """DiceMetric"""

#     def __init__(
#         self,
#     ):
#         super(SegIOUMetric, self).__init__()
#         self.miou_sum = 0.0
#         self.count = 0.0
#         self.clear()
#         self.sigmoid = P.Sigmoid()
#         self.reducesum = P.ReduceSum()
#         self.argmax = P.Argmax(axis=1)
#         self.one_hot = P.OneHot(axis=1)
#         self.cast = P.Cast()
#         self.on_value, self.off_value = ms.Tensor(1.0, mstype.float32), ms.Tensor(
#             0.0, mstype.float32
#         )

#     def clear(self):
#         """Resets the internal evaluation result to initial state."""
#         self.miou_sum = 0.0
#         self.count = 0.0

#     def update(self, output, target):
#         """Updates the internal evaluation result.

#         Parameters
#         ----------
#         labels : 'NumpyArray' or list of `NumpyArray`
#             The labels of the data.
#         preds : 'NumpyArray' or list of `NumpyArray`
#             Predicted values.
#         """

#         c = output.shape[1]
#         target = self.cast(target, mstype.int32)
#         target = self.one_hot(target, c, self.on_value, self.off_value)

#         output = self.argmax(output)
#         output = self.cast(output, mstype.float32)
#         intersection = self.reducesum(output * target)

#         miou = intersection / (
#             self.reducesum(output) + self.reducesum(target) - intersection
#         )
#         self.miou_sum += miou * output.shape[0]

#         self.count += output.shape[0]

#     def eval(self):
#         return self.miou_sum / self.count
