import mindspore as ms
from mindspore.common import dtype as mstype
from mindspore.nn.learning_rate_schedule import LearningRateSchedule
from mindspore.ops import operations as P


class MultiStepLR(LearningRateSchedule):
    def __init__(self, lr, milestones, decay):
        super(MultiStepLR, self).__init__()
        if not isinstance(lr, float):
            raise TypeError(
                "For 'MultiStepLR', the argument 'lr' must be type of float, "
                "but got 'lr' type: {}.".format(type(lr))
            )

        self.lr = lr
        self.milestones = ms.Tensor(milestones, mstype.float32)
        self.decay = decay

        self.min = P.Minimum()
        self.lessequal = P.LessEqual()
        self.sum = P.CumSum()
        self.cast = P.Cast()

    def construct(self, global_step):
        p = self.cast(self.min(global_step, self.milestones[-1]), mstype.float32)
        step = self.sum(
            self.cast(self.lessequal(self.milestones, p), ms.dtype.int32), 0
        )[-1]
        return self.lr * self.decay**step


class PluginWarmUpLR(LearningRateSchedule):
    def __init__(self, steps, schedule):
        super(PluginWarmUpLR, self).__init__()

        self.steps = float(steps)
        self.schedule = schedule
        self.min = P.Minimum()

    def construct(self, global_step):
        warmup_step = self.min(global_step, self.steps)
        return warmup_step / self.steps * self.schedule(global_step)


class ConstantLR(LearningRateSchedule):
    def __init__(self, lr):
        super(ConstantLR, self).__init__()
        if not isinstance(lr, float):
            raise TypeError(
                "For 'MultiStepLR', the argument 'lr' must be type of float, "
                "but got 'lr' type: {}.".format(type(lr))
            )
        self.lr = lr

    def construct(self, global_step):
        return self.lr


class PluginResumeLR(LearningRateSchedule):
    def __init__(self, step, schedule):
        super(PluginResumeLR, self).__init__()

        self.step = step
        self.schedule = schedule

    def construct(self, global_step):
        return self.schedule(self.step + global_step)