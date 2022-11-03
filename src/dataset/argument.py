import random
from functools import wraps

import numpy as np
from mindspore.dataset import vision as vs
from mindspore.dataset.vision import Inter
from PIL import Image
from skimage.util import random_noise


# TODO 不太行
class RandomNoise:
    def __init__(self, intensity=(0, 0.01), noise_type='mix', prob=0.25) -> None:
        self.intensity = intensity
        self.noise_type = noise_type
        self.prob = prob

    def __call__(self, img):
        if random.random() > self.prob:
            return img
        if self.noise_type == 'gaussian':
            img = random_noise(
                np.array(img), mode='gaussian', var=random.uniform(*self.intensity)
            )
        elif self.noise_type == 's&p':
            img = random_noise(
                np.array(img), mode='s&p', amount=random.uniform(*self.intensity)
            )
        elif self.noise_type == 'mix':
            if random.randint(0, 2):
                img = random_noise(
                    np.array(img), mode='gaussian', var=random.uniform(0, 0.02)
                )
            else:
                img = random_noise(
                    np.array(img), mode='s&p', amount=random.uniform(0, 0.05)
                )
        else:
            raise
        return Image.fromarray(np.uint8(img * 255))


class RandomParams:
    def __init__(self, rand_params={}, process_prob=1, **kwds) -> None:
        self.rand_params = rand_params
        self.process_prob = process_prob
        self.kwds = kwds

        self.is_run = True

    def process(*args, **kwds):
        pass

    def get_param(self, p):
        range = self.rand_params[p]
        if type(range[0]) is int:
            param = random.randint(*range)
        elif type(range[0]) is float:
            param = random.random() * (range[1] - range[0]) + range[0]
        else:
            raise NotImplementedError
        return param

    def new_params(self):
        for p in self.rand_params:
            self.kwds[p] = self.get_param(p)
        if random.random() <= self.process_prob:
            self.is_run = True
        else:
            self.is_run = False

    def __call__(self, *inputs):
        self.new_params()
        if not self.is_run:
            return inputs
        fun = self.process(**self.kwds)

        outputs = []
        for i in inputs:
            outputs.append(fun(i))
        return tuple(outputs)


class RescaleCrop:
    def __init__(self, size, scale, ratio, **kwds) -> None:
        self.rc = vs.RandomResizedCrop(size, (scale, scale), (ratio, ratio), **kwds)

    def __call__(self, img):
        return self.rc(img)


class Affine:
    def __init__(self, degrees, shift_x, shift_y, shear_x, shear_y):
        self.af = vs.RandomAffine(
            degrees=(degrees, degrees),
            translate=(shift_x, shift_x, shift_y, shift_y),
            shear=(shear_x, shear_x, shear_y, shear_y),
            resample=Inter.BICUBIC,
        )

    def __call__(self, img):
        return self.af(img)


class RandomCustom(RandomParams):
    def __init__(self, process, rand_params={}, process_prob=1, **kwds) -> None:
        super().__init__(rand_params, process_prob, **kwds)
        self.process = process


class Rot90:
    def __init__(self, k=1) -> None:
        self.k = k

    def __call__(self, img):
        img = np.rot90(img, self.k)
        return img


randrot = RandomCustom(Rot90, process_prob=0.5, rand_params={'k': (1, 3)})
