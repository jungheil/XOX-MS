import random

import numpy as np
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

class RandomRot:
    def __init__(self, prob=0.5) -> None:
        self.prob=prob
    def __call__(self, *img):
        k = random.randint(1,3)
        if random.random()>self.prob:
            return img
        img = np.rot90(img, k)
        return img