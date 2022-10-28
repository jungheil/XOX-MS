import multiprocessing
import os
import random
from glob import glob

import mindspore.dataset as de
import numpy as np
from utils.medicine import GetBodyArea, LoadNii, ReSp
from utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY
class NIIDS:
    def __init__(self, opt):
        root = opt['path']
        if root[-1] != '/':
            root += '/'
        self.path_img = glob(root + opt['img_re'])
        self.path_seg = glob(root + opt['seg_re'])
        assert len(self.path_img) == len(self.path_seg) and len(self.path_img) != 0
        self.nii_size = len(self.path_img)
        self.slides = opt['slides']
        self.shuffle = opt['shuffle']
        assert self.slides % 2 != 0
        self.range_max = opt['range'].split('/')[0]
        self.range_shift = opt['range'].split('/')[1]
        self.resp = opt['resp']

        self.img_size = []
        for s in self.path_seg:
            seg = LoadNii(s)
            self.img_size.append(seg.shape[0] - self.slides + 1)

        self.ci = self.nii_size
        self.si = 0
        self.slides_idx = None
        self.len = sum(self.img_size)

        self.next_img(shuffle=self.shuffle)

    def next_img(self, shuffle=False):
        self.ci = self.ci + 1
        if self.ci >= self.nii_size:
            self.ci = 0
            if shuffle:
                seed = random.random()
                random.shuffle(self.path_img, lambda: seed)
                random.shuffle(self.path_seg, lambda: seed)
                random.shuffle(self.img_size, lambda: seed)

        self.cache_img = LoadNii(
            self.path_img[self.ci], self.range_max, self.range_shift
        )
        self.cache_seg = LoadNii(
            self.path_seg[self.ci], self.range_max, self.range_shift
        )
        if self.resp:
            x, y, w, h = GetBodyArea(self.cache_img)
            self.cache_img = ReSp(self.cache_img, x, y, w, h)
            self.cache_seg = ReSp(self.cache_seg, x, y, w, h)

        self.cache_img = self.cache_img.transpose((1, 2, 0))
        self.cache_seg = self.cache_seg.transpose((1, 2, 0))

        self.slides_idx = list(range(self.img_size[self.ci]))
        if shuffle:
            random.shuffle(self.slides_idx)
        self.si = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.si >= self.img_size[self.ci]:
            self.next_img(shuffle=self.shuffle)
            if self.ci == 0:
                raise StopIteration

        i = self.slides_idx[self.si]
        img = np.float32(self.cache_img[:, :, i : i + self.slides])
        seg = np.float32(self.cache_img[:, :, i + self.slides // 2])
        self.si += 1

        return img, seg

    def __len__(self):
        return self.len
