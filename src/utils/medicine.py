import cv2
import nibabel as nib
import numpy as np


def GetBodyArea(ct, padding=5):
    cs, ch, cw = ct.shape
    src = np.zeros([3, ch, cw])
    src[0, :, :] = ct[cs // 4 * 1, :, :]
    src[1, :, :] = ct[cs // 4 * 2, :, :]
    src[2:, :] = ct[cs // 4 * 3, :, :]
    src = src.max(axis=0)
    src = (src - src.min()) * 255 / (src.max() - src.min())
    src = np.uint8(src)

    _, src = cv2.threshold(src, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    src = cv2.morphologyEx(src, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, hu = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return 0, 0, cw, ch
    max_area = 0
    max_cnt = cnts[0]
    for c in cnts:
        area = cv2.contourArea(c)
        if max_area < area:
            max_area = area
            max_cnt = c
    if max_area < cw * ch / 4:
        return 0, 0, cw, ch
    x, y, w, h = cv2.boundingRect(max_cnt)
    w = w + padding if x + w + padding < cw else cw - x
    h = h + padding if x + h + padding < ch else ch - x
    x = x - padding if x > padding else 0
    y = y - padding if y > padding else 0

    return x, y, w, h


def cvresize(src, size,interpolation=cv2.INTER_CUBIC):
    minc = min(size)
    s = src.shape[-1] // minc
    if s != 0 and src.shape[-1] != minc:
        dst = np.zeros((size[1], size[0], src.shape[-1]))
        for i in range(s + 1):
            if i != s:
                dst[:, :, i * minc : (i + 1) * minc] = cv2.resize(
                    src[:, :, i * minc : (i + 1) * minc], size
                )
            else:
                dst[:, :, i * minc :] = cv2.resize(src[:, :, i * minc :], size)
    else:
        dst = cv2.resize(src, size,interpolation=interpolation)
    return dst


def ReSp(src, x, y, w, h, restore=False, size=512,interpolation=cv2.INTER_CUBIC):
    if restore:
        dst = np.zeros((src.shape[0], size, size), np.float64)
        src = src.transpose((1, 2, 0))
        src = cvresize(src, [w, h],interpolation)
        src = src.transpose((2, 0, 1))
        dst[:, y : y + h, x : x + w] = src
    else:
        dst = src[:, y : y + h, x : x + w]
        dst = dst.transpose((1, 2, 0))
        dst = cvresize(dst, [size, size])
        dst = dst.transpose((2, 0, 1))
    return dst


def LoadNii(path, max=None, shift=None):
    ret = nib.load(path).get_fdata()
    if max:
        ret = (ret - ret.min()) * float(max) / (ret.max() - ret.min())
    if shift:
        ret = ret + float(shift)
    return ret
