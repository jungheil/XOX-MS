import mindspore.nn as nn
from mindspore import ops as P
from utils.registry import MODEL_REGISTRY


class SE(nn.Cell):
    def __init__(self, num_feat, reduction=16, dropout=1.):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.max_pool = nn.AdaptiveMaxPool2d(1)
        r_feat = max(8, num_feat // reduction)
        self.fc = nn.SequentialCell(
            nn.Conv2d(num_feat, r_feat, 1, has_bias=False),
            nn.LeakyReLU(alpha=0.2),
            nn.Conv2d(r_feat, num_feat, 1, has_bias=False),
        )
        self.sigmoid = nn.HSigmoid()
        self.dropout = nn.Dropout(dropout)

    def construct(self, x):
        # s = P.shape(x)
        # a = self.avg_pool(x).view(s[0], s[1])
        # m = self.max_pool(x).view(s[0], s[1])
        # y = (self.fc(a) + self.fc(m)).view(s[0], s[1], 1, 1)
        y = self.fc(x)
        # y = self.fc(a).view(b, c, 1, 1)
        y = self.sigmoid(y)
        y = self.dropout(y)
        return x * y.expand_as(x)


class PixelShuffle2D(nn.Cell):
    def __init__(self, scale=2):
        super().__init__()
        self.scale = scale

    def construct(self, x):
        n, c, h, w = P.shape(x)
        oc = c // self.scale**2
        out = x.view(n, oc, self.scale, self.scale, h, w)
        out = P.Transpose()(out, (0, 1, 4, 2, 5, 3))
        out = out.view(n, oc, self.scale * h, self.scale * w)
        return out


class Conv2d(nn.Cell):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        pad_mode='CONSTANT',
        padding=0,
        **kwds
    ):
        super().__init__()
        self.pad = nn.Pad(
            ((0, 0), (0, 0), (padding, padding), (padding, padding)), pad_mode
        )
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            **kwds
        )

    def construct(self, x):
        out = self.pad(x)
        out = self.conv(x)
        return out


def get_norm(type, *args, **kwds):
    if type == 'BatchNorm2d':
        return nn.BatchNorm2d(*args, **kwds)
    elif type == 'InstanceNorm2d':
        return nn.InstanceNorm2d(*args, **kwds)
    else:
        raise NotImplementedError


class ChannelShuffle(nn.Cell):
    def __init__(self):
        super().__init__()

    def construct(self, x):
        batchsize, num_channels, height, width = P.Shape()(x)
        x = P.Reshape()(x, (batchsize * num_channels // 2, 2, height * width))
        x = P.Transpose()(
            x,
            (
                1,
                0,
                2,
            ),
        )
        x = P.Reshape()(x, (2, -1, num_channels // 2, height, width))
        return x[0], x[1]


class ShuffleBLockN(nn.Cell):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, norm='BatchNorm2d', dropout=1.
    ):
        super().__init__()
        assert in_channels % 2 == 0
        mid_channels = out_channels // 2
        padding = kernel_size // 2
        output = out_channels - mid_channels

        self.channel_shuffle = ChannelShuffle()
        self.main = nn.SequentialCell(
            [
                nn.Conv2d(
                    in_channels=mid_channels,
                    out_channels=mid_channels,
                    kernel_size=1,
                    stride=1,
                    has_bias=False,
                ),
                get_norm(norm, num_features=mid_channels, momentum=0.9),
                nn.HSigmoid(),
                # dw
                Conv2d(
                    in_channels=mid_channels,
                    out_channels=mid_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    pad_mode='SYMMETRIC',
                    padding=padding,
                    group=mid_channels,
                    has_bias=False,
                ),
                get_norm(norm, num_features=mid_channels, momentum=0.9),
                # pw-linear
                nn.Conv2d(
                    in_channels=mid_channels,
                    out_channels=output,
                    kernel_size=1,
                    stride=1,
                    has_bias=False,
                ),
                get_norm(norm, num_features=output, momentum=0.9),
                nn.HSigmoid(),
                SE(output, dropout=dropout),
            ]
        )

    def construct(self, x):
        x_pass, x = self.channel_shuffle(x)
        x = self.main(x)
        return P.concat((x_pass, x), 1)


class ShuffleBLockD(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size=3, norm='BatchNorm2d'):
        super().__init__()
        padding = kernel_size // 2
        output = out_channels - in_channels

        self.channel_shuffle = ChannelShuffle()
        self.main = nn.SequentialCell(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=1,
                    stride=1,
                    has_bias=False,
                ),
                get_norm(norm, num_features=in_channels, momentum=0.9),
                nn.HSigmoid(),
                # dw
                Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=kernel_size,
                    stride=2,
                    pad_mode='SYMMETRIC',
                    padding=padding,
                    group=in_channels,
                    has_bias=False,
                ),
                get_norm(norm, num_features=in_channels, momentum=0.9),
                # pw-linear
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=output,
                    kernel_size=1,
                    stride=1,
                    has_bias=False,
                ),
                get_norm(norm, num_features=output, momentum=0.9),
                nn.HSigmoid(),
                SE(output),
            ]
        )
        self.bp = nn.SequentialCell(
            [
                Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=kernel_size,
                    stride=2,
                    pad_mode='SYMMETRIC',
                    padding=padding,
                    group=in_channels,
                    has_bias=False,
                ),
                get_norm(norm, num_features=in_channels, momentum=0.9),
                # pw-linear
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=1,
                    stride=1,
                    has_bias=False,
                ),
                get_norm(norm, num_features=in_channels, momentum=0.9),
                nn.HSigmoid(),
            ]
        )

    def construct(self, x):
        return P.concat((self.bp(x), self.main(x)), 1)


class ShuffleBLockU(nn.Cell):
    def __init__(self, in_channels, kernel_size=3, norm='BatchNorm2d'):
        super().__init__()
        assert in_channels % 4 == 0
        padding = kernel_size // 2

        output = in_channels // 4

        self.channel_shuffle = ChannelShuffle()
        self.main = nn.SequentialCell(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=1,
                    stride=1,
                    has_bias=False,
                ),
                PixelShuffle2D(),
                get_norm(norm, num_features=output, momentum=0.9),
                nn.HSigmoid(),
                # dw
                Conv2d(
                    in_channels=output,
                    out_channels=output,
                    kernel_size=kernel_size,
                    stride=1,
                    pad_mode='SYMMETRIC',
                    padding=padding,
                    group=output,
                    has_bias=False,
                ),
                get_norm(norm, num_features=output, momentum=0.9),
                # pw-linear
                nn.Conv2d(
                    in_channels=output,
                    out_channels=output,
                    kernel_size=1,
                    stride=1,
                    has_bias=False,
                ),
                get_norm(norm, num_features=output, momentum=0.9),
                nn.HSigmoid(),
                SE(output),
            ]
        )

    def construct(self, x, res):
        x = self.main(x)
        res.append(x)
        return P.concat(res, 1)


@MODEL_REGISTRY
class ShuffleUnet(nn.Cell):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        stage_channels=[88, 224, 512, 1024],
        stage_repeats=[1, 2, 2, 2, 3],
        res=[[32], [32, 64], [64, 64, 128]],
        ksize=7,
        norm='BatchNorm2d',
        dropout=1.,
    ):
        super().__init__()
        self.res = res

        self.first_conv = nn.SequentialCell(
            [
                Conv2d(
                    in_channels=in_channels,
                    out_channels=stage_channels[0],
                    kernel_size=3,
                    stride=1,
                    pad_mode='SYMMETRIC',
                    padding=1,
                    has_bias=False,
                ),
            ]
        )
        self.out_conv = nn.SequentialCell(
            [
                Conv2d(
                    in_channels=stage_channels[0],
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    pad_mode='SYMMETRIC',
                    padding=1,
                    has_bias=False,
                ),
                nn.HSigmoid(),
            ]
        )
        self.sf = nn.SequentialCell(
            [
                ShuffleBLockN(stage_channels[0], stage_channels[0], ksize, norm)
                for _ in range(stage_repeats[0])
            ]
        )

        drop = [1.] * stage_repeats[-1]
        drop[stage_repeats[-1] // 2] = dropout

        self.df = nn.SequentialCell(
            [
                ShuffleBLockN(
                    stage_channels[-1], stage_channels[-1], ksize, norm, drop[i]
                )
                for i in range(stage_repeats[-1])
            ]
        )

        self.down = nn.CellList(
            [
                ShuffleBLockD(stage_channels[i], stage_channels[i + 1], ksize, norm)
                for i in range(len(stage_channels) - 1)
            ]
        )
        self.up = nn.CellList(
            [
                ShuffleBLockU(stage_channels[-i - 1], ksize, norm)
                for i in range(len(stage_channels) - 1)
            ]
        )
        self.down_feature = []
        self.up_feature = []

        for i in range(len(stage_channels) - 1):
            self.down_feature.append(
                nn.SequentialCell(
                    [
                        ShuffleBLockN(
                            stage_channels[i + 1], stage_channels[i + 1], ksize, norm
                        )
                        for _ in range(stage_repeats[i + 1] - 1)
                    ]
                )
            )
            channels = stage_channels[-i - 1] // 4 + sum(res[-i - 1])
            self.up_feature.append(
                nn.SequentialCell(
                    [
                        ShuffleBLockN(channels, channels, ksize, norm)
                        for _ in range(stage_repeats[-i - 2] - 1)
                    ]
                )
            )
        self.down_feature = nn.CellList(self.down_feature)
        self.up_feature = nn.CellList(self.up_feature)

        self.reshape = nn.CellList()
        for i in range(len(res) - 1):
            rs = nn.CellList()
            for j in range(len(res[-i - 1]) - 1):
                c = res[-i - 1][j]
                rs.append(
                    Conv2d(
                        c,
                        c,
                        3,
                        2 * (j + 1),
                        'SYMMETRIC',
                        1,
                        group=c,
                        has_bias=False,
                    )
                )
            self.reshape.append(rs)

    def construct(self, x):
        x = self.first_conv(x)
        x = self.sf(x)

        ps = []
        for d in self.down:
            ps.append(x)
            x = d(x)

        x = self.df(x)

        for i, u in enumerate(self.up):
            p = []
            for j, r in enumerate(self.res[-i - 1]):
                if i < len(self.up) - 1 and j < len(self.reshape[i]):
                    p.append(self.reshape[i][-j - 1](ps[j][:, -r:, :, :]))
                else:
                    p.append(ps[j][:, -r:, :, :])

            x = u(x, p)
            x = self.up_feature[i](x)

        x = self.out_conv(x)

        return [x,x,x,x]
