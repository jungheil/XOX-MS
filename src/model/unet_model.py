import mindspore.nn as nn
import mindspore.ops.operations as F
from mindspore.common.initializer import TruncatedNormal
from mindspore.nn import CentralCrop
from utils.registry import MODEL_REGISTRY


class DoubleConv(nn.Cell):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        init_value_0 = TruncatedNormal(0.06)
        init_value_1 = TruncatedNormal(0.06)
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.SequentialCell(
            [
                nn.Conv2d(
                    in_channels,
                    mid_channels,
                    kernel_size=3,
                    has_bias=True,
                    weight_init=init_value_0,
                    pad_mode="valid",
                ),
                nn.ReLU(),
                nn.Conv2d(
                    mid_channels,
                    out_channels,
                    kernel_size=3,
                    has_bias=True,
                    weight_init=init_value_1,
                    pad_mode="valid",
                ),
                nn.ReLU(),
            ]
        )

    def construct(self, x):
        return self.double_conv(x)


class Down(nn.Cell):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.maxpool_conv = nn.SequentialCell(
            [
                nn.MaxPool2d(kernel_size=2, stride=2),
                DoubleConv(in_channels, out_channels),
            ]
        )

    def construct(self, x):
        return self.maxpool_conv(x)


class Up1(nn.Cell):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.concat = F.Concat(axis=1)
        self.factor = 56.0 / 64.0
        self.center_crop = CentralCrop(central_fraction=self.factor)
        self.print_fn = F.Print()
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        self.up = nn.Conv2dTranspose(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.relu = nn.ReLU()

    def construct(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.relu(x1)
        x2 = self.center_crop(x2)
        x = self.concat((x1, x2))
        return self.conv(x)


class Up2(nn.Cell):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.concat = F.Concat(axis=1)
        self.factor = 104.0 / 136.0
        self.center_crop = CentralCrop(central_fraction=self.factor)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        self.up = nn.Conv2dTranspose(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.relu = nn.ReLU()

    def construct(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.relu(x1)
        x2 = self.center_crop(x2)
        x = self.concat((x1, x2))
        return self.conv(x)


class Up3(nn.Cell):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.concat = F.Concat(axis=1)
        self.factor = 200 / 280
        self.center_crop = CentralCrop(central_fraction=self.factor)
        self.print_fn = F.Print()
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        self.up = nn.Conv2dTranspose(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.relu = nn.ReLU()

    def construct(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.relu(x1)
        x2 = self.center_crop(x2)
        x = self.concat((x1, x2))
        return self.conv(x)


class Up4(nn.Cell):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.concat = F.Concat(axis=1)
        self.factor = 392 / 568
        self.center_crop = CentralCrop(central_fraction=self.factor)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        self.up = nn.Conv2dTranspose(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.relu = nn.ReLU()

    def construct(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.relu(x1)
        x2 = self.center_crop(x2)
        x = self.concat((x1, x2))
        return self.conv(x)


class OutConv(nn.Cell):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        init_value = TruncatedNormal(0.06)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            has_bias=True,
            weight_init=init_value,
        )

    def construct(self, x):
        x = self.conv(x)
        return x


@MODEL_REGISTRY
class UNet(nn.Cell):
    def __init__(self, n_channels=7, n_classes=2):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up1(1024, 512)
        self.up2 = Up2(512, 256)
        self.up3 = Up3(256, 128)
        self.up4 = Up4(128, 64)
        self.outc = OutConv(64, n_classes)

    def construct(self, x):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
