from torch.functional import F
from torch import nn


CONV_KERNEL = (3, 3)
POOL_KERNEL = (2, 2)
STRIDE_SIZE = (1, 1)
PADDING_TYPE = 1


class DownBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels, **kwargs):
        super(DownBlock, self).__init__(**kwargs)

        self.conv_1 = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=CONV_KERNEL,
                                padding=PADDING_TYPE)
        self.conv_2 = nn.Conv2d(in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=CONV_KERNEL,
                                padding=PADDING_TYPE)
        self.batch_1 = nn.BatchNorm2d(num_features=out_channels)
        self.batch_2 = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=POOL_KERNEL)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.batch_1(x)
        x = self.relu(x)
        x = self.conv_2(x)
        x = self.batch_2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class CustomConvTranspose2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | tuple,
                 stride=STRIDE_SIZE,
                 padding=PADDING_TYPE,
                 output_padding=PADDING_TYPE):
        super(CustomConvTranspose2d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding - 1

    def forward(self, input, weight):
        return F.conv_transpose2d(input, weight,
                                  stride=self.stride,
                                  padding=self.padding,
                                  output_padding=self.output_padding)


def UpperBlockCustom(ConvTranspose, in_channels, out_channels):
    class UpperBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, **kwargs):
            super(UpperBlock, self).__init__(**kwargs)
            self.convT = ConvTranspose(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=CONV_KERNEL,
                                       stride=STRIDE_SIZE,
                                       padding=PADDING_TYPE)

        def forward(self, x, w=None):
            if w is not None:
                x = self.convT(x, w)
            else:
                x = self.convT(x)
            s = x.shape[2:]
            x = F.interpolate(x,
                              size=(s[0] * 2, s[1] * 2),
                              mode='bilinear',
                              align_corners=False)
            return x

    return UpperBlock(in_channels, out_channels)


class Autoencoder(nn.Module):

    def __init__(self, **kwargs):
        super(Autoencoder, self).__init__(**kwargs)
        self.down_l1 = DownBlock(in_channels=1,
                                 out_channels=64)
        self.down_l2 = DownBlock(in_channels=64,
                                 out_channels=128)
        self.down_l3 = DownBlock(in_channels=128,
                                 out_channels=256)
        self.up_l3 = UpperBlockCustom(nn.ConvTranspose2d, 256, 128)
        self.up_l2 = UpperBlockCustom(nn.ConvTranspose2d, 128, 64)
        self.up_l1 = UpperBlockCustom(nn.ConvTranspose2d, 64, 1)

    def forward(self, x):
        x = self.down_l1(x)
        x = self.down_l2(x)
        x = self.down_l3(x)

        x = self.up_l3(x)
        x = self.up_l2(x)
        x = self.up_l1(x)
        x = F.sigmoid(x)
        return x


class AutoencoderZeroDecoder(nn.Module):

    def __init__(self, **kwargs):
        super(AutoencoderZeroDecoder, self).__init__(**kwargs)
        self.down_l1 = DownBlock(in_channels=1,
                                 out_channels=64)
        self.down_l2 = DownBlock(in_channels=64,
                                 out_channels=128)
        self.down_l3 = DownBlock(in_channels=128,
                                 out_channels=256)
        self.up_l3 = UpperBlockCustom(CustomConvTranspose2d, 256, 128)
        self.up_l2 = UpperBlockCustom(CustomConvTranspose2d, 128, 64)
        self.up_l1 = UpperBlockCustom(CustomConvTranspose2d, 64, 1)

    def forward(self, x):
        x = self.down_l1(x)
        x = self.down_l2(x)
        x = self.down_l3(x)

        w1 = self.down_l1.conv_1.weight.data
        w2 = self.down_l2.conv_1.weight.data
        w3 = self.down_l3.conv_1.weight.data

        x = self.up_l3(x, w3)
        x = self.up_l2(x, w2)
        x = self.up_l1(x, w1)
        x = F.sigmoid(x)
        return x


if __name__ == "__main__":
    import torch
    model = Autoencoder()
    X = torch.randn(1, 1, 128, 128)
    Y = model(X)
    print(Y.shape)
    model = AutoencoderZeroDecoder()
    X = torch.randn(1, 1, 128, 128)
    Y = model(X)
    print(Y.shape)
