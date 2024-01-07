import torch
import torch.nn as nn
from torchvision.models import resnet50

from utils import check_model_params_and_size


class Block(nn.Module):
    """
    Convolution Block to be used in the encoder
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, max_pool=False,
                 pool_kernel_size=2, pool_stride=2):
        super(Block, self).__init__()
        if max_pool:
            self.conv = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),  # to prevent loss of information or complete neuron deactivation
                nn.MaxPool2d(pool_kernel_size, pool_stride),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
            )

    def forward(self, x):
        x = self.conv(x)
        return x


class wideEncoder(nn.Module):
    """
    Wide Neural Network Implementation
    """

    def __init__(self, in_channels=1, out_channels=256, features=32):
        super(wideEncoder, self).__init__()

        # no batch norm for initial block
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=9, stride=4, padding=0),
            nn.LeakyReLU(),
        )

        # Downsampling - output shape: [batch_size, 256, 12, 12]
        self.down1 = Block(features, features * 4, kernel_size=7, stride=4,
                           max_pool=False)  # 32 channels -> 128 channels
        self.down2 = Block(features * 4, out_channels, kernel_size=7, stride=2,
                           max_pool=False)  # 128 channels -> 256 channels

    def forward(self, x):
        d1 = self.initial(x)
        d2 = self.down1(d1)
        wide_output = self.down2(d2)
        return wide_output


class deepEncoder(nn.Module):
    """
    Deep Neural Network Implementation
    """

    def __init__(self, in_channels=1, out_channels=256, features=8):
        super(deepEncoder, self).__init__()

        # no batch norm for initial block
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Downsampling - output shape: [batch_size, 256, 12, 12]
        self.down1 = Block(features, features * 2, max_pool=True)  # 8 channels-> 16
        self.down2 = Block(features * 2, features * 4, max_pool=True)  # 16 channels-> 32
        self.down3 = Block(features * 4, features * 8, max_pool=True)  # 32 channels-> 64
        self.down4 = Block(features * 8, features * 16, max_pool=True)  # 64 channels-> 128
        self.down5 = Block(features * 16, out_channels, max_pool=False)  # 128 channels-> 256

    def forward(self, x):
        d1 = self.initial(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        deep_output = self.down5(d5)
        return deep_output


class combinedEncoder(nn.Module):
    """
    Combined Encoder Model.
    """

    def __init__(self, in_channels=1, out_channels=256, wide_features=32, deep_features=8):
        super(combinedEncoder, self).__init__()

        self.wide_encoder = wideEncoder(in_channels, out_channels, wide_features)
        self.deep_encoder = deepEncoder(in_channels, out_channels, deep_features)

    def forward(self, x):
        wide_output = self.wide_encoder(x)
        deep_output = self.deep_encoder(x)
        combined_output = torch.cat((wide_output, deep_output), dim=1)  # output shape: [batch_size, 512, 12, 12]
        return combined_output


class resnet50Encoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=256, wide_features=32, deep_features=8):
        super(resnet50Encoder, self).__init__()
        self.mod = resnet50(weights='DEFAULT')
        del self.mod.avgpool
        del self.mod.fc
        self.conv = nn.Conv2d(2048, out_channels * 2, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.mod.conv1(x)
        x = self.mod.bn1(x)
        x = self.mod.relu(x)
        x = self.mod.maxpool(x)
        x = self.mod.layer1(x)
        x = self.mod.layer2(x)
        x = self.mod.layer3(x)
        x = self.mod.layer4(x)
        x = self.conv(x)
        return x


if __name__ == '__main__':
    # model = combinedEncoder()
    # ip = torch.rand(1, 1, 512, 512)
    model = resnet50Encoder()
    ip = torch.rand(1, 3, 512, 512)
    op = model(ip)
    print(f"Input shape: {ip.shape}, Output Shape: {op.shape}")
    check_model_params_and_size(model)
