import torch
import torch.nn as nn


class UCNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UCNet, self).__init__()
        # 3x3,64
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        # 3x3,3
        self.conv2 = nn.Conv2d(out_channels, 3, kernel_size=3, stride=1, padding=1)
        self.residual_blocks = nn.Sequential(
            ResidualBlock(out_channels, out_channels),  # 3x3,64
            ResidualBlock(out_channels, out_channels),
            ResidualBlock(out_channels, out_channels)
        )

    def forward(self, x):
        out = self.conv1(x)
        # out = self.relu(out)
        residual = self.residual_blocks(out)
        out = self.conv2(residual)
        final = x - out
        return final


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = x + out
        return out
