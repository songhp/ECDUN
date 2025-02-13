import torch
import torch.nn as nn


class EAFM(nn.Module):
    def __init__(self):
        super(EAFM, self).__init__()
        # 第一个Conv层
        self.conv1 = nn.Conv2d(8, 32, kernel_size=3, padding=1)
        # 第一个BaseBlock
        self.baseblock = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Conv2d(32, 8, kernel_size=3, padding=1)
        # 第二个BaseBlock
        # self.baseblock2 = nn.Sequential(
        #     nn.Conv2d(8, 32, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(32, 32, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(32, out_channels, kernel_size=3, padding=1)
        # )

    def forward(self, z):
        # 第一个Conv层
        x = self.conv1(z)
        # 第一个BaseBlock
        out1 = x + self.baseblock(x)
        # 第二个BaseBlock
        out2 = out1 + self.baseblock(out1)
        # 第二个Conv层
        out3 = self.conv2(out2)
        # 残差连接2
        out3 += z
        # 返回更新后的特征映射
        return out3
