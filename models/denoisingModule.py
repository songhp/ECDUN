import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()

        m = []
        for i in range(2):
            # Conv2d的输入输出大小分别是前两个参数
            m.append(nn.Conv2d(n_feat, n_feat, kernel_size, padding=(kernel_size // 2), bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):  # x(n_feat) -> res(n_feat)
        res = self.body(x).mul(self.res_scale)  # 所有元素根据残差比例缩小
        res += x
        return res


class EncodingBlock(nn.Module):
    def __init__(self, ch_in):
        super(EncodingBlock, self).__init__()

        body = [
            nn.Conv2d(ch_in, 64, kernel_size=3, padding=3 // 2),
            nn.ReLU(),
            ResBlock(n_feat=64, kernel_size=3),
            ResBlock(n_feat=64, kernel_size=3),
            nn.Conv2d(64, 128, kernel_size=3, padding=3 // 2)
        ]
        self.body = nn.Sequential(*body)
        self.down = nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=3 // 2)
        self.act = nn.ReLU()

    def forward(self, input):  # input -> f_e(128),down(64)
        f_e = self.body(input)
        down = self.act(self.down(f_e))
        return f_e, down  # f_e是包含更多更全面的高级特征，而down是粗糙的特征，是f_e降维之后的特征


class EncodingBlockEnd(nn.Module):
    def __init__(self, ch_in):
        super(EncodingBlockEnd, self).__init__()

        head = [
            nn.Conv2d(in_channels=ch_in, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU()
        ]
        body = [
            ResBlock(n_feat=64, kernel_size=3),
            ResBlock(n_feat=64, kernel_size=3),
            ResBlock(n_feat=64, kernel_size=3),

            ResBlock(n_feat=64, kernel_size=3),
            ResBlock(n_feat=64, kernel_size=3),
            ResBlock(n_feat=64, kernel_size=3),

            ResBlock(n_feat=64, kernel_size=3),
            ResBlock(n_feat=64, kernel_size=3),
            ResBlock(n_feat=64, kernel_size=3),

            ResBlock(n_feat=64, kernel_size=3),
            ResBlock(n_feat=64, kernel_size=3),
            ResBlock(n_feat=64, kernel_size=3),

            ResBlock(n_feat=64, kernel_size=3),
            ResBlock(n_feat=64, kernel_size=3),
            ResBlock(n_feat=64, kernel_size=3),

            ResBlock(n_feat=64, kernel_size=3),
            ResBlock(n_feat=64, kernel_size=3),
            ResBlock(n_feat=64, kernel_size=3),
        ]
        tail = [
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=3 // 2)
        ]
        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)

    def forward(self, input):  # input -> f_e(128)
        out = self.head(input)
        f_e = self.body(out) + out
        f_e = self.tail(f_e)
        return f_e


class DecodingBlock(nn.Module):
    def __init__(self, ch_in):
        super(DecodingBlock, self).__init__()

        body = [
            nn.Conv2d(in_channels=ch_in, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(),
            ResBlock(n_feat=64, kernel_size=3),
            ResBlock(n_feat=64, kernel_size=3),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, padding=1 // 2)
        ]

        self.up = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.act = nn.ReLU()
        self.body = nn.Sequential(*body)

    def forward(self, input, map):  # input(128),map(128) -> out(256)
        # 保证逆向卷积出来的shape和map一致
        up = self.up(input, output_size=[input.shape[0], input.shape[1], map.shape[2], map.shape[3]])
        up = self.act(up)
        out = torch.cat((up, map), 1)  # 在channel 纬度上
        out = self.body(out)
        return out


# 最后一个decoding和之前的decoding的唯一区别就是结尾处多加了一个卷积层用来提高图像的通道数
class DecodingBlockEnd(nn.Module):
    def __init__(self, ch_in):
        super(DecodingBlockEnd, self).__init__()

        body = [
            nn.Conv2d(ch_in, 64, kernel_size=3, padding=3 // 2),
            nn.ReLU(),
            ResBlock(n_feat=64, kernel_size=3),
            ResBlock(n_feat=64, kernel_size=3),
        ]

        self.up = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.act = nn.ReLU()
        self.body = nn.Sequential(*body)

    def forward(self, input, map):  # input(128),map(128) -> out(64)
        # 保证逆向卷积出来的shape和map一致
        up = self.up(input, output_size=[input.shape[0], input.shape[1], map.shape[2], map.shape[3]])
        out = self.act(up)
        out = torch.cat((out, map), 1)  # 在channel 纬度上
        out = self.body(out)
        return out
