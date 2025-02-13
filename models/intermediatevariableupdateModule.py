import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PFM(nn.Module):  # 并行交叉融合模块
    def __init__(self):
        super(PFM, self).__init__()

        self.convf = nn.Conv2d(8, 32, kernel_size=3, padding=1)  # 将f映射为32通道
        self.convh = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 将h映射为32通道
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.convf_1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.convh_1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        # self.gate = nn.LSTM(64, 32, batch_first=True)  # 选取互补特征

    def forward(self, h, f):
        # print("f的shape", f.shape)   # [1,8,400,400]
        f = F.relu(self.convf(f))  # 将f映射为32通道
        # print(f.shape)
        h = F.relu(self.convh(h))  # 将h映射为32通道
        f_s = self.sigmoid(f)  # 以元素方式添加并Sigmoid激活
        h_s = self.sigmoid(h)
        h_m = h * f_s
        f_m = f * h_s
        f_p = f_m + f
        h_p = h_m + h
        f = self.convf_1(f_p)
        h = self.convh_1(h_p)
        f = self.sigmoid(f)
        h = self.tanh(h)
        # f, _ = self.gate(f.unsqueeze(0))  # 使用门机制选取互补特征
        final = f * h
        return final  # 32 channel


class NonLocalBlock(nn.Module):
    def __init__(self, in_channels):
        super(NonLocalBlock, self).__init__()

        self.inter_channels = in_channels // 2
        self.theta = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1, stride=1)
        self.phi = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1, stride=1)
        self.g = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1, stride=1)

        self.conv_out = nn.Conv2d(self.inter_channels, in_channels, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        theta = self.theta(x)
        phi = self.phi(x)
        g = self.g(x)
        theta = theta.view(theta.size(0), self.inter_channels, -1)
        phi = phi.view(phi.size(0), self.inter_channels, -1).permute(0, 2, 1)
        g = g.view(g.size(0), self.inter_channels, -1).permute(0, 2, 1)

        f = torch.matmul(theta, phi)
        f_div_c = self.softmax(f)

        f_div_c = f_div_c.permute(0, 2, 1)
        g = g.permute(0, 2, 1)
        # print("f_div_c的shape:", f_div_c.shape)
        # print("g的shape:", g.shape)
        y = torch.matmul(f_div_c, g)

        y = y.permute(0, 2, 1).contiguous()
        y = y.view(y.size(0), self.inter_channels, *x.size()[2:])
        y = self.conv_out(y)
        output = x + y
        return output


class EGIM(nn.Module):
    def __init__(self):
        super(EGIM, self).__init__()

        # self.conv = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.baseBlock = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        self.pfm = PFM()
        self.nonlocalblock = NonLocalBlock(32)

    def forward(self, h, f):
        pfm = self.pfm(h, f)
        bb_1 = self.baseBlock(pfm)
        output_up = self.nonlocalblock(bb_1)
        # -----------LSTM-------------------------------------------------------------------------
        # 先获取输入LSTM图像的各个维度的信息
        # batch_size, channel, height, width = bb_1.shape
        # # print(bb_1.shape)
        # target_size = bb_1.shape[2:]
        # # 设置特征的维度，因为输入是个序列，就是对应的这个序列的长度，作为LSTM初始化的输入参数
        # input_size = channel * width
        # # 初始化参数列表，对应的第二个参数是hidden_size，第三个参数是需要的LSTM的个数
        # lstm = nn.LSTM(input_size, 256, 2)
        # # 图像的高度或者宽度时间步的维度
        # seq_dim = height
        # # 转换维度的顺序
        # bb_1_permuted = bb_1.permute(0, 2, 1, 3)
        # bb_1_reshaped = bb_1_permuted.reshape(batch_size * seq_dim, channel, width)
        # # 将图像重塑为LSTM的输入形状
        # seq_length = batch_size * seq_dim
        # input_size = channel * width
        # lstm_input = bb_1_reshaped.reshape(seq_length, batch_size, input_size)
        # lstm = lstm.to(lstm_input.device)  # 把所有参数都转换成和输入图像一个处理设备上进行
        # output, _ = lstm(lstm_input)
        # # print(output.shape)   # [width,1,256]
        # # print("test_only", test_only)
        # if not self.training:
        #     maxpool = nn.MaxPool1d(kernel_size=2)
        #     output = maxpool(output)
        #     output_width, output_batch, output_len = output.shape
        #     # output_batch = output.shape[1]
        #     # output_len = output.shape[2]
        #     sum = output_len * output_batch * output_width
        #     output_shape = sum // channel
        #     output_shape = math.sqrt(output_shape)
        #     output_shape = int(output_shape)
        # else:
        #     # print("执行了else分支")
        #     output_shape = 20
        # output_reshaped = output.view(batch_size, channel, output_shape, output_shape)
        # # print(output_reshaped.shape)
        # output_up = F.interpolate(output_reshaped, size=target_size, mode='bilinear',
        #                           align_corners=False)
        # -------------------------------------------------------------------------------------
        bb_2 = self.baseBlock(output_up)
        bb_3 = self.conv2(bb_2)
        final = bb_3 + h

        return final
