import torch
import torch.nn as nn
import torch.nn.functional as F
import math


from models.intermediatevariableupdateModule import NonLocalBlock

class PFM_1(nn.Module):  # 并行交叉融合模块
    def __init__(self):
        super(PFM_1, self).__init__()

        self.convh = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 将h映射为32通道
        self.convr = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 将r映射为32通道
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.convh_1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.convr_1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        # self.gate = nn.LSTM(64, 32, batch_first=True)  # 选取互补特征

    def forward(self, r, h):
        # print("h的shape", h.shape)
        h = F.relu(self.convh(h))  # 将h映射为32通道
        # print(h.shape)
        r = F.relu(self.convr(r))  # 将h映射为32通道
        h_s = self.sigmoid(h)  # 以元素方式添加并Sigmoid激活
        r_s = self.sigmoid(r)
        r_m = r * h_s
        h_m = h * r_s
        h_p = h_m + h
        r_p = r_m + r
        h = self.convh_1(h_p)
        r = self.convr_1(r_p)
        h = self.sigmoid(h)
        r = self.tanh(r)
        # f, _ = self.gate(f.unsqueeze(0))  # 使用门机制选取互补特征
        final = r * h
        return final  # 32 channel




class IGRM(nn.Module):
    def __init__(self):
        super(IGRM, self).__init__()

        self.pfm = PFM_1()
        self.nonlocalblock = NonLocalBlock(32)

        self.conv = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        self.baseBlock = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        # self.lstm = nn.LSTM(32, 32, batch_first=True)
        # self.lstm = SR_LSTM()

    def forward(self, r, h):
        # print("h的shape", h.shape)
        p = self.pfm(r, h)
        bb_1 = self.baseBlock(p)
        # output_up = self.nonlocalblock(bb_1)
        # ------------------------------------LSTM块-------------------------------------------
        # 先获取输入LSTM图像的各个维度的信息
        batch_size, channel, height, width = bb_1.shape
        target_size =bb_1.shape[2:]
        # 设置特征的维度，因为输入是个序列，就是对应的这个序列的长度，作为LSTM初始化的输入参数
        input_size = channel * width
        # 初始化参数列表，对应的第二个参数是hidden_size，第三个参数是需要的LSTM的个数
        lstm = nn.LSTM(input_size, 256, 2)
        # 图像的高度或者宽度时间步的维度
        seq_dim = height
        # 转换维度的顺序
        bb_1_permuted = bb_1.permute(0, 2, 1, 3)
        bb_1_reshaped = bb_1_permuted.reshape(batch_size * seq_dim, channel, width)
        # 将图像重塑为LSTM的输入形状
        seq_length = batch_size * seq_dim
        input_size = channel * width
        lstm_input = bb_1_reshaped.reshape(seq_length, batch_size, input_size)
        # 把所有参数都转换成和输入图像一个处理设备上
        lstm = lstm.to(lstm_input.device)
        output, _ = lstm(lstm_input)
        if not self.training:
            maxpool = nn.MaxPool1d(kernel_size=2)
            output = maxpool(output)
            output_width, output_batch, output_len = output.shape
            # output_batch = output.shape[1]
            # output_len = output.shape[2]
            sum = output_len * output_batch * output_width
            output_shape = sum // channel
            output_shape = math.sqrt(output_shape)
            output_shape = int(output_shape)
        else:
            output_shape = 20
        output_reshaped = output.view(batch_size, channel, output_shape, output_shape)
        output_up = F.interpolate(output_reshaped, size=target_size, mode='bilinear',
                                  align_corners=False)
        # -------------------------------------------------------------------------------

        bb_2 = self.baseBlock(output_up)
        con = self.conv(bb_2)
        final = con + r

        return final
