import torch
import torch.nn.functional as F
from torch import nn


class EdgeMap(nn.Module):
    def __init__(self, num_channels=3):
        super(EdgeMap, self).__init__()

        # 定义8个梯度卷积核
        # 水平方向上的Sobel算子核
        # 其中的unsqueeze表示的是在指定维度上扩展维度大小，使用两次是把对应的图像（只有高度和宽度）扩展为4维度（batch_size,input_channels,height,width)
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        # repeat是对于指定维度的数据进行复制，这个方法目前是在通道数维度上进行复制，以便按照通道数对每一个通道的图像进行卷积操作
        self.sobel_x = self.sobel_x.repeat(1, num_channels, 1, 1)
        self.sobel_x = self.sobel_x.cuda()

        # 垂直方向上的Sobel算子核
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.sobel_y = self.sobel_y.repeat(1, num_channels, 1, 1)
        self.sobel_y = self.sobel_y.cuda()

        # 水平prewitt算子核
        self.prewitt_x = torch.tensor([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(
            0)
        self.prewitt_x = self.prewitt_x.repeat(1, num_channels, 1, 1)

        self.prewitt_x = self.prewitt_x.cuda()

        # 垂直prewitt算子核
        self.prewitt_y = torch.tensor([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(
            0)
        self.prewitt_y = self.prewitt_y.repeat(1, num_channels, 1, 1)
        self.prewitt_y = self.prewitt_y.cuda()

        # sobel主对角线方向上的边缘
        self.sobel_d1 = torch.tensor([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]], dtype=torch.float32).unsqueeze(0).unsqueeze(
            0)
        self.sobel_d1 = self.sobel_d1.repeat(1, num_channels, 1, 1)
        self.sobel_d1 = self.sobel_d1.cuda()

        # sobel副对角线上的边缘
        self.sobel_d2 = torch.tensor([[-2, -1, 0], [1, 0, -1], [0, 1, 2]], dtype=torch.float32).unsqueeze(0).unsqueeze(
            0)
        self.sobel_d2 = self.sobel_d2.repeat(1, num_channels, 1, 1)
        self.sobel_d2 = self.sobel_d2.cuda()

        # prewitt主对角线方向上的边缘
        self.prewitt_d1 = torch.tensor([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=torch.float32).unsqueeze(
            0).unsqueeze(0)
        self.prewitt_d1 = self.prewitt_d1.repeat(1, num_channels, 1, 1)
        self.prewitt_d1 = self.prewitt_d1.cuda()

        # prewitt副对角线方向上的边缘
        self.prewitt_d2 = torch.tensor([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=torch.float32).unsqueeze(
            0).unsqueeze(0)
        self.prewitt_d2 = self.prewitt_d2.repeat(1, num_channels, 1, 1)
        self.prewitt_d2 = self.prewitt_d2.cuda()

    def forward(self, x):
        # 计算8个卷积结果

        sobel_x = F.conv2d(x, self.sobel_x, padding=1)
        sobel_y = F.conv2d(x, self.sobel_y, padding=1)
        prewitt_x = F.conv2d(x, self.prewitt_x, padding=1)
        prewitt_y = F.conv2d(x, self.prewitt_y, padding=1)
        sobel_d1 = F.conv2d(x, self.sobel_d1, padding=1)
        sobel_d2 = F.conv2d(x, self.sobel_d2, padding=1)
        prewitt_d1 = F.conv2d(x, self.prewitt_d1, padding=1)
        prewitt_d2 = F.conv2d(x, self.prewitt_d2, padding=1)

        # 将8个结果合并为一个8通道的Tensor
        edge = torch.cat((sobel_x, sobel_y, prewitt_x, prewitt_y, sobel_d1, sobel_d2, prewitt_d1, prewitt_d2), dim=1)

        return edge
