import torch
import torch.nn as nn
import torchvision.transforms as transforms

from models.denoisingModule import EncodingBlock, EncodingBlockEnd, DecodingBlock, DecodingBlockEnd
from models.residualprojectionModule import UCNet
from models.textureReconstructionModule import ConvDown, ConvUp
from models.edgemap import EdgeMap
from models.edgefeatureextractionModule import EAFM
from models.intermediatevariableupdateModule import EGIM
from models.variableguidereconstructionModule import IGRM
import torch.nn.functional as F


def make_model(args, parent=False):
    return EDDUN(args)


class EDDUN(nn.Module):
    def __init__(self, args):
        super(EDDUN, self).__init__()

        # -------降噪块-------------------------------
        self.channel0 = args.n_colors  # channel的数量
        self.up_factor = args.scale[0]  # 放大倍数
        self.down_factor = args.scale[0]
        self.patch_size = args.patch_size
        self.batch_size = int(args.batch_size / args.n_GPUs)
        # 降噪块
        self.Encoding_block1 = EncodingBlock(64)
        self.Encoding_block2 = EncodingBlock(64)
        self.Encoding_block3 = EncodingBlock(64)
        self.Encoding_block4 = EncodingBlock(64)

        self.Encoding_block_end = EncodingBlockEnd(64)

        self.Decoding_block1 = DecodingBlock(256)
        self.Decoding_block2 = DecodingBlock(256)
        self.Decoding_block3 = DecodingBlock(256)
        self.Decoding_block4 = DecodingBlock(256)

        self.feature_decoding_end = DecodingBlockEnd(256)
        # ReLU激活层的实例化
        self.act = nn.ReLU()
        # 二维卷积的实例化，输入通道64，输出通道3，卷积核大小3x3
        self.construction = nn.Conv2d(64, 3, 3, padding=1)

        G0 = 64
        kSize = 3
        T = 4
        self.Fe_e = nn.ModuleList(  # 特征提取模块，通过卷积提取初始图像的特征
            [nn.Sequential(
                *[
                    nn.Conv2d(3, G0, kSize, padding=(kSize - 1) // 2, stride=1),
                    nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)
                ]
            ) for _ in range(T)]
        )

        self.RNNF = nn.ModuleList(
            [nn.Sequential(
                *[
                    nn.Conv2d((i + 2) * G0, G0, 1, padding=0, stride=1),
                    nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1),
                    self.act,
                    nn.Conv2d(64, 3, 3, padding=1)
                ]
            ) for i in range(T)]
        )

        self.Fe_f = nn.ModuleList(
            [nn.Sequential(
                *[
                    nn.Conv2d((2 * i + 3) * G0, G0, 1, padding=0, stride=1)
                ]
            ) for i in range(T - 1)]
        )

        # ----------------------纹理重构模块-----------------------------------------------
        self.eta = nn.ParameterList([nn.Parameter(torch.tensor(0.5)) for _ in range(T)])
        self.mu = nn.ParameterList([nn.Parameter(torch.tensor(0.1)) for _ in range(T)])
        self.delta = nn.ParameterList([nn.Parameter(torch.tensor(0.1)) for _ in range(T)])
        self.delta_1 = nn.ParameterList([nn.Parameter(torch.tensor(0.5)) for _ in range(T)])
        self.delta_2 = nn.ParameterList([nn.Parameter(torch.tensor(0.5)) for _ in range(T)])
        self.delta_3 = nn.ParameterList([nn.Parameter(torch.tensor(0.5)) for _ in range(T)])
        self.gama = nn.Parameter(torch.tensor(0.01))
        self.conv_up = ConvUp(3, self.up_factor)
        self.conv_down = ConvDown(3, self.up_factor)
        # ----------------------残差投影块---------------------------------------------------
        # 模糊核
        self.blur = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1,
                              padding=1, bias=False)
        # 下采样
        # self.down_sample = nn.MaxPool2d(kernel_size=self.up_factor + 1, stride=1)
        self.UCNet = UCNet(3, 64)
        self.delta_down = nn.Conv2d(3, 3, kernel_size=3, stride=16, padding=1)  # 对应下采样率修改为8
        self.linear_layer = nn.Linear(3, 3)
        # -------------------边缘引导块----------------------------------------------------
        self.input_down = nn.Conv2d(3, 3, kernel_size=3, stride=8, padding=1)
        self.f_down = nn.Conv2d(8, 8, kernel_size=3, stride=8, padding=1)
        self.y_down = nn.Conv2d(3, 3, kernel_size=3, stride=4, padding=1)
        self.edgemap = EdgeMap()
        self.EAFM = EAFM()
        self.EGIM = EGIM()
        self.IGRM = IGRM()
        # self.test_only = args.test_only

    def forward(self, y):  # [batch_size ,3 ,7 ,270 ,480] ;
        # if self.test_only:
        #     print("test_only:", self.test_only)
        # print(y.shape)  #[batch_size,3,200,200]
        fea_list = []  # 保存每一个阶段的在输入denoising module之前的初始提取的特征
        V_list = []  # 用于保存每一层RNNF模块的输入特征图
        outs = []  # 用于保存每一个阶段被所有的模块处理过的图像x
        x_texture = []  # 用于保存每一层的texture module的输出图像
        delta_list = []  # 用于保存每一个阶段的补偿

        f_init = []
        x_init = []
        v_init = []
        if not self.training:  # 测试模式
            y = F.interpolate(y, size=(256, 256), mode='bilinear', align_corners=False)

        x_texture.append(torch.nn.functional.interpolate(
            # 原始的低分辨率图像y插值得到初始的高分辨率图像x
            y, scale_factor=self.up_factor, mode='bilinear', align_corners=False))
        target_size = y.shape[2:]
        # x_texture[0]是双三次上采样的x，f(0)通过edgemap进行初始化，v(0)=x(0)
        # #--------------------Initial module--------------------------------------------------
        x = x_texture[0]  # 上采样之后的初始SR图像
        # print("x的shape是", x.shape)  # [batch_size,3,400,400]
        # 1.网络输入初始化
        f_init.append(self.edgemap(x))  # 初始化f(0) channel=8
        x_init.append(x)  # 初始化x(0) channel=3
        v_init.append(x)  # 初始化v(0) channel=3
        z_ls = []
        t_ls = []
        r_ls = []
        # #-------------------------------------------------------------------------------------
        # #--------------------------------------把denoising拿到外边-----------------------------
        fea = self.Fe_e[0](x_init[0])
        encode0, down0 = self.Encoding_block1(fea)
        encode1, down1 = self.Encoding_block2(down0)
        encode2, down2 = self.Encoding_block3(down1)
        encode3, down3 = self.Encoding_block4(down2)
        media_end = self.Encoding_block_end(down3)

        decode3 = self.Decoding_block1(media_end, encode3)
        decode2 = self.Decoding_block2(decode3, encode2)
        decode1 = self.Decoding_block3(decode2, encode1)
        decode0 = self.feature_decoding_end(decode1, encode0)

        decode0 = self.construction(self.act(decode0))
        # #-------------------------------------------------------------------------------------
        for i in range(len(self.Fe_e)):
            # --------------------denoising module---------------------------------------------
            # fea = self.Fe_e[i](x_init[i])
            # fea_list.append(fea)
            # if i != 0:  # 如果不是第一个特征图，就把它和之前处理过的特征图按照通道的方向级联起来
            #     fea = self.Fe_f[i - 1](torch.cat(fea_list, 1))
            # # print("fea的shape是", fea.shape)  # [batch_size,64,400,400]
            # # 1.encoding块-----------------------------------------------------------------------
            # encode0, down0 = self.Encoding_block1(fea)
            # # print("down0:", down0.shape)  # [batch_size,64,200,200]
            # # print("encode0:", encode0.shape)  # [batch_size,128,400,400]
            # encode1, down1 = self.Encoding_block2(down0)
            # # print("down1", down1.shape)  # [batch_size,64,100,100]
            # # print("encode1", encode1.shape)  # [batch_size,128,200,200]
            # encode2, down2 = self.Encoding_block3(down1)
            # # print("down2", down2.shape)  # [batch_size,64,50,50]
            # # print("encode2", encode2.shape)  # [batch_size,128,100,100]
            # encode3, down3 = self.Encoding_block4(down2)
            # # print("down3", down3.shape)  # [batch_size,64,25,25]
            # # print("encode3", encode3.shape)  # [batch_size,128,50,50]
            #
            # media_end = self.Encoding_block_end(down3)
            # # print("media_end的shape", media_end.shape)  # [batch_size,128,25,25]
            #
            # # 利用原始图像提取出来的高级图像的粗糙特征去在编码器中处理，可以进一步给粗糙特征降噪，
            # # 然后利用提取出来的高级特征和经过解码器处理的粗糙特征进一步在通道上融合，得到media_end，
            # # 来进一步增加图像的特征。之后，再通过解码器进行解码，然后将之前融合了一次的特征和之前每一个
            # # 维度上提取到的高级特征融合起来，来增加图像特征【通过decoding来实现】
            # # 2.decoding块---------------------------------------------------------------------
            # decode3 = self.Decoding_block1(media_end, encode3)
            # decode2 = self.Decoding_block2(decode3, encode2)
            # decode1 = self.Decoding_block3(decode2, encode1)
            # decode0 = self.feature_decoding_end(decode1, encode0)
            #
            # fea_list.append(decode0)  # 去噪之后的特征图
            # V_list.append(decode0)
            # if i == 0:  # 对于初始的降噪后的特征图，通过添加RELU激活函数和卷积层进一步的提取特征
            #     decode0 = self.construction(self.act(decode0))
            # else:
            #     decode0 = self.RNNF[i - 1](torch.cat(V_list, 1))
            x_init[i] = x_init[i] + decode0
            input_target = x_init[i].shape[2:]
            # print("input_target的shape", input_target)   # [400,400]
            y_target = y.shape[2:]  # [200,200]
            # 在原始图像中加入这些提取到的降噪后的特征[原本网络的用于更新变量v，因为后期不再需要v，
            # 所以把他作为临时变量，而后期的x有需要，所以用列表来保存每一个阶段的更新后的x]
            # v = x_texture[i] + decode0
            # print("v:"+str(v.max()))
            # # --------------------texture module[这块要删除]--------------------------------------
            # x_texture.append(x_texture[i] - self.delta[i] * (
            #         self.conv_up(self.conv_down(x) - y) + self.eta[i] * (x - v)))
            # # -----------------------edge guided module---------------------------------------------
            # 下采样输入
            x_init[i] = self.input_down(x_init[i])
            v_init[i] = self.input_down(v_init[i])
            f_init[i] = self.f_down(f_init[i])
            y = self.y_down(y)
            # 1.EAFM模块
            z_ls.append(f_init[i] - self.delta_1[i] * (f_init[i] - self.edgemap(x_init[i])))
            f_init.append(self.EAFM(z_ls[i]))
            # 2.EGIM模块
            t_ls.append(v_init[i] - self.delta_2[i] * (v_init[i] - x_init[i]))
            v_init.append(self.EGIM(t_ls[i], f_init[i + 1]))
            # 3.IGRM模块
            # print("x_init的shape", x_init[i].shape)  # [1,3,50,50]
            # print("y的shape", y.shape)  # [1,3,50,50]
            # print("conv_down的shape", self.conv_down(x_init[i]).shape)  # [1,3,25,25]
            # print("conv_up的shape", self.conv_up(self.conv_down(x_init[i])).shape)  # [1,3,50,50]
            temp_size = self.conv_down(x_init[i]).shape[2:]
            temp_y = F.interpolate(y, size=temp_size, mode='bilinear', align_corners=False)
            # print("temp_y的shape:", temp_y.shape)  # [1,3,25,25]
            r_ls.append(x_init[i] - self.delta_3[i] * (
                    self.conv_up(self.conv_down(x_init[i]) - temp_y) + self.mu[i] * (v_init[i + 1] - x_init[i])))
            x_init.append(self.IGRM(r_ls[i], v_init[i + 1]))
            # 上采样输出
            x_init[i + 1] = F.interpolate(x_init[i + 1], size=input_target, mode='bilinear',
                                          align_corners=False)
            v_init[i + 1] = F.interpolate(v_init[i + 1], size=input_target, mode='bilinear',
                                          align_corners=False)
            f_init[i + 1] = F.interpolate(f_init[i + 1], size=input_target, mode='bilinear',
                                          align_corners=False)
            y = F.interpolate(y, size=y_target, mode='bilinear', align_corners=False)
            # #-----------------------RPM module------------------------------------------------------
            blurred_x = self.blur(x_init[i + 1])  # x进行模糊核处理
            # 下采样处理之后的模糊核到和y一样的大小
            down_out = F.interpolate(blurred_x, size=target_size, mode='bilinear',
                                     align_corners=False)
            difference = y - down_out
            delta_uc = F.interpolate(difference, scale_factor=self.up_factor, mode='bilinear', align_corners=False)
            # print(delta_uc.shape)  # [batch_size,3,400,400]
            delta_down = self.delta_down(delta_uc)  # 对应记得调整下采样的stride为8
            # print("delta_down", delta_down.shape)
            delta = self.UCNet(delta_down)
            # print("delta:", delta.shape)  # [batch_size,3,400,400]
            texture_size = x_init[i + 1].shape[2:]
            delta_up = F.interpolate(delta, size=texture_size, mode='bilinear',
                                     align_corners=False)  # 上采样之后方便后期和原始的图像x进行求和
            # print("delta_up:", delta_up.shape)
            # print("x_texture：", x_texture[i + 1].shape)
            # delta_up = self.linear_layer(delta_up.view(1, 3, -1))
            delta_list.append(delta_up)
            x = x_init[i + 1] + delta_up
            # #--------------------next stage update--------------------------------------------------
            outs.append(x)
            # x[i + 1] = x

        return x
