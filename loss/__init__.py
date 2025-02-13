import os
from importlib import import_module

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

matplotlib.use('Agg')   #将图像渲染为PNG/PDF等格式。而不会显示在屏幕上


class Loss(nn.modules.loss._Loss):
    def __init__(self, args, ckp):
        super(Loss, self).__init__()

        print('Preparing loss function:')

        self.n_GPUs = args.n_GPUs
        self.loss = []  # 存储字典{type,weight,function}的列表
        self.loss_module = nn.ModuleList()
        self.log = torch.Tensor()  # loss-log   #用于存储训练和验证损失
        device = torch.device('cpu' if args.cpu else 'cuda')

        # step-1 筹备loss用于存储定制loss函数列表
        for loss in args.loss.split('+'):
            # 获取weight，loss_type
            weight, loss_type = loss.split('*') #对应传入参数格式就是weight*loss_type
            # MSE
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            # L1
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            # VGG
            elif loss_type.find('VGG') >= 0:
                module = import_module('loss.vgg')
                loss_function = getattr(module, 'VGG')(
                    loss_type[3:],  #把VGG去掉
                    rgb_range=args.rgb_range    #图像像素点的亮度范围
                )
            # GAN
            elif loss_type.find('GAN') >= 0:
                module = import_module('loss.adversarial')  #这里的adversarial类是一个抽象基类，只定义接口不提供具体实现
                loss_function = getattr(module, 'Adversarial')(
                    args,
                    loss_type
                )
            # 整理到loss列表中
            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function
            })
            if loss_type.find('GAN') >= 0:  #因为GAN模型还涉及到判别器，所以如果是GAN还需要额外添加一个判别器损失
                self.loss.append({'type': 'DIS', 'weight': 1, 'function': None})
        #如果列表中的损失函数不止一个，那么就需要记录所有的损失函数的总损失
        if len(self.loss) > 1:  # 损失函数不止一个，添加一个total损失函数用于记录总的损失函数
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})
            #其中type表示是总损失，weight表示当前该类型的损失函数在总损失函数中的权重
        # step-2 将loss中的loss_function载入到loss_module中
        for l in self.loss:
            if l['function'] is not None:   #所有已经被具体实现的损失函数
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])  #将其添加到损失函数模型中去，用于计算总的损失函数

        # step-3 将模型存储到device中
        self.loss_module.to(device)

        # step-4 配置模型精度
        if args.precision == 'half':
            self.loss_module.half()

        # step-5 确定不使用CPU并且GPU数量大于1，使用nn.DataParallel函数来用多个GPU来加速训练
        if not args.cpu and args.n_GPUs > 1:
            self.loss_module = nn.DataParallel(
                self.loss_module, range(args.n_GPUs)
            )
        # step-6 如果存在已有配置，进行加载
        if args.load != '.':    #.表示当前目录，如果不是当前目录就是需要加载指定目录下的预训练模型
            self.load(ckp.dir, cpu=args.cpu)

    def forward(self, sr, hr):
        """计算多loss函数加权loss
        :param sr: 超分辨率后图像
        :param hr: 原始高分辨率图像
        :return: 混合加权loss
        """
        losses = []
        # 遍历所有loss的函数，并且记录effective_loss到losses列表中
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                loss = l['function'](sr, hr)  # 将sr与hr带入loss函数，获取loss
                effective_loss = l['weight'] * loss  # effective_loss = weight * loss
                losses.append(effective_loss)
                # item 将一个零维张量effective_loss转换成浮点数，将effective_loss记录到log中
                self.log[-1, i] += effective_loss.item()    #最后一行，第i列
            elif l['type'] == 'DIS':
                # 因为额外定义了一个用于记录总的损失函数的损失函数，这个总的损失函数位于判别器损失函数之后，所以需要对应坐标-1才可以获得对应的鉴别器损失函数
                self.log[-1, i] += self.loss[i - 1]['function'].loss
        # 将losses集合中不同loss函数计算得到的loss，求和
        loss_sum = sum(losses)
        if len(self.loss) > 1:
            self.log[-1, -1] += loss_sum.item()
        return loss_sum

    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'): #如果对应遍历的损失函数模块又学习率的更新模块，就调用对应的模块
                l.scheduler.step()

    def start_log(self):
        """开始记录loss-log 每一条记录的大小为shape=(1, loss函数的数量)"""
        self.log = torch.cat(
            (self.log, torch.zeros(1, len(self.loss)))
        )

    def end_log(self, n_batches):
        """结束记录loss-log"""
        self.log[-1].div_(n_batches)

    def display_loss(self, batch):
        """展示当前loss情况，除去了batch
        :return: str
        """
        n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.log[-1]):
            log.append('[{}: {:.4f}]'.format(l['type'], c / n_samples))

        return ''.join(log)

    def plot_loss(self, apath, epoch):
        """ 绘制各个函数loss-epoch曲线，存储为pdf
        :param apath: 文件存储位置
        :param epoch: epoch
        """
        axis = np.linspace(1, epoch, epoch)
        for i, l in enumerate(self.loss):  # 遍历loss函数列表
            label = '{} Loss'.format(l['type'])
            fig = plt.figure()
            plt.title(label)
            plt.plot(axis, self.log[:, i].numpy(), label=label)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig('{}/loss_{}.pdf'.format(apath, l['type']))
            plt.close(fig)

    def get_loss_module(self):
        """返回loss函数
        :return: loss_module
        """
        if self.n_GPUs == 1:
            return self.loss_module
        else:
            return self.loss_module.module

    def save(self, apath):
        """存储模型的当前状态，以及loss-log"""
        torch.save(self.state_dict(),  # 存储这个模型的当前状态
                   os.path.join(apath, 'loss.pt'))
        torch.save(self.log,  # 存储loss-log
                   os.path.join(apath, 'loss_log.pt'))

    def load(self, apath, cpu=False):
        if cpu:  # 如果需要将文件加载到cpu中，需要格外设置
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}
        # 在Pytorch中构建好一个模型后，一般需要进行预训练权重中加载。torch.load_state_dict()函数就是用于将预训练的参数权重加载到新的模型之中
        self.load_state_dict(
            # torch.load()的作用：从文件加载用torch.save()保存的对象。
            torch.load(os.path.join(apath, 'loss.pt'), **kwargs)
        )
        # 载入loss-log
        self.log = torch.load(os.path.join(apath, 'loss_log.pt'))
        # 依据log的数量更新学习率
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                for _ in range(len(self.log)):
                    l.scheduler.step()  # 学习率更新
