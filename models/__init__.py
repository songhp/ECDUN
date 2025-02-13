import os
from importlib import import_module

import torch
import torch.nn as nn  # 神经网络模块


class Model(nn.Module):
    # 构造函数，进行模型的初始化
    def __init__(self, args, ckp):  # ckp是一个checkpoint对象，包含了训练到哪一个epoch以及训练过程中的准确度和其他指标
        super(Model, self).__init__()
        print('Making model...')

        self.scale = args.scale
        self.idx_scale = 0
        self.self_ensemble = args.self_ensemble  # 是否采用自集成技术
        self.chop = args.chop  # 是否采用分块技术
        self.precision = args.precision  # 是否采用混合精度，如果为True则采用混合精度
        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.n_GPUs = args.n_GPUs  # 用于指定GPU的个数
        self.save_models = args.save_models

        # 加载对应的整体模型
        module = import_module('models.' + args.model.lower())
        # 调用对应的make_model函数返回整体的模型，并进行初始化，然后把他加载到GPU上进行训练,
        # 最后的实例化对象赋值给self.model参数
        # 1.这是原本加载过来的模型
        self.model = module.make_model(args).to(self.device)
        # 2.这是是否缩减精度为半精度后的模型
        if args.precision == 'half': self.model.half()  # 精度为半精度，GPU支持半精度16位，可提高计算速度
        # 3.这是是否对模型进行并行化处理后的模型
        if not args.cpu and args.n_GPUs > 1:
            # 对模型进行并行化处理，最后处理好的并行化模型可直接用于后续训练和测试
            self.model = nn.DataParallel(self.model, range(args.n_GPUs))

        # 调用自定义的load函数用于加载已保存的权重模型
        self.load(
            ckp.dir,  # 表示模型保存的路径
            pre_train=args.pre_train,  # 如果为TRUE表示从预训练模型中加载权重
            resume=args.resume,  # 如果为TRUE则从最近一次检查点中恢复模型
            cpu=args.cpu  # 是否将模型加载到CPU上
        )
        if args.print_model:
            print(self.model)

        # 前向传播

    def forward(self, x, idx_scale):
        self.idx_scale = idx_scale  # 输入数据放大尺度
        target = self.get_model()  # 获得模型对象
        if hasattr(target, 'set_scale'):  # 检查当前target对象中是否有set_scale这个属性
            target.set_scale(idx_scale)
        # self_ensemble是否进行自我集成，也就是对输入图像进行多次缩放和SR重构，
        # 然后取平均值作为最终输出，从而减少图像中的伪像和噪声
        # self.training表示当前不处于模型的训练模式，也就是测试模式，
        # 1.因为要自我集成，所以对应的forward_x8就是自我集成版本的forward函数
        if self.self_ensemble and not self.training:    # 测试阶段
            if self.chop:  # 需要砍掉图像的边缘进行多次局部缩放
                forward_function = self.forward_chop
            else:
                forward_function = self.model.forward
            # self.forward_x8实现自我集成操作，将输入的LR图像进行
            # 水平翻转、垂直翻转、对角线反转、水平和垂直方向镜像翻转，
            # 然后分别进行前向计算，最后的结果取平均值
            return self.forward_x8(x, forward_function)
        # self.ensemble和self.chop的虽然都是集成方法，但是区别是前者是基于不同的训
        # 练数据集来构建集成模型的，而后者是基于同一个训练集进行不同的裁剪方式来构建
        # 集成模型的
        # 2.因为要通过chop来进行集成，所以对应的forward_chop就是chop版本的forward函数
        elif self.chop and not self.training:
            return self.forward_chop(x)
        # 3.当前处于训练阶段，那么对应的这两个if语句都不会被执行，而只是简单的返回原本的模型
        else:
            return self.model(x)

    def get_model(self):
        if self.n_GPUs == 1:
            return self.model
        else:
            # 当GPU个数大于1的时候，对应的self.model是一个nn.DataParallel对象，
            # 也就是进行并行计算的实例对象，此时需要再调用module才是实际的模型对象
            return self.model.module

    def state_dict(self, **kwargs):  # 保存模型状态
        target = self.get_model()
        return target.state_dict(**kwargs)

    def save(self, apath, epoch, is_best=False):  # 其中apath表示保存路径的根目录
        target = self.get_model()
        # torch.save(
        #    target.state_dict(),
        #    os.path.join(apath, 'model', 'model_latest.pt')
        # )
        if is_best:
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model', 'model_best.pt')
            )

        if self.save_models:
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model', 'model_{}.pt'.format(epoch))
            )

    def load(self, apath, pre_train='.', resume=-1, cpu=False):
        # 其中pre_train是可选择的预训练模型位置，对应的.表示当前路径，也就是不加载任何预训练模型
        # resume是指定加载的模型状态（-1最近一次，0预训练模型，>0历史模型状态
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        if resume == -1:
            self.get_model().load_state_dict(
                torch.load(
                    os.path.join(apath, 'model', 'model_latest.pt'),
                    **kwargs
                ),
                strict=False
            )
        elif resume == 0:
            if pre_train != '.':  # pre_train!=.表示不是当前路径，说明是有预训练模型的
                print('Loading model from {}'.format(pre_train))
                self.get_model().load_state_dict(
                    torch.load(pre_train, **kwargs),
                    strict=False
                    # 表示在加载预训练模型状态时，可以忽略模型架构不匹配等问题，从而获得更好
                    # 的预测效果
                )
        else:
            self.get_model().load_state_dict(
                torch.load(
                    os.path.join(apath, 'model', 'model_{}.pt'.format(resume)),
                    **kwargs
                ),
                strict=False
            )

    def forward_chop(self, x, shave=10, min_size=160000):  # shave是预留的边缘像素的大小
        scale = self.scale[self.idx_scale]
        n_GPUs = min(self.n_GPUs, 4)
        b, c, h, w = x.size()  # batch_size,channel,height,width
        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave
        lr_list = [
            x[:, :, 0:h_size, 0:w_size],  # 左上部分
            x[:, :, 0:h_size, (w - w_size):w],  # 右上部分
            x[:, :, (h - h_size):h, 0:w_size],  # 左下部分
            x[:, :, (h - h_size):h, (w - w_size):w]]  # 右下部份

        if w_size * h_size < min_size:  # 当切分后的图片小于最小设置的值，就可以使用模型直接进行超分辨率处理
            sr_list = []  # 用于存储处理后的图片
            for i in range(0, 4, n_GPUs):
                # 将每一块分好的图像按照dim=0，即batch的维度，进行拼接，具体的batch大小为n_GPUS
                lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
                # 将每一个batch的图像传入模型进行处理
                sr_batch = self.model(lr_batch)
                # 因为之前是按照batch维度将对应的图像块拼接起来的，现在是将拼接起来的处理好的图像batch再分开
                sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
        else:  # 因为对裁剪的大小并不小于预设值因此，重新递归调用forward_chop函数，
            # 其中反斜杠是对应的所有的处理过的图像块拼接在一起
            sr_list = [
                self.forward_chop(patch, shave=shave, min_size=min_size) \
                for patch in lr_list
            ]

        h, w = scale * h, scale * w
        h_half, w_half = scale * h_half, scale * w_half
        h_size, w_size = scale * h_size, scale * w_size
        shave *= scale

        output = x.new(b, c, h, w)  # 新建一个和原来的图像大小相同的图像，方便下文的像素的填充
        # 对应将第一个大的图像块的每一个小像素块填充到对应的输出图像的左上角的每一个小像素块上，下同
        output[:, :, 0:h_half, 0:w_half] \
            = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        return output

    def forward_x8(self, x, forward_function):
        def _transform(v, op):
            if self.precision != 'single': v = v.float()
            # 将不是单精度的张量转化为单精度张量
            # 将转换好类型的数据从GPU上复制到内存，并转化为numpy数组
            v2np = v.data.cpu().numpy()
            if op == 'v':  # 垂直翻转
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':  # 水平翻转
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':  # 转置
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            if self.precision == 'half': ret = ret.half()  # 转化为半精度浮点类型

            return ret

        # 调用自己函数中定义过的transform函数对原始的图像数据集进行一系列翻转操作
        lr_list = [x]
        for tf in 'v', 'h', 't':
            lr_list.extend([_transform(t, tf) for t in lr_list])
        # 然后调用前向传播函数进行SR处理
        sr_list = [forward_function(aug) for aug in lr_list]
        # 根据不同的索引值，对于处理过的SR图像进行垂直水平的旋转操作
        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        output = output_cat.mean(dim=0, keepdim=True)  # 对处理过的所有SR图像取平均值

        return output
