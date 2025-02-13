import gc

import torch

import loss
import models
import mydata
import utility
from option import args
from trainer import Trainer

# from torchstat import stat
# import torchvision.models as models

torch.manual_seed(args.seed)  # 设置初始化参数的随机数种子，确保每次运行代码生成的随机序列相同
checkpoint = utility.checkpoint(args)  # 检查点用于将训练过的进度信息、模型参数等信息存入磁盘，以便在需要的时候恢复训练进度
# os.environ['CUDA_VISIBLE_DEVICES'] = '4,2,3,5'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# 自动内存管理，强制执行垃圾回收
gc.collect()
# 对torch在GPU上的数据进行内存回收
torch.cuda.empty_cache()
# 用于判断之前训练好的模型是否可用
if checkpoint.ok:
    loader = mydata.Data(args)  # 调用mydata模块下的
    # 创建数据加载器（既有训练加载器，也有测试加载器，加载模型所需要的数据
    # print(loader)
    model = models.Model(args, checkpoint)  # 用checkpoint中存储的模型参数初始化模型，否则就使用默认的参数初始化模型
    # 如果是测试模型则不进行损失函数的计算
    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    # 将数据加载器、模型、损失函数传入训练其中
    t = Trainer(args, loader, model, loss, checkpoint)  # 实例化初始化参数之后可以调用对应的对象的方法
    while not t.terminate():
        # args.test_only = False
        t.train()  # 模型训练
        # args.test_only = True
        # print("test_only_before:", args.test_only)  # False
        total_params = sum(p.numel() for p in model.parameters())
        print("总的参数数量：", total_params)
        t.test()  # 模型测试
        # print("test_only_after:",args.test_only)
    #  stat(model,(3,64,64))
    checkpoint.done()  # 模型训练结束之后将训练结果保存到checkpoint当中
