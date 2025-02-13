import argparse

import template

parser = argparse.ArgumentParser(description='EDSR and MDSR')

parser.add_argument('--debug', action='store_true',  # debug开关
                    help='Enables debug mode')
parser.add_argument('--template', default='.',  # 您可以在option.py中设置各种模板
                    help='You can set various templates in option.py')

# 硬件设置
parser.add_argument('--n_threads', type=int, default=4,  # 使用多少个线程加载数据，默认为4
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',  # 是否只使用cpu
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,  # GPU数量，默认为1
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,  # 随机种子，默认为1
                    help='random seed')

# data 设置
parser.add_argument('--dir_data', type=str, default='./data/',  # 数据集文件夹
                    help='dataset directory')
parser.add_argument('--dir_demo', type=str, default='../DPDNN',  # demo 图片文件夹
                    help='demo image directory')
parser.add_argument('--data_train', type=str, default='DIV2K',  # 训练集名称
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='Set5',  # 测试数据集名称
                    help='test dataset name')
parser.add_argument('--benchmark_noise', action='store_true',  # 使用噪音较大的基准测试集
                    help='use noisy benchmark sets')
parser.add_argument('--n_train', type=int, default=799,  # 训练集大小
                    help='number of training set')
parser.add_argument('--n_val', type=int, default=100,  # 验证集大小
                    help='number of validation set')
parser.add_argument('--offset_val', type=int, default=908,  # validation index offest
                    help='validation index offest')
parser.add_argument('--ext', type=str, default='img',  # 数据集文件扩展名 默认为img
                    help='dataset file extension')
parser.add_argument('--scale', default='2',  # 超分辨率比例
                    help='super resolution scale')
parser.add_argument('--patch_size', type=int, default=400,  # 输出patch大小
                    help='output patch size')
parser.add_argument('--rgb_range', type=int, default=255,  # RGB的最大值
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,  # 要使用的颜色通道数量 默认为3
                    help='number of color channels to use')
parser.add_argument('--noise', type=str, default='.',  # 高斯噪声标准
                    help='Gaussian noise std.')
parser.add_argument('--chop', action='store_true',  # 启用内存高效转发
                    help='enable memory-efficient forward')

# model 设置
parser.add_argument('--model', default='ECDUNCBP',  # 使用的模型
                    help='model name')

parser.add_argument('--act', type=str, default='relu',  # 激活函数
                    help='activation function')
parser.add_argument('--precision', type=str, default='Single',  # 精密度
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')

# trainer 设置
parser.add_argument('--reset', action='store_true',  # 重置训练
                    help='reset the training')
parser.add_argument('--test_every', type=int, default=1000,  # 每N个批次进行测试
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=10,  # number of epochs to train
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=1,  # 训练输入batch大小
                    help='input batch size for training')
parser.add_argument('--split_batch', type=int, default=1,  # 将批次分成更小的块
                    help='split the batch into smaller chunks')
parser.add_argument('--self_ensemble', action='store_true',  # 使用自集成方法进行测试
                    help='use self-ensemble method for test')
parser.add_argument('--test_only', action='store_true',  # 设置此选项以测试模型
                    help='set this option to test the model')
parser.add_argument('--gan_k', type=int, default=1,  # 对抗性损失的k值
                    help='k value for adversarial loss')

# Optimization 设置
parser.add_argument('--lr', type=float, default=1e-4,  # learning rate
                    help='learning rate')
parser.add_argument('--lr_decay', type=int, default=400,  # 每N个epochs的学习率衰减，默认400
                    help='learning rate decay per N epochs')
parser.add_argument('--decay_type', type=str, default='Mstep_300_450_600',  # 学习率衰减类型
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,  # 阶跃衰减的学习速率衰减系数
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),  # optimizer to use (SGD | ADAM | RMSprop)
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,  # SGD momentum
                    help='SGD momentum')
parser.add_argument('--beta1', type=float, default=0.9,  # ADAM beta1
                    help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999,  # ADAM beta2
                    help='ADAM beta2')
parser.add_argument('--epsilon', type=float, default=1e-8,  # ADAM epsilon for numerical stability
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,  # weight_decay
                    help='weight decay')

# Loss 设置
parser.add_argument('--loss', type=str, default='1*L1',  # 损失函数配置
                    help='loss function configuration')
parser.add_argument('--skip_threshold', type=float, default='1e10',  # 跳过错误较大的批处理
                    help='skipping batch that has large error')

# Log 设置
parser.add_argument('--pre_train', type=str, default='experiment/SR_X2_BI/model/model_best.pt',  # 预训练模型位置
                    help='pre-trained model directory')
parser.add_argument('--save', type=str, default='../experiment/SR_X2_BI',  # 用于保存的文件名
                    help='file name to save')
parser.add_argument('--load', type=str, default='.',  # 用于加载的文件名
                    help='file name to load')
parser.add_argument('--resume', type=int, default=0,  # 从特定检查点恢复
                    help='resume from specific checkpoint')
parser.add_argument('--print_model', action='store_true',  # 打印模型
                    help='print model')
parser.add_argument('--save_models', action='store_true',  # 保存所有中间模型
                    help='save all intermediate models')
parser.add_argument('--print_every', type=int, default=50,  # 在记录训练状态之前需要等待多少批次
                    help='how many batches to wait before logging training status')
# parser.add_argument('--save_results', action='store_true',
parser.add_argument('--save_results', type=bool, default=True,  # 保存结果
                    help='save output results')

# options for residual group and feature channel reduction 残差组和特征通道减少的选项
parser.add_argument('--n_resgroups', type=int, default=10,  # 残差组的数量
                    help='number of residual groups')
parser.add_argument('--reduction', type=int, default=16,  # 减少特征图的数量
                    help='number of feature maps reduction')
# options for test
parser.add_argument('--testset', type=str, default='DHT',  # 用于测试的数据集名称
                    help='dataset name for testing')
#以上代码就是添加命令行选项
args = parser.parse_args()  #解析命令行参数，将参数转化为python对象，包含了所有在命令行中指定的选项和参数
template.set_template(args)

args.scale = list(
    map(lambda x: int(x), args.scale.split('+'))
)
#其中的args.scale.split('+')用于将args.scale字符串，例如“2+3+4”按照+分割，返回一个
#由子字符串组成的列表，之后将每一个字符串元素转换成整数类型，lambda表达式
#指定了每一个字符串转换成整数的操作，然后将所有的转化的整数转换成一个整数列表
if args.epochs == 0:
    args.epochs = 1e8
#将字符串类型的布尔值转化为布尔值类型
#vars(args)返回的是参数字典，而vars(args)[arg]对应是返回参数的字典对应的值
for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False
