import os
from decimal import Decimal

import torch
from tqdm import tqdm

import utility
import time
import torch.nn.functional as F


class Trainer:
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale  # 缩放尺度

        self.ckp = ckp  # checkpoint

        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        # model属性存储之前Model通过make_model实例化的EGDUN对象
        self.model = my_model
        # loss属性存储损失函数的实例化对象
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)  # 优化器
        self.scheduler = utility.make_scheduler(args, self.optimizer)  # 调度器

        if self.args.load != '.':  # 不等于表示需要加载优化器的状态字典
            self.optimizer.load_state_dict(
                torch.load(
                    os.path.join(ckp.dir, 'optimizer.pt')
                )
            )
            for _ in range(len(ckp.log)):  # 根据日志，也就是训练过程中的步数进行学习率调整
                self.scheduler.step()

        self.error_last = 1e8

    def test(self):
        epoch = self.scheduler.last_epoch + 2
        self.ckp.write_log('Evaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        # torch.nn.module的内置方法，将切换模型的评估模式为评估模式
        self.model.eval()
        # self.args.test_only = True
        # print("self.args.test_only:", self.args.test_only)  #True
        # 因为是测试集，所以对应的自动求导机制被关闭，with创建了一个不会自动求导的上下文环境
        # 一旦跳出了with这个上下文代码块，自动求导机制会被重新激活
        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                eval_acc = 0
                self.loader_test.dataset.set_scale(idx_scale)
                tqdm_test = tqdm(self.loader_test, ncols=80)  # 进度条显示
                for idx_img, (lr, hr, filename, _) in enumerate(tqdm_test):
                    filename = filename[0]  # 测试数据集名称
                    no_eval = (hr.nelement() == 1)  # 用于判断当前HR图像的像素是否为1
                    if not no_eval:  # 如果当前HR图像像素不为1
                        lr, hr = self.prepare([lr, hr])  # 对LR和HR同时进行数据预处理
                    else:  # 否则就只处理LR图像
                        lr = self.prepare([lr])[0]
                    sr = self.model(lr, idx_scale)  # 测试模型
                    if isinstance(sr, list):
                        sr = sr[-1]  # 保存重构的SR图像的最后一个版本
                    # print("hr的shape:", hr.shape)
                    # print("sr的shape:", sr.shape)
                    sr = utility.quantize(sr, self.args.rgb_range)  # 将图像张量转化为RGB范围的数值
                    # hr_size = hr.shape[2:]  # 可以hr或者sr两个采样都试一下，看那个PSNR好
                    # sr = F.interpolate(sr, size=hr_size, mode='bilinear', align_corners=False)
                    sr_size = sr.shape[2:]
                    hr = F.interpolate(hr, size=sr_size, mode='bilinear', align_corners=False)
                    save_list = [sr]  # 保存处理之后的SR图像
                    if not no_eval:  # 计算PSNR
                        eval_acc += utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range,
                            benchmark=self.loader_test.dataset.benchmark
                        )
                        save_list.extend([lr, hr])
                    if self.args.save_results:
                        self.ckp.save_results(filename, save_list, scale)
                self.ckp.log[-1, idx_scale] = eval_acc / len(self.loader_test)
                best = self.ckp.log.max(0)  # 返回PSNR最大值和对应的索引位置
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        self.args.data_test,
                        scale,
                        self.ckp.log[-1, idx_scale],
                        best[0][idx_scale],
                        best[1][idx_scale] + 1  # 索引从0开始，+1让它从1开始
                    )
                )
        # print("self.args.test_only_test:", self.args.test_only)

        if not self.args.test_only:  # 如果只是测试集，那么对应的不需要保存模型
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))  # 比较最大值所存储的epoch是否和当前一样，不一样这个值为false

    def train(self):
        # torch.cuda.synchronize()
        # start = time.time()
        self.scheduler.step()  # 调整学习率
        self.loss.step()  # 优化损失函数
        epoch = self.scheduler.last_epoch + 2
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log(
            '\n[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        # torch.nn.module的内置方法，将切换模型的评估模式为训练模式
        self.model.train()
        # self.args.test_only = False
        # print("self.args.test_only_train:", self.args.test_only)
        timer_data, timer_model = utility.timer(), utility.timer()
        # tqdm_train = tqdm(self.loader_train, ncols=80)
        for batch, (lr, hr, _, idx_scale) in enumerate(self.loader_train):

            lr, hr = self.prepare([lr, hr])  # 将传入的LR,HR图像转为半精度

            timer_data.hold()  # 暂停计时器
            timer_model.tic()  # 重启计时器

            self.optimizer.zero_grad()  # 清除梯度
            sr = self.model(lr, idx_scale)

            # 计算loss
            if isinstance(sr, list):  # 如果对应恢复的SR网络是一个列表格式，那么对应对每一个SR求损失，然后对损失求和然后求平均
                loss = 0
                for sr_ in sr:
                    loss += self.loss(sr_, hr)
                loss = loss / len(sr)
            else:
                loss = self.loss(sr, hr)

            if loss.item() < self.args.skip_threshold * self.error_last:  # 算出的损失比上一次还要小，可以进行更新
                loss.requires_grad_(True)  # 设置当前损失函数的需要计算梯度的属性为真
                loss.backward()  # 自动计算require_grad属性为真的张量的梯度
                self.optimizer.step()  # 更新参数
            else:  # 否则直接跳过这一个批次的数据
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))

            timer_model.hold()
            # batch编号从0开始，对应的+1表示编号从1开始，*batch_size用于计算当前已经训练的数据量
            if (batch + 1) * self.args.batch_size % self.args.print_every == 0:
                self.ckp.write_log('==> [{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),  # 模型运行时间
                    timer_data.release()))  # 数据加载时间

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))  # 结束批次训练日志记录
        self.error_last = self.loss.log[-1, -1]  # 最后又一批最后一个损失值，表示最终的训练损失值，用于后续的评估

        torch.cuda.synchronize()
        # end = time.time()
        # print("running time is", end - start)

    def prepare(self, l):  # 数据预处理，修改精度
        device = torch.device('cpu' if self.args.cpu else 'cuda')

        def _prepare(tensor):
            if self.args.precision == 'half':
                tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(_l) for _l in l]  # 递归函数

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs
