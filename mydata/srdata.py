import os

import imageio
import numpy as np
import torch.utils.data as data

import mydata.common as common


class SRData(data.Dataset):
    """ SR数据集接口，建立数据集需要实现这个接口
    """

    def __init__(self, args, train=True, benchmark=False):
        self.args = args
        self.train = train
        self.split = 'train' if train else 'test'
        self.benchmark = benchmark
        self.scale = args.scale
        self.idx_scale = 0
        self._set_filesystem(args.dir_data)

        def _load_bin():
            """ 载入二进制文件images_hr，images_lr
            """
            self.images_hr = np.load(self._name_hrbin())
            self.images_lr = [
                np.load(self._name_lrbin(s))
                for s in self.scale
            ]

        if args.ext == 'img' or benchmark:
            self.images_hr, self.images_lr = self._scan()

        elif args.ext.find('sep') >= 0:  # seperated binary files
            self.images_hr, self.images_lr = self._scan()
            if args.ext.find('reset') >= 0:
                print('Preparing seperated binary files')
                for v in self.images_hr:
                    hr = imageio.imread(v)
                    # print hr.shape
                    # lr = imageio.imresize(hr,[hr.shape[0]/3,hr.shape[1]/3,3],'bicubic')
                    # i=i+1;
                    # lr_name = '/home/work/NQ/DIV2K/DIV2K_train_LR_bicubic_PIL/X3/{:04d}x3.png'.format(i)
                    # imageio.imsave(lr_name,lr)
                    name_sep = v.replace(self.ext, '.npy')
                    np.save(name_sep, hr)
                for si, s in enumerate(self.scale):
                    for v in self.images_lr[si]:
                        lr = imageio.imread(v)
                        name_sep = v.replace(self.ext, '.npy')
                        np.save(name_sep, lr)

            self.images_hr = [
                v.replace(self.ext, '.npy')
                for v in self.images_hr
            ]
            self.images_lr = [
                [v.replace(self.ext, '.npy') for v in self.images_lr[i]]
                for i in range(len(self.scale))
            ]

        elif args.ext.find('bin') >= 0:
            try:
                if args.ext.find('reset') >= 0:
                    raise IOError
                print('Loading a binary file')
                _load_bin()
            except:
                print('Preparing a binary file')
                bin_path = os.path.join(self.apath, 'bin')
                if not os.path.isdir(bin_path):
                    os.mkdir(bin_path)
                list_hr, list_lr = self._scan()
                hr = [imageio.imread(f) for f in list_hr]
                np.save(self._name_hrbin(), hr)
                del hr
                for si, s in enumerate(self.scale):
                    lr_scale = [imageio.imread(f) for f in list_lr[si]]
                    np.save(self._name_lrbin(s), lr_scale)
                    del lr_scale
                print('Loading a binary file')
                _load_bin()

        else:
            print('Please define data type')

    def _scan(self):
        raise NotImplementedError

    def _set_filesystem(self, dir_data):
        raise NotImplementedError

    def _name_hrbin(self):
        """返回hr文件的load/save路径"""
        raise NotImplementedError

    def _name_lrbin(self, scale):
        """返沪lr文件的load/save路径"""
        raise NotImplementedError

    def __getitem__(self, idx):
        lr, hr, filename = self._load_file(idx)  # 获取对应下标的数据集
        lr, hr = self._get_patch(lr, hr)
        lr, hr = common.set_channel([lr, hr], self.args.n_colors)
        lr_tensor, hr_tensor = common.np2Tensor([lr, hr], self.args.rgb_range)
        return lr_tensor, hr_tensor, filename

    def __len__(self):
        """返回数据集的数量"""
        return len(self.images_hr)

    def _get_index(self, idx):
        return idx

    def _load_file(self, idx):
        """加载对应下标的数据"""
        idx = self._get_index(idx)
        lr = self.images_lr[self.idx_scale][idx]
        hr = self.images_hr[idx]
        if self.args.ext == 'img' or self.benchmark:
            filename = hr
            lr = imageio.imread(lr)
            hr = imageio.imread(hr)
        elif self.args.ext.find('sep') >= 0:
            filename = hr
            lr = np.load(lr)
            hr = np.load(hr)
        else:
            filename = str(idx + 1)

        filename = os.path.splitext(os.path.split(filename)[-1])[0]

        return lr, hr, filename

    def _get_patch(self, lr, hr):
        patch_size = self.args.patch_size
        scale = self.scale[self.idx_scale]
        multi_scale = len(self.scale) > 1
        if self.train:
            lr, hr = common.get_patch(  # 裁剪
                lr, hr, patch_size, scale, multi_scale=multi_scale
            )
            lr, hr = common.augment([lr, hr])  # 图片增墒
            lr = common.add_noise(lr, self.args.noise)  # 增加噪音
        else:
            ih, iw = lr.shape[0:2]
            hr = hr[0:ih * scale, 0:iw * scale]  # 简单裁切

        return lr, hr

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale
