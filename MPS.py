import numpy as np
import copy
import torch as tc
import matplotlib.pyplot as plt


class mps:

    """
    ！！！重要成员函数！！！
    - init_tensor：随机初始化张量，或输入指定张量列表，需满足注1要求
    - mps2tensor: 收缩所有虚拟指标，返回MPS代表张量
    - mps2vec: 收缩所有虚拟指标，将MPS代表张量转化为向量输出
    - center_orth：指定正交中心，对MPS进行中心正交化，或移动MPS正交中心至指定位置

    注：
        1. 中间每个张量为三阶，指标顺序为：左辅助、物理、右辅助
               1
               |
         0  —  A  —  2
        
        2. 第0个张量为二阶，指标顺序为：物理、右辅助
         0
         |
         A  —  1
        
        3. 第-1个张量为二阶，指标顺序为：左辅助、物理
               1
               |
         0  —  A
    
    """

    def __init__(self):
        """
        :param d: 物理指标维数
        :param chi: 辅助指标截断维数
        :param length: 张量个数

        :param d: physical bond dimension
        :param chi: virtual bond dimension cut-off
        :param length: number of tensors
        
        """
        self.center = -1  # 正交中心 当center为负数时，MPS非中心正交
        self.dtype = None
        
    @classmethod
    def init_rand(cls, d, chi, length):
        """
        随机初始化

        """
        cls.d = d  # physical bond dimension
        cls.chi = chi  # cut off dimension
        cls.length = length  # number of tensors
        cls.tensors = list()  # tensors in MPS
        cls.physdim = list()  # physical bond dimensions
        cls.virtdim = list()  # virtual bond dimensions
        cls.init()
        cls.tensors = [np.random.randn(cls.d, cls.chi)]
        for n in range(1,cls.length-1):
            cls.tensors += [np.random.randn(cls.chi, cls.d, cls.chi)]
        cls.tensors += [np.random.randn(cls.chi, cls.d)]
        cls.virtdim = [cls.chi] * (cls.length - 1)
        cls.physdim = [cls.d] * cls.length

    @classmethod
    def init_tensors(cls, tensors):
        cls.length = len(tensors)-1  # number of tensors
        cls.tensors = list()  # tensors in MPS
        cls.physdim = list()  # physical bond dimensions
        cls.virtdim = list()  # virtual bond dimensions
        cls.center = -1  # 正交中心 当center为负数时，MPS非中心正交
        cls.dtype = None
        
        for n in range(0, cls.length):
            assert tensors[n].shape[-1] == tensors[n+1].shape[0]
            cls.virtdim += [cls.tensors[n].shape[0]]

        for n in range(0,cls.length):
            cls.tensors += [copy.deepcopy(tensors[n])]
            cls.physdim[n] = tensors[n].shape[1]
        cls.physdim[0] = tensors[n].shape[0]

        return cls()

    def mps2tensor(self):
        """
        收缩所有虚拟指标，返回MPS代表张量

        """
        tensor = self.tensors[0]
        for n in range(1, self.length):
            tensor_ = self.tensors[n]
            tensor = np.tensordot(tensor, tensor_, [[-1], [0]])
        return np.squeeze(tensor)

    def get_tensor(self, nt, if_copy = True):
        """
        :param if_copy = True: get copy
        :param if_copy = False: get self
        
        """
        if if_copy:
            return copy.deepcopy(self.tensors[nt])
        else:
            return self.tensors[nt]
    

    def mps2vec(self):
        vec = self.mps2tensor().reshape((-1,))
        return vec

    
    def center_orth(self, center, way = 'svd', cut_dim = -1, normalize = False):
        """
        使center成为正交中心
        :param way: 正交方法，默认SVD，可选QR
        :param cut_dim: 裁剪维数，-1表示不裁剪，如果裁剪则使用SVD
        :param normalize: 归一化，默认不归一

        """
        if self.center < 0:
            self.orth_n1_n2(0, center, way, cut_dim, normalize)
            self.orth_n1_n2(self.length-1, center, way, cut_dim, normalize)
        elif self.center != center:
            self.orth_n1_n2(self.center, center, way, cut_dim, normalize)
        if normalize:
            self.normalize_center()
        self.center = center

    def normalize_center(self):
        if self.center > -1:
            nt = self.center
            norm = np.linalg.norm(self.tensors[nt])
            self.tensors[nt] /= norm

    def orth_n1_n2(self, n1, n2, way = 'svd', cut_dim = -1, normalize = False):
        """
        使n1->n2正交
        :param way: 正交方法，默认SVD，可选QR
        :param cut_dim: 裁剪维数，-1表示不裁剪，如果裁剪则使用SVD
        :param normalize: 归一化，默认不归一

        """
        if n1 < n2:
            for nt in range(n1, n2, 1):
                self.orth_left2right(nt, way, cut_dim, normalize)
        else:
            for nt in range(n1, n2, -1):
                self.orth_right2left(nt, way, cut_dim, normalize)


    
    def orth_left2right(self, nt, way = 'svd', cut_dim = -1, normalize = False):
        """
        使nt->nt+1从左向右正交
        :param nt: 正交位置
        :param way: 正交方法，默认SVD，可选QR
        :param cut_dim: 裁剪维数，-1表示不裁剪，如果裁剪则使用SVD
        :param normalize: 归一化，默认不归一

        """

        if 0 < cut_dim < self.virtdim[nt]:
            # In this case, truncation is required
            way = 'svd'
            if_trun = True
        else:
            if_trun = False

        assert way.lower() == 'svd' or way.lower() == 'qr'
        
        tensor = self.get_tensor(nt, False)
        if nt == 0:
            u, lm, r = self.svd_or_qr(way, tensor, if_trun, cut_dim)

        else:
            tensor = tensor.reshape(self.virtdim[nt-1]*self.physdim[nt], self.virtdim[nt])
            u, lm, r = self.svd_or_qr(way, tensor, if_trun, cut_dim)
            u = u.reshape(self.virtdim[nt-1], self.physdim[nt], -1)
        self.tensors[nt] = u
        if normalize:
            r /= np.linalg.norm(r)
        tensor_ = self.get_tensor(nt+1, False)
        tensor_ = np.tensordot(r, tensor_, [[1], [0]])
        self.tensors[nt+1] = tensor_
        self.virtdim[nt] = r.shape[0]
        return lm

    def orth_right2left(self, nt, way = 'svd', cut_dim = -1, normalize = False):
        """
        使nt->nt-1从右向左正交
        :param nt: 正交位置
        :param way: 正交方法，默认SVD，可选QR
        :param cut_dim: 裁剪维数，-1表示不裁剪，如果裁剪则使用SVD
        :param normalize: 归一化，默认不归一

        """

        if 0 < cut_dim < self.virtdim[nt-1]:
            # In this case, truncation is required
            way = 'svd'
            if_trun = True
        else:
            if_trun = False

        assert way.lower() == 'svd' or way.lower() == 'qr'
        
        tensor = self.get_tensor(nt, False)
        if nt == self.length-1:
            tensor = tensor.T
            u, lm, r = self.svd_or_qr(way, tensor, if_trun, cut_dim)
            u = u.T

        else:
            tensor = tensor.reshape(self.virtdim[nt-1], self.physdim[nt]*self.virtdim[nt]).T
            u, lm, r = self.svd_or_qr(way, tensor, if_trun, cut_dim)
            u = u.T.reshape(-1, self.physdim[nt], self.virtdim[nt])
        self.tensors[nt] = u
        if normalize:
            r /= np.linalg.norm(r)
        tensor_ = self.get_tensor(nt-1, False)
        tensor_ = np.tensordot(tensor_, r, [[-1], [1]])
        self.tensors[nt-1] = tensor_
        self.virtdim[nt-1] = r.shape[0]
        return lm

    def svd_or_qr(self, way, tensor, if_trun, cut_dim):
        if way.lower() == 'svd':
            u, lm, v = np.linalg.svd(tensor, full_matrices = False)
            if if_trun:
                u = u[:, :cut_dim]
                r = np.diag(lm[:cut_dim]).dot(v[:cut_dim, :])
            else:
                r = np.diag(lm).dot(v)
        if way.lower() == 'qr':
            u, r = np.linalg.qr(tensor)
            lm = None
        return u, lm, r


    

