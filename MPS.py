import copy
import numpy as np
import torch as tc
from scipy.linalg import expm


class mps:

    """
    ！！！重要变量！！！
    - length: MPS长度
    - tensors: 为list，储存MPS中每个小张量
    - center: 正交中心，-1表示不为中心正交形式
    - physdim: 物理指标list
    - virtdim: 虚拟指标list

    ！！！重要成员函数！！！
    - init_tensor：输入指定张量列表，需满足注中要求
    - init_rand: 随机初始化
    - mps2tensor: 收缩所有虚拟指标，返回MPS代表张量
    - mps2vec: 收缩所有虚拟指标，将MPS代表张量转化为向量输出
    - inner: MPS态间做内积
    - center_orth：指定正交中心，对MPS进行中心正交化，或移动MPS正交中心至指定位置
    - evolve_gate: 演化一次门操作
    - TEBD: 时间演化模拟

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

    def __init__(self, tensors):
        """
        建议使用init_rand, init_tensors生成
        """
        self.length = len(tensors)
        self.tensors = copy.deepcopy(tensors)
        self.center = -1
        self.physdim = [self.tensors[0].shape[0]]
        self.virtdim = list()
        for n in range(1, self.length):
            self.physdim += [self.tensors[n].shape[1]]
            self.virtdim += [self.tensors[n].shape[0]]

        
    @classmethod
    def init_rand(cls, d, chi, length):
        """
        随机初始化

        """
        tensors = [np.random.rand(d, chi)]
        for _ in range(1,length-1):
            tensors += [np.random.rand(chi, d, chi)]
        tensors += [np.random.rand(chi, d)]
        return cls(tensors)

    @classmethod
    def init_tensors(cls, tensors):  
        for n in range(0, len(tensors)-1):
            assert tensors[n].shape[-1] == tensors[n+1].shape[0]
        return cls(tensors)

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
        self.center = center
        if normalize:
            self.normalize_center()
        

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
            u, lm, r = self.__svd_or_qr(way, tensor, if_trun, cut_dim)

        else:
            tensor = tensor.reshape(self.virtdim[nt-1]*self.physdim[nt], self.virtdim[nt])
            u, lm, r = self.__svd_or_qr(way, tensor, if_trun, cut_dim)
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
            u, lm, r = self.__svd_or_qr(way, tensor, if_trun, cut_dim)
            u = u.T

        else:
            tensor = tensor.reshape(self.virtdim[nt-1], self.physdim[nt]*self.virtdim[nt]).T
            u, lm, r = self.__svd_or_qr(way, tensor, if_trun, cut_dim)
            u = u.T.reshape(-1, self.physdim[nt], self.virtdim[nt])
        self.tensors[nt] = u
        if normalize:
            r /= np.linalg.norm(r)
        tensor_ = self.get_tensor(nt-1, False)
        tensor_ = np.tensordot(tensor_, r, [[-1], [1]])
        self.tensors[nt-1] = tensor_
        self.virtdim[nt-1] = r.shape[0]
        return lm

    def __svd_or_qr(self, way, tensor, if_trun, cut_dim):
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

    def evolve_gate(self, gate, nt, cut_dim = -1, center = None, debug = False):
        """
        在nt, nt+1上做gate操作
         0    1
         |    |
          gate
         |    |
         2    3
        :param gate: the two-body gate to evolve the MPS
        :param nt: the position of the first spin in the gate, nt <= length - 2
        :param center: where to put the new center; nt(None) or nt+1
        
        """
        if self.center < nt:
            self.center_orth(nt, 'qr', cut_dim, True)
        else:
            self.center_orth(nt + 1, 'qr', cut_dim, True)
        tensor1 = self.get_tensor(nt)
        tensor2 = self.get_tensor(nt+1)

        if nt == 0:
            tensor = np.einsum('ba,acj,klbc->klj', tensor1, tensor2, gate)
            s = tensor.shape
            u, lm, v = np.linalg.svd(tensor.reshape(s[0], s[1]*s[2]), full_matrices=False)
            if debug:
                print(s)
                print(u.shape, lm.size, v.shape)
            if 0 < cut_dim < lm.size:
                if center == nt or center == None:
                    u = u[:, :cut_dim].dot(np.diag(lm[:cut_dim]))
                    v = v[:cut_dim, :].reshape(cut_dim, s[1], s[2])
                    self.center = nt
                else:
                    u = u[:, :cut_dim]
                    v = np.diag(lm[:cut_dim]).dot(v[:cut_dim, :]).reshape(cut_dim, s[1], s[2])
                    self.center = nt+1
                self.virtdim[nt] = cut_dim
            else:
                if center == nt or center == None:
                    u = lm * u
                    v = v.reshape(lm.size, s[1], s[2])
                    self.center = nt
                else:
                    v = np.diag(lm).dot(v).reshape(lm.size, s[1], s[2])
                    self.center = nt+1
                self.virtdim[nt] = lm.size
        
        elif nt == self.length - 2:
            tensor = np.einsum('iba,ac,klbc->ikl', tensor1, tensor2, gate)
            s = tensor.shape
            u, lm, v = np.linalg.svd(tensor.reshape(s[0]*s[1], s[2]), full_matrices=False)
            if debug:
                print(s)
                print(u.shape, lm.size, v.shape)
            if 0 < cut_dim < lm.size:
                if center == nt or center == None:
                    u = u[:, :cut_dim].dot(np.diag(lm[:cut_dim])).reshape(s[0], s[1], cut_dim)
                    v = v[:cut_dim, :]
                    self.center = nt
                else:
                    u = u[:, :cut_dim].reshape(s[0], s[1], cut_dim)
                    v = np.diag(lm[:cut_dim]).dot(v[:cut_dim, :])
                    self.center = nt+1
                self.virtdim[nt] = cut_dim
            else:
                if center == nt or center == None:
                    u = (lm*u).reshape(s[0], s[1], lm.size)
                    v = v
                    self.center = nt
                else:
                    u = u.reshape(s[0], s[1], lm.size)
                    v = np.diag(lm).dot(v)
                    self.center = nt+1
                self.virtdim[nt] = lm.size

        else:
            tensor = np.einsum('iba,acj,klbc->iklj', tensor1, tensor2, gate)
            s = tensor.shape
            u, lm, v = np.linalg.svd(tensor.reshape(s[0]*s[1], s[2]*s[3]), full_matrices=False)
            if debug:
                print(s)
                print(u.shape, lm.size, v.shape)
            if 0 < cut_dim < lm.size:
                if center == nt or center == None:
                    u = u[:, :cut_dim].dot(np.diag(lm[:cut_dim])).reshape(s[0], s[1], cut_dim)
                    v = v[:cut_dim, :].reshape(cut_dim, s[2], s[3])
                    self.center = nt
                else:
                    u = u[:, :cut_dim].reshape(s[0], s[1], cut_dim)
                    v = np.diag(lm[:cut_dim]).dot(v[:cut_dim, :]).reshape(cut_dim, s[2], s[3])
                    self.center = nt+1
                self.virtdim[nt] = cut_dim
            else:
                if center == nt or center == None:
                    u = (lm*u).reshape(s[0], s[1], lm.size)
                    v = v.reshape(lm.size, s[2], s[3])
                    self.center = nt
                else:
                    u = u.reshape(s[0], s[1], lm.size)
                    v = np.diag(lm).dot(v).reshape(lm.size, s[2], s[3])
                    self.center = nt+1
                self.virtdim[nt] = lm.size

        self.tensors[nt] = u
        self.tensors[nt+1] = v

    def TEBD(self, hamiltonion, tau = 1e-4, cut_dim = -1, tol = None, times = 1):
        """
        :param hamiltonion: 时间演化二体哈密顿量，不同哈密顿量使用list输入，相同直接输入矩阵
        :param tau: 模拟步长
        :param cut_dim: 裁剪维数，如果设置tol则为初始裁剪维数
        :param tol: 单步演化允许误差，自适应调节裁剪维数        #还没写#
        :param times: 模拟次数

        """
        if type(cut_dim) != list:
            cut_dim = cut_dim * np.ones(self.length-1)
        if type(hamiltonion) == list:
            for n in range(0, self.length - 1):
                hamiltonion[n] = expm(-1j * tau * hamiltonion[n]) # e^(-iHt)
                if np.linalg.norm(np.imag(hamiltonion[n])) < 1e-15:
                    hamiltonion[n] = np.real(hamiltonion[n])
                hamiltonion[n] = hamiltonion[n].reshape(self.physdim[n], self.physdim[n+1], self.physdim[n], self.physdim[n+1])
            for _ in range(0, times):
                for n in range(0, self.length - 1):
                    self.evolve_gate(hamiltonion[n], n, cut_dim[n], center = n+1)
        else:
            hamiltonion = expm(-1j * tau * hamiltonion) # e^(-iHt)
            if np.linalg.norm(np.imag(hamiltonion)) < 1e-15:
                hamiltonion = np.real(hamiltonion)
            hamiltonion = hamiltonion.reshape(self.physdim[0], self.physdim[1], self.physdim[0], self.physdim[1])
            for _ in range(0, times):
                for n in range(0, self.length - 1):
                    self.evolve_gate(hamiltonion, n, cut_dim[n], center = n+1)


    def __sub__(self, rhs):
        assert self.length == rhs.length
        tensors = list()
        for n in range(0, self.length):
            assert self.tensors[n].shape == rhs.tensors[n].shape
            tensors += [self.tensors[n] - rhs.tensors[n]]
        return mps.init_tensors(tensors)

    def norm(self):
        sum = 0.0
        for n in range(0, self.length):
            sum += np.linalg.norm(self.tensors[n])
        return sum

    def inner(self, rhs):
        """
        <self|rhs>做内积
        :param self: 左矢
        :param rhs: 右矢
        :return tensor: 内积结果
        """
        assert self.length == rhs.length
        assert self.physdim == rhs.physdim

        tensor = np.tensordot(self.tensors[0].conj(), rhs.tensors[0], [[0], [0]])
        for n in range(1, self.length-1):
            tensor = np.einsum('ij, ik, jl->kl', tensor, self.tensors[n].conj(), rhs.tensors[n])
        tensor = np.einsum('ij, ik, jk->', tensor, self.tensors[-1].conj(), rhs.tensors[-1])
        return tensor


    

def ground_state(length, hamiltonion, physdim = None, tau = 1e-4, tol = 1e-6, times = 1e3):
    """
    求哈密顿量list对应的基态，返回MPS
    :param hamiltonion: 时间演化二体哈密顿量，不同哈密顿量使用list输入，相同直接输入矩阵
    :param physdim: 如果hamiltonion为list，必须输入物理指标维度list
    :param tau: 模拟步长
    :param tol: 结果允许误差
    :param times: 模拟次数
    :return m1: 基态对应的MPS态

    """
    if type(hamiltonion) == list:
        for n in range(0, length - 1):
            hamiltonion[n] = -1j * hamiltonion[n]
        tensors = list()
        cut_dim = list()
        tensors += [np.random.rand(physdim[0], physdim[0]*physdim[1])]
        cut_dim += [physdim[0]*physdim[1]]
        for n in range(1, length - 1):
            tensors += [np.random.rand(physdim[n-1]*physdim[n], physdim[n], physdim[n]*physdim[n+1])]
            cut_dim += physdim[n]*physdim[n+1]
        tensors += [np.random.rand(physdim[-1]*physdim[-2], physdim[-1])]

        m0 = mps.init_tensors(tensors)

    else:
        hamiltonion = -1j * hamiltonion
        cut_dim = hamiltonion.shape[0]
        d = np.sqrt(cut_dim).astype(int)
        m0 = mps.init_rand(d, cut_dim, length)

    m0.TEBD(hamiltonion, tau, cut_dim, tol)
    m1 = mps.init_tensors(m0.tensors)
    m1.TEBD(hamiltonion, tau, cut_dim, tol)
    err = (m1 - m0).norm()/m0.norm()
    n = 0
    while err > tol and n < times:
        m0 = mps.init_tensors(m1.tensors)
        m1.TEBD(hamiltonion, tau, d**2, tol)
        err = (m1 - m0).norm()/m0.norm()
        n += 1
    else:
        if n == times:
            print('time out!')
    return m1
    


class mpo:
    """
    MPO算符类，目前仅用于PEPS中的运算，无法作用于MPS

    ！！！重要变量！！！
    - length: MPS长度
    - tensors: 为list，储存MPO中每个小张量
    - center: 正交中心，-1表示不为中心正交形式
    - pd: 物理指标矩阵，[0,:]为上，[1,:]为下，[2,:]为前两者相乘
    - vd: 虚拟指标数组

    注：
        垂直方向为物理指标，水平方向为虚拟指标
        1. 中间每个张量为四阶
               1
               |
         0  —  A  —  3
               |
               2
        
        2. 第0个张量为二阶
         0
         |
         A  —  2
         |
         1
        
        3. 第-1个张量为二阶
               1
               |
         0  —  A
               |
               2
    """
    def __init__(self, tensors, pd = None, vd = None):
        """
        建议使用init_rand, init_tensors生成
        """
        self.length = len(tensors)
        self.tensors = copy.deepcopy(tensors)
        self.center = -1

        self.pd = np.zeros((3, self.length), dtype=int)
        self.vd = np.zeros(self.length - 1, dtype=int)

        if pd == None or vd == None:
            self.pd[0][0] = tensors[0].shape[0]
            self.pd[1][0] = tensors[0].shape[1]
            self.vd[0] = tensors[0].shape[-1]
            for n in range(1, self.length - 1):
                self.pd[0][n] = tensors[n].shape[1]
                self.pd[1][n] = tensors[n].shape[2]
                self.vd[n] = tensors[n].shape[-1]
            self.pd[0][-1] = tensors[-1].shape[1]
            self.pd[1][-1] = tensors[-1].shape[2]
        else:
            self.pd[:2, :] = pd
            self.vd = vd 

        for n in range(0, self.length):
            self.pd[2][n] = self.pd[0][n] * self.pd[1][n]

        
    @classmethod
    def init_rand(cls, pd, vd, length):
        """
        随机初始化
        :param pd: 物理指标矩阵，[0,:]为上，[1,:]为下
        :param vd: 虚拟指标数组

        """
        tensors = [np.random.rand(pd[0][0], pd[1][0], vd[0])]
        for n in range(1,length-1):
            tensors += [np.random.rand(vd[n-1], pd[0][n], pd[1][n], vd[n])]
        tensors += [np.random.rand(vd[-1], pd[0][-1], pd[1][-1])]
        return cls(tensors, pd, vd)

    @classmethod
    def init_tensors(cls, tensors):  
        for n in range(0, len(tensors)-1):
            assert tensors[n].shape[-1] == tensors[n+1].shape[0]
        return cls(tensors)
    
    def get_tensor(self, nt, if_copy = True):
        """
        :param if_copy = True: get copy
        :param if_copy = False: get self
        
        """
        if if_copy:
            return copy.deepcopy(self.tensors[nt])
        else:
            return self.tensors[nt]

    def center_orth(self, center, way = 'svd', cut_dim = -1, normalize = False):
        """
        使center成为正交中心
        :param way: 正交方法，默认SVD，可选QR
        :param cut_dim: 裁剪维数，-1表示不裁剪，如果裁剪则使用SVD
        :param normalize: 归一化，默认不归一

        """
        # 转化为MPS
        self.tensors[0] = self.tensors[0].reshape(self.pd[2][0], self.vd[0])
        for n in range(1, self.length - 1):
            self.tensors[n] = self.tensors[n].reshape(self.vd[n-1], self.pd[2][n], self.vd[n])
        self.tensors[-1] = self.tensors[-1].reshape(self.vd[n-1], self.pd[2][n])

        # 复用MPS代码
        if self.center < 0:
            self.orth_n1_n2(0, center, way, cut_dim, normalize)
            self.orth_n1_n2(self.length-1, center, way, cut_dim, normalize)
        elif self.center != center:
            self.orth_n1_n2(self.center, center, way, cut_dim, normalize)
        self.center = center
        if normalize:
            self.normalize_center()

        # 转化为MPO
        self.tensors[0] = self.tensors[0].reshape(self.pd[0][0], self.pd[1][0], self.vd[0])
        for n in range(1, self.length - 1):
            self.tensors[n] = self.tensors[n].reshape(self.vd[n-1], self.pd[0][n], self.pd[1][n], self.vd[n])
        self.tensors[-1] = self.tensors[-1].reshape(self.vd[n-1], self.pd[0][n], self.pd[1][n])
        

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

        if 0 < cut_dim < self.vd[nt]:
            # In this case, truncation is required
            way = 'svd'
            if_trun = True
        else:
            if_trun = False

        assert way.lower() == 'svd' or way.lower() == 'qr'
        
        tensor = self.get_tensor(nt, False)
        if nt == 0:
            u, lm, r = self.__svd_or_qr(way, tensor, if_trun, cut_dim)

        else:
            tensor = tensor.reshape(self.vd[nt-1]*self.pd[2][nt], self.vd[nt])
            u, lm, r = self.__svd_or_qr(way, tensor, if_trun, cut_dim)
            u = u.reshape(self.vd[nt-1], self.pd[2][nt], -1)
        self.tensors[nt] = u
        if normalize:
            r /= np.linalg.norm(r)
        tensor_ = self.get_tensor(nt+1, False)
        tensor_ = np.tensordot(r, tensor_, [[1], [0]])
        self.tensors[nt+1] = tensor_
        self.vd[nt] = r.shape[0]
        return lm

    def orth_right2left(self, nt, way = 'svd', cut_dim = -1, normalize = False):
        """
        使nt->nt-1从右向左正交
        :param nt: 正交位置
        :param way: 正交方法，默认SVD，可选QR
        :param cut_dim: 裁剪维数，-1表示不裁剪，如果裁剪则使用SVD
        :param normalize: 归一化，默认不归一

        """

        if 0 < cut_dim < self.vd[nt-1]:
            # In this case, truncation is required
            way = 'svd'
            if_trun = True
        else:
            if_trun = False

        assert way.lower() == 'svd' or way.lower() == 'qr'
        
        tensor = self.get_tensor(nt, False)
        if nt == self.length-1:
            tensor = tensor.T
            u, lm, r = self.__svd_or_qr(way, tensor, if_trun, cut_dim)
            u = u.T

        else:
            tensor = tensor.reshape(self.vd[nt-1], self.pd[2][nt]*self.vd[nt]).T
            u, lm, r = self.__svd_or_qr(way, tensor, if_trun, cut_dim)
            u = u.T.reshape(-1, self.pd[2][nt], self.vd[nt])
        self.tensors[nt] = u
        if normalize:
            r /= np.linalg.norm(r)
        tensor_ = self.get_tensor(nt-1, False)
        tensor_ = np.tensordot(tensor_, r, [[-1], [1]])
        self.tensors[nt-1] = tensor_
        self.vd[nt-1] = r.shape[0]
        return lm

    def __svd_or_qr(self, way, tensor, if_trun, cut_dim):
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
    



class canonical_form:
    """
    canonical form of mps
    only can be created from an mps
    """

    def __init__(self, mps, cut_dim = -1, normalize = False):
        """
        :param mps: MPS态
        :param cut_dim: 裁剪维数，-1表示不裁剪，如果裁剪则使用SVD
        :param normalize: 归一化，默认不归一

        """
        self.tensors = mps.tensors


    

