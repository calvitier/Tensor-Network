import copy
import numpy as np
import torch as tc
from scipy.linalg import expm
import MPS

class peps:

    """
    ！！！重要变量！！！
    - n：行数 >= 2
    - m：列数 >= 2
    - shape: (n, m)
    - tensors: 二维list，储存PEPS中每个小张量
    - pd: 物理指标矩阵，大小为shape
    - vd_hori: 水平虚拟指标维数矩阵，n * m-1
    - vd_vert: 竖直虚拟指标维数矩阵，n-1 * m
    - is_GL: True表示为Gamma-Lambda形式，False表示不为
    - Lambda_hori：二维list，储存Gamma-Lambda形式时的Lambda对角阵，is_GL = False时为空，n * m-1
    - Lambda_vert：二维list，储存Gamma-Lambda形式时的Lambda对角阵，is_GL = False时为空，n-1 * m

    ！！！重要成员函数！！！
    - init_tensor：输入指定张量列表，需满足注中要求
    - init_rand: 随机初始化
    - inner: PEPS态间做内积
    - to_Gamma_Lambda：PEPS转化为Gamma-Lambda形式
    - to_PEPS：Gamma-Lambda形式转化为PEPS
    - evolve_gate: 演化一次门操作
    - TEBD: 时间演化模拟

    注：
        物理指标皆为第-1个指标
        物理指标用两道//标注，虚拟指标为一道

        1. 内部每个张量为五阶，指标顺序为：左上右下4个辅助指标、物理指标
              1   4
              | //
        0  —  A  —  2
              |
              3
        
        2. 四个顶角张量为三阶
        A  —  0       0  —  A
        | \\             // |
        1   2           2   1

        1   2           2   1
        | //             \\ |
        A  —  0       0  —  A
        
        3. 四边张量为四阶
                0  —  A  —  1
                      | \\
                      2   3
        0   3                  3   0
        | //                    \\ |
        A  —  1              1  —  A
        |                          |
        2                          2
                      2   3
                      | //
                0  —  A  —  1
        
    """

    def __init__(self, tensors):
        """
        建议使用init_rand, init_tensors生成
        """
        self.n = len(tensors)
        self.m = len(tensors[0])
        self.shape = (self.n, self.m)
        self.tensors = copy.deepcopy(tensors)
        self.is_GL = False
        self.Lambda_hori = list()
        self.Lambda_vert = list()
        self.pd = np.zeros(self.shape, dtype=int)
        self.vd_hori = np.zeros((self.n, self.m - 1), dtype=int)
        self.vd_vert = np.zeros((self.n - 1, self.m), dtype=int)

        # 物理指标
        for i in range(0, self.n):
            for j in range(0, self.m):
                self.pd[i][j] = self.tensors[i][j].shape[-1]

        # 虚拟指标
        # 第一行 i = 0
        self.vd_hori[0][0] = self.tensors[0][0].shape[0]
        self.vd_vert[0][0] = self.tensors[0][0].shape[1]
        for j in range(1, self.m - 1):
            self.vd_hori[0][j] = self.tensors[0][j].shape[1]
            self.vd_vert[0][j] = self.tensors[0][j].shape[2]
        self.vd_vert[0][-1] = self.tensors[0][-1].shape[1]

        # i = 1 ~ n-2
        for i in range(1, self.n - 1):
            self.vd_hori[i][0] = self.tensors[i][0].shape[1]
            self.vd_vert[i][0] = self.tensors[i][0].shape[2]
            for j in range(1, self.m - 1):
                self.vd_hori[i][j] = self.tensors[i][j].shape[2]
                self.vd_vert[i][j] = self.tensors[i][j].shape[3]
            self.vd_vert[i][-1] = self.tensors[i][-1].shape[2]

        # i = n-1
        self.vd_hori[-1][0] = self.tensors[0][0].shape[0]
        for j in range(1, self.m - 1):
            self.vd_hori[-1][j] = self.tensors[0][j].shape[1]

        
    @classmethod
    def init_rand(cls, pd, vd, shape):
        """
        随机初始化

        """
        n = shape[0]
        m = shape[1]
        assert n >= 2 and m >= 2
        tensors = list()

        # i = 0
        tensor = [np.random.rand(vd, vd, pd)]
        for _ in range(1, m - 1):
            tensor += [np.random.rand(vd, vd, vd, pd)]
        tensor += [np.random.rand(vd, vd, pd)]
        tensors += [tensor]

        # i = 1 ~ n-2
        for _ in range(1, n - 1):
            tensor = list()
            tensor += [np.random.rand(vd, vd, vd, pd)]
            for __ in range(1, m - 1):
                tensor += [np.random.rand(vd, vd, vd, vd, pd)]
            tensor += [np.random.rand(vd, vd, vd, pd)]
            tensors += [tensor]

        # i = n-1
        tensor = [np.random.rand(vd, vd, pd)]
        for _ in range(1, m - 1):
            tensor += [np.random.rand(vd, vd, vd, pd)]
        tensor += [np.random.rand(vd, vd, pd)]
        tensors += [tensor]

        return cls(tensors)

    @classmethod
    def init_tensors(cls, tensors):  
        """
        暂无格式检查
        """
        return cls(tensors)


    def inner(self, rhs, cut_dim = None, debug = False):
        """
        <self|rhs>做内积
        目前仅支持一般形式，GL形式会转化为一般形式做内积
        按行收缩，头尾两侧分别进行
        中间结果为一MPO
        :param self: 左矢
        :param rhs: 右矢
        :param cut_dim: 收缩过程裁剪维数，默认为最大虚拟指标的平方
        :param tol: 收缩过程中近似允许误差
        :return tensor: 内积结果

        """
        assert self.shape == rhs.shape
        assert (self.pd == rhs.pd).all()

        self.to_PEPS()
        rhs.to_PEPS()
        
        if cut_dim == None:
            cut_dim = max(self.vd_max(), rhs.vd_max()) ** 2

        # i = 0
        mpo_t1 = list()
        tensor = np.einsum('ijp, mnp ->imjn', self.tensors[0][0].conj(), rhs.tensors[0][0])
        tensor = tensor.reshape(self.vd_vert[0][0], rhs.vd_vert[0][0], self.vd_hori[0][0] * rhs.vd_hori[0][0])
        mpo_t1 += [tensor]
        for j in range(1, self.m -1):
            tensor = np.einsum('ijkp, lmnp ->ilknjm', self.tensors[0][j].conj(), rhs.tensors[0][j])
            tensor = tensor.reshape(self.vd_hori[0][j-1] * rhs.vd_hori[0][j-1], self.vd_vert[0][j], rhs.vd_vert[0][j], self.vd_hori[0][j] * rhs.vd_hori[0][j])
            mpo_t1 += [tensor]
        tensor = np.einsum('ijp, mnp ->imjn', self.tensors[0][0].conj(), rhs.tensors[0][0])
        tensor = tensor.reshape(self.vd_hori[0][-1] * rhs.vd_hori[0][-1], self.vd_vert[0][0], rhs.vd_vert[0][0])
        mpo_t1 += [tensor]
        mpo_t1 = MPS.mpo.init_tensors(mpo_t1)
        mpo_t1.center_orth(0, cut_dim=cut_dim)

        # i = 1 ~ n-2
        for i in range(1, self.n-1):
            mpo_t2 = list()
            # j = 0
            t = list()
            t += [tc.from_numpy(self.tensors[i][0].conj())]
            t += [tc.from_numpy(mpo_t1.tensors[0])]
            t += [tc.from_numpy(rhs.tensors[i][0])]
            mpo_t2 += [tc.einsum('abcd, axk, xyzd -> czbky', t[0], t[1], t[2])]
            vd1 = self.vd_hori[i][0] * mpo_t1.vd[0] * rhs.vd_hori[i][0]
            mpo_t2[0] = mpo_t2[0].reshape(self.vd_vert[i][0], rhs.vd_vert[i][0], vd1)
            mpo_t2[0] = mpo_t2[0].numpy()
            # j = 1 ~ m-2
            for j in range(1, self.m-1):
                t = list()
                t += [tc.from_numpy(self.tensors[i][j].conj())]
                t += [tc.from_numpy(mpo_t1.tensors[j])]
                t += [tc.from_numpy(rhs.tensors[i][j])]
                mpo_t2 += [tc.einsum('abcde, ibyl, xyzwe -> aixdwclz', t[0], t[1], t[2])]
                vd2 = self.vd_hori[i][j] * mpo_t1.vd[j] * rhs.vd_hori[i][j]
                mpo_t2[j] = mpo_t2[j].reshape(vd1, self.vd_vert[i][0], rhs.vd_vert[i][0], vd2)
                mpo_t2[j] = mpo_t2[j].numpy()
                vd1 = vd2
            # j = -1
            t = list()
            t += [tc.from_numpy(self.tensors[i][-1].conj())]
            t += [tc.from_numpy(mpo_t1.tensors[-1])]
            t += [tc.from_numpy(rhs.tensors[i][-1])]
            mpo_t2 += [tc.einsum('abcd, iax, xyzd -> biycz', t[0], t[1], t[2])]
            mpo_t2[-1] = mpo_t2[-1].reshape(vd1, self.vd_vert[i][-1], rhs.vd_vert[i][-1])
            mpo_t2[-1] = mpo_t2[-1].numpy()

            mpo_t1 = MPS.mpo.init_tensors(mpo_t2)
            mpo_t1.center_orth(0, cut_dim=cut_dim)
        
        # i = -1
        # j = 0
        t = list()
        t += [tc.from_numpy(self.tensors[-1][0].conj())]
        t += [tc.from_numpy(mpo_t1.tensors[0])]
        t += [tc.from_numpy(rhs.tensors[-1][0])]
        tensor = tc.einsum('abc, byk, xyc -> akx', t[0], t[1], t[2])
        # j = 1 ~ m-2
        for j in range(1, self.m-1):
            t = list()
            t += [tc.from_numpy(self.tensors[-1][j].conj())]
            t += [tc.from_numpy(mpo_t1.tensors[j])]
            t += [tc.from_numpy(rhs.tensors[-1][j])]
            tensor_t = tc.einsum('abcd, iczl, xyzd -> aixbly', t[0], t[1], t[2])
            tensor = tc.einsum('abc, abcdef -> def', tensor, tensor_t)
        # j = -1
        t = list()
        t += [tc.from_numpy(self.tensors[-1][-1].conj())]
        t += [tc.from_numpy(mpo_t1.tensors[-1])]
        t += [tc.from_numpy(rhs.tensors[-1][-1])]
        tensor_t = tc.einsum('abc, iby, xyc -> aix', t[0], t[1], t[2])
        inner = tc.einsum('abc, abc ->', tensor, tensor_t)

        return inner.item()

    """
    def inner(self, rhs, cut_dim = None, tol = 1e-3, debug = False):
        
        pd = np.zeros((2, self.m), dtype=int)
        pd[0, :] = self.vd_vert[i, :]
        pd[1, :] = rhs.vd_vert[i, :]
        vd = np.ones(self.m - 1, dtype=int) * cut_dim
        mpo_t2 = MPS.mpo.init_rand(pd, vd, self.m)
        mpo_t1_inner = mpo_t1.inner(mpo_t1)
        f = 1

        while f > tol:
            mpo_t2 = self.__inner_left2right(rhs, i, mpo_t1, mpo_t2, cut_dim=cut_dim, debug=debug)
            mpo_t2 = self.__inner_right2left(rhs, i, mpo_t1, mpo_t2, cut_dim=cut_dim, debug=debug)
            f = mpo_t1_inner + mpo_t2.inner(mpo_t2) - mpo_t2.inner(mpo_t1) - mpo_t1.inner(mpo_t2)
            print(f)
        
        mpo_t1 = mpo_t2
    """

    def __inner_left2right(self, rhs, i, mpo, mpo_t, cut_dim = -1, debug = False):
        """
        做内积过程中，对i行，从左至右求解近似
        :param i: 行
        :param mpo: 当前mpo
        :param mpo_t: 近似过程量

        """
        mpo_t.center_orth(0, cut_dim=cut_dim)

        # 计算r
        r = list([tc.zeros(1)] * (self.m - 1))
        # numpy无法做五阶张量einsum，改用torch
        t = list()
        t += [tc.from_numpy(mpo_t.tensors[-1].conj())]
        t += [tc.from_numpy(self.tensors[i][-1].conj())]
        t += [tc.from_numpy(rhs.tensors[i][-1])]
        t += [tc.from_numpy(mpo.tensors[-1])]
        r[-1] = tc.einsum('aij, ibkl, jcml, dkm -> dbca', t[0], t[1], t[2], t[3])
        for j in range(self.m - 2, 0, -1):
            t = list()
            t += [tc.from_numpy(mpo_t.tensors[j].conj())]
            t += [tc.from_numpy(self.tensors[i][j].conj())]
            t += [tc.from_numpy(rhs.tensors[i][j])]
            t += [tc.from_numpy(mpo.tensors[j])]
            r[j-1] = tc.einsum('xyzw, aijx, biykl, cjzml, dkmw -> dbca',r[j], t[0], t[1], t[2], t[3])
        
        # 更新mpo_t.tensors
        # j = 0
        t = list()
        t += [tc.from_numpy(mpo.tensors[0])]
        t += [tc.from_numpy(self.tensors[i][0].conj())]
        t += [tc.from_numpy(rhs.tensors[i][0])]
        mpo_t.tensors[0] = tc.einsum('ija, iblm, jcnm, abcd -> lnd', t[0], t[1], t[2], r[0])
        t += [mpo_t.tensors[0].conj()]
        mpo_t.tensors[0] = mpo_t.tensors[0].numpy()
        # 计算l
        mpo_t.center_orth(1, cut_dim=cut_dim)
        l = list([tc.zeros(1)] * (self.m))  # l[0]不使用
        l[1] = tc.einsum('ija, iblm, jcnm, lnd -> abcd', t[0], t[1], t[2], t[3])
        # j = 1 ~ m-2
        for j in range(1, self.m - 1):
            t = list()
            t += [tc.from_numpy(mpo.tensors[j])]
            t += [tc.from_numpy(self.tensors[i][j].conj())]
            t += [tc.from_numpy(rhs.tensors[i][j])]
            mpo_t.tensors[j] = tc.einsum('abcd, aijx, biykl, cjzml, xyzw -> dkmw', l[j], t[0], t[1], t[2], r[j])
            t += [mpo_t.tensors[j].conj()]
            mpo_t.tensors[j] = mpo_t.tensors[j].numpy()

            mpo_t.center_orth(j+1, cut_dim=cut_dim)
            l[j+1] = tc.einsum('abcd, aijx, biykl, cjzml, dkmw -> xyzw', l[j], t[0], t[1], t[2], t[3])
        
        # j = m-1
        t = list()
        t += [tc.from_numpy(mpo.tensors[-1])]
        t += [tc.from_numpy(self.tensors[i][-1].conj())]
        t += [tc.from_numpy(rhs.tensors[i][-1])]
        mpo_t.tensors[-1] = tc.einsum('abcd, aij, ibkm, jclm -> dkl', l[-1], t[0], t[1], t[2])
        mpo_t.tensors[-1] = mpo_t.tensors[-1].numpy()

        if debug:
            print('success!')

        return mpo_t

    def __inner_right2left(self, rhs, i, mpo, mpo_t, cut_dim = -1, debug = False):
        """
        做内积过程中，对i行，从右至左求解近似
        :param i: 行
        :param mpo: 当前mpo
        :param mpo_t: 近似过程量

        """
        mpo_t.center_orth(-1, cut_dim=cut_dim)

        # 计算l
        l = list([tc.zeros(1)] * self.m)    # l[0]不使用
        # numpy无法做五阶张量einsum，改用torch
        t = list()
        t += [tc.from_numpy(mpo.tensors[0])]
        t += [tc.from_numpy(self.tensors[i][0].conj())]
        t += [tc.from_numpy(rhs.tensors[i][0])]
        t += [tc.from_numpy(mpo_t.tensors[0].conj())]
        l[1] = tc.einsum('ija, iblm, jcnm, lnd -> abcd', t[0], t[1], t[2], t[3])

        for j in range(self.m - 2, 0, -1):
            t = list()
            t += [tc.from_numpy(mpo.tensors[j])]
            t += [tc.from_numpy(self.tensors[i][j].conj())]
            t += [tc.from_numpy(rhs.tensors[i][j])]
            t += [tc.from_numpy(mpo_t.tensors[j].conj())]
            l[j+1] = tc.einsum('abcd, aijx, biykl, cjzml, dkmw -> xyzw', l[j], t[0], t[1], t[2], t[3])
        
        # 更新mpo_t.tensors
        # j = m-1
        t = list()
        t += [tc.from_numpy(mpo.tensors[-1])]
        t += [tc.from_numpy(self.tensors[i][-1].conj())]
        t += [tc.from_numpy(rhs.tensors[i][-1])]
        mpo_t.tensors[-1] = tc.einsum('abcd, aij, ibkm, jclm -> dkl', l[-1], t[0], t[1], t[2])
        t += [mpo_t.tensors[-1].conj()]
        mpo_t.tensors[-1] = mpo_t.tensors[-1].numpy()
        # 计算r
        mpo_t.center_orth(-2, cut_dim=cut_dim)
        r = list([tc.zeros(1)] * (self.m - 1))
        r[-1] = tc.einsum('aij, ibkl, jcml, dkm -> dbca', t[3], t[1], t[2], t[0])
        # j = m-2 ~ 1
        for j in range(self.m - 2, 0, -1):
            t = list()
            t += [tc.from_numpy(mpo.tensors[j])]
            t += [tc.from_numpy(self.tensors[i][j].conj())]
            t += [tc.from_numpy(rhs.tensors[i][j])]
            mpo_t.tensors[j] = tc.einsum('abcd, aijx, biykl, cjzml, xyzw -> dkmw', l[j], t[0], t[1], t[2], r[j])
            t += [mpo_t.tensors[j].conj()]
            mpo_t.tensors[j] = mpo_t.tensors[j].numpy()

            mpo_t.center_orth(j-1, cut_dim=cut_dim)
            r[j-1] = tc.einsum('xyzw, aijx, biykl, cjzml, dkmw -> dbca',r[j], t[3], t[1], t[2], t[0])
        
        # j = 0
        t = list()
        t += [tc.from_numpy(mpo.tensors[0])]
        t += [tc.from_numpy(self.tensors[i][0].conj())]
        t += [tc.from_numpy(rhs.tensors[i][0])]
        mpo_t.tensors[0] = tc.einsum('ija, iblm, jcnm, abcd -> lnd', t[0], t[1], t[2], r[0])
        mpo_t.tensors[0] = mpo_t.tensors[0].numpy()

        if debug:
            print('success!')

        return mpo_t

    def vd_max(self):
        """
        返回虚拟指标最大维数
        """
        return max(self.vd_hori.max(), self.vd_vert.max())


    def to_PEPS(self):
        """
        转化为一般形式

        """
        if self.is_GL:

            # i = 0
            # j = 0
            self.tensors[0][0] = np.einsum('abc, ai, bj -> ijc', self.tensors[0][0], np.diag(self.Lambda_hori[0][0]), np.diag(self.Lambda_vert[0][0]))
            # j = 1 ~ m-2
            for j in range(1, self.m-1):
                self.tensors[0][j] = np.einsum('abcd, bi, cj -> aijd', self.tensors[0][j], np.diag(self.Lambda_hori[0][j]), np.diag(self.Lambda_vert[0][j]))
            # j = -1
            self.tensors[0][-1] = np.einsum('abc, bi -> aic', self.tensors[0][-1], np.diag(self.Lambda_vert[0][-1]))

            # i = 1 ~ n-2
            for i in range(1, self.n-1):
                # j = 0
                self.tensors[i][0] = np.einsum('abcd, bi, cj -> aijd', self.tensors[i][0], np.diag(self.Lambda_hori[i][0]), np.diag(self.Lambda_vert[i][0]))
                # j = 1 ~ m-2
                for j in range(1, self.m-1):
                    self.tensors[i][j] = np.einsum('abcde, ci, dj -> abije', self.tensors[i][j], np.diag(self.Lambda_hori[i][j]), np.diag(self.Lambda_vert[i][j]))
                # j = -1
                self.tensors[i][-1] = np.einsum('abcd, ci -> abid', self.tensors[i][-1], np.diag(self.Lambda_vert[i][-1]))

            # i = -1
            # j = 0
            self.tensors[-1][0] = np.einsum('abc, ai -> ibc', self.tensors[-1][0], np.diag(self.Lambda_hori[-1][0]))
            # j = 1 ~ m-2
            for j in range(1, self.m-1):
                self.tensors[-1][j] = np.einsum('abcd, bi -> aicd', self.tensors[-1][j], np.diag(self.Lambda_hori[-1][j]))
            # j = -1
            self.tensors[-1][-1] = self.tensors[-1][-1]

            self.is_GL = False




    def to_Gamma_Lambda(self, cut_dim = -1, debug = False):
        """
        转化为GL形式

        """
        if not self.is_GL:
            # horizontal
            # i = 0
            matrix = self.tensors[0][0].reshape(self.vd_hori[0, 0], self.vd_vert[0, 0] * self.pd[0, 0])
            u, lm, v = svd_cut(matrix, cut_dim)
            self.vd_hori[0, 0] = lm.size
            self.tensors[0][1] = np.tensordot(u, self.tensors[0][1], [[0], [0]])
            Lambda = [lm]
            self.tensors[0][0] = v.reshape(self.vd_hori[0, 0], self.vd_vert[0, 0], self.pd[0, 0])
            for j in range(1, self.m - 1):
                matrix = np.rollaxis(self.tensors[0][j], 1, 0)
                matrix = matrix.reshape(self.vd_hori[0, j], self.vd_hori[0, j-1] * self.vd_vert[0, j] * self.pd[0, j])
                u, lm, v = svd_cut(matrix, cut_dim)
                self.vd_hori[0, j] = lm.size
                self.tensors[0][j+1] = np.tensordot(u, self.tensors[0][j+1], [[0], [0]])
                Lambda += [lm]
                self.tensors[0][j] = v.reshape(self.vd_hori[0, j], self.vd_hori[0, j-1], self.vd_vert[0, j], self.pd[0, j])
                self.tensors[0][j] = np.rollaxis(self.tensors[0][j], 0, 1)
            self.Lambda_hori += [Lambda]

            # i = 1 ~ n-2
            for i in range(1, self.n - 1):
                matrix = np.rollaxis(self.tensors[i][0], 1, 0)
                matrix = matrix.reshape(self.vd_hori[i, 0], self.vd_vert[i-1, 0] * self.vd_vert[i, 0] * self.pd[i, 0])
                u, lm, v = svd_cut(matrix, cut_dim)
                self.vd_hori[i][0] = lm.size
                self.tensors[i][1] = np.tensordot(u, self.tensors[i][1], [[0], [0]])
                Lambda = [lm]
                self.tensors[i][0] = v.reshape(self.vd_hori[i, 0], self.vd_vert[i-1, 0], self.vd_vert[i, 0], self.pd[i, 0])
                for j in range(1, self.m - 1):
                    matrix = np.rollaxis(self.tensors[i][j], 2, 0)
                    matrix = matrix.reshape(self.vd_hori[i, j], self.vd_hori[i-1, j] * self.vd_vert[i-1, j] * self.vd_vert[i, j] * self.pd[i, j])
                    u, lm, v = svd_cut(matrix, cut_dim)
                    self.vd_hori[i][j] = lm.size
                    self.tensors[i][j+1] = np.tensordot(u, self.tensors[i][j+1], [[0], [0]])
                    Lambda += [lm]
                    self.tensors[i][j] = v.reshape(self.vd_hori[i, j], self.vd_hori[i-1, j], self.vd_vert[i-1, j], self.vd_vert[i, j], self.pd[i, j])
                    self.tensors[i][j] = np.rollaxis(self.tensors[i][j], 0, 2)
                self.Lambda_hori += [Lambda]

            # i = -1
            matrix = self.tensors[-1][0].reshape(self.vd_hori[-1, 0], self.vd_vert[-1, 0] * self.pd[-1, 0])
            u, lm, v = svd_cut(matrix, cut_dim)
            self.vd_hori[-1, 0] = lm.size
            self.tensors[-1][1] = np.tensordot(u, self.tensors[-1][1], [[0], [0]])
            Lambda = [lm]
            self.tensors[-1][0] = v.reshape(self.vd_hori[-1, 0], self.vd_vert[-1, 0], self.pd[-1, 0])
            for j in range(1, self.m - 1):
                matrix = np.rollaxis(self.tensors[-1][j], 1, 0)
                matrix = matrix.reshape(self.vd_hori[-1, j], self.vd_hori[-1, j-1] * self.vd_vert[-1, j] * self.pd[-1, j])
                u, lm, v = svd_cut(matrix, cut_dim)
                self.vd_hori[-1, j] = lm.size
                self.tensors[-1][j+1] = np.tensordot(u, self.tensors[-1][j+1], [[0], [0]])
                Lambda += [lm]
                self.tensors[-1][j] = v.reshape(self.vd_hori[-1, j], self.vd_hori[-1, j-1], self.vd_vert[-1, j], self.pd[-1, j])
                self.tensors[-1][j] = np.rollaxis(self.tensors[-1][j], 0, 1)
            self.Lambda_hori += [Lambda]
            
            # vertical
            # i = 1 ~ n-2
            for i in range(1, self.n - 1):
                matrix = self.tensors[i][0].reshape(self.vd_vert[i-1, 0], self.vd_hori[i, 0] * self.vd_vert[i, 0] * self.pd[i, 0])
                u, lm, v = svd_cut(matrix, cut_dim)
                self.vd_vert[i-1][0] = lm.size
                self.tensors[i-1][0] = np.tensordot(u, self.tensors[i-1][0], [[0], [-2]])
                self.tensors[i-1][0] = np.rollaxis(self.tensors[i-1][0], 0, -2)
                Lambda = [lm]
                self.tensors[i][0] = v.reshape(self.vd_vert[i-1][0], self.vd_hori[i, 0], self.vd_vert[i, 0], self.pd[i, 0])
                for j in range(1, self.m - 1):
                    matrix = np.rollaxis(self.tensors[i][j], 1, 0)
                    matrix = matrix.reshape(self.vd_vert[i-1][j], self.vd_hori[i, j-1] * self.vd_hori[i, j] * self.vd_vert[i, j] * self.pd[i, j])
                    u, lm, v = svd_cut(matrix, cut_dim)
                    self.vd_vert[i-1][j] = lm.size
                    self.tensors[i-1][j] = np.tensordot(u, self.tensors[i-1][j], [[0], [-2]])
                    self.tensors[i-1][j] = np.rollaxis(self.tensors[i-1][j], 0, -2)
                    Lambda += [lm]
                    self.tensors[i][j] = v.reshape(self.vd_vert[i-1][j], self.vd_hori[i, j-1], self.vd_hori[i, j], self.vd_vert[i, j], self.pd[i, j])
                    self.tensors[i][j] = np.rollaxis(self.tensors[i][j], 0, 1)
                matrix = self.tensors[i][-1].reshape(self.vd_vert[i-1, -1], self.vd_hori[i, -1] * self.vd_vert[i, -1] * self.pd[i, -1])
                u, lm, v = svd_cut(matrix, cut_dim)
                self.vd_vert[i-1][-1] = lm.size
                self.tensors[i-1][-1] = np.tensordot(u, self.tensors[i-1][-1], [[0], [-2]])
                self.tensors[i-1][-1] = np.rollaxis(self.tensors[i-1][-1], 0, -2)
                Lambda += [lm]
                self.tensors[i][-1] = v.reshape(self.vd_vert[i-1][-1], self.vd_hori[i, -1], self.vd_vert[i, -1], self.pd[i, -1])
                self.Lambda_vert += [Lambda]
            
            if debug:
                print(self.tensors[1][1].shape)
            # i = n-1
            matrix = np.rollaxis(self.tensors[-1][0], 1, 0)
            matrix = matrix.reshape(self.vd_hori[-1, 0], self.vd_vert[-1, 0] * self.pd[-1, 0])
            u, lm, v = svd_cut(matrix, cut_dim)
            self.vd_vert[-1][0] = lm.size
            self.tensors[-2][0] = np.tensordot(u, self.tensors[-2][0], [[0], [-2]])
            self.tensors[-2][0] = np.rollaxis(self.tensors[-2][0], 0, -2)
            Lambda = [lm]
            self.tensors[-1][0] = v.reshape(self.vd_hori[-1, 0], self.vd_vert[-1, 0], self.pd[-1, 0])
            self.tensors[-1][0] = np.rollaxis(self.tensors[-1][0], 0, 1)
            for j in range(1, self.m - 1):
                matrix = np.rollaxis(self.tensors[-1][j], 2, 0)
                matrix = matrix.reshape(self.vd_vert[-1][j], self.vd_hori[-1, j-1] * self.vd_hori[-1, j] * self.pd[-1, j])
                u, lm, v = svd_cut(matrix, cut_dim)
                self.vd_vert[-1][j] = lm.size
                self.tensors[-2][j] = np.tensordot(u, self.tensors[-2][j], [[0], [-2]])
                self.tensors[-2][j] = np.rollaxis(self.tensors[-2][j], 0, -2)
                Lambda += [lm]
                self.tensors[-1][j] = v.reshape(self.vd_vert[-1][j], self.vd_hori[-1, j-1], self.vd_hori[-1, j], self.pd[-1, j])
                self.tensors[-1][j] = np.rollaxis(self.tensors[-1][j], 0, 2)
            matrix = np.rollaxis(self.tensors[-1][-1], 1, 0)
            matrix = matrix.reshape(self.vd_hori[-1, -1], self.vd_vert[-1, -1] * self.pd[-1, -1])
            u, lm, v = svd_cut(matrix, cut_dim)
            self.vd_vert[-1][-1] = lm.size
            self.tensors[-2][-1] = np.tensordot(u, self.tensors[-2][-1], [[0], [-2]])
            self.tensors[-2][-1] = np.rollaxis(self.tensors[-2][-1], 0, -2)
            Lambda += [lm]
            self.tensors[-1][-1] = v.reshape(self.vd_hori[-1, -1], self.vd_vert[-1, -1], self.pd[-1, -1])
            self.tensors[-1][-1] = np.rollaxis(self.tensors[-1][-1], 0, 1)
            self.Lambda_vert += [Lambda]

            if debug:
                print(self.tensors[1][1].shape)
            self.is_GL = True
    
    def evolve_gate(self, gate, pos1, pos2, cut_dim = -1, debug = False):
        """
        在pos1, pos2上做gate操作，自动转化为Gamma-Lambda形式
         0    1
         |    |
          gate
         |    |
         2    3
        :param gate: the two-body gate to evolve the MPS
        :param pos1: the position of the first spin in the gate, input as (i, j)
        :param pos2: the position of the second spin in the gate, input as (i, j)
        
        """

        self.to_Gamma_Lambda(cut_dim=cut_dim, debug=debug)

        if pos1[0] == pos2[0]:
            hori = True
            if pos1[1] > pos2[1]:
                pos1 = pos2
        else:
            hori = False
            if pos1[0] > pos2[0]:
                pos1 = pos2
        """
        if debug:
            print(pos1)
            for i in range(0, 3):
                for j in range(0, 3):
                    print(self.tensors[i][j].shape)
        """
                    
        if hori:
            self.__evolve_gate_hori(gate, pos1, cut_dim=cut_dim, debug=debug)
        # else:
        # self.__evolve_gate_vert(gate, pos1, cut_dim=cut_dim, debug=debug)

    def __evolve_gate_hori(self, gate, pos, cut_dim = -1, debug = False):
        (i, j) = pos
        tensor1 = self.tensors[i][j]
        tensor2 = self.tensors[i][j+1]

        # i = 0
        if i == 0:
            # tensor1
            if j == 0:
                tensor1 = np.einsum('abc, bi -> iac', tensor1, np.diag(self.Lambda_vert[i][j]))
                tensor1 = tensor1.reshape(self.vd_vert[i][j], self.vd_hori[i][j] * self.pd[i][j])
            else:
                tensor1 = np.einsum('abcd, ai, cj -> ijbd', tensor1, np.diag(self.Lambda_hori[i][j-1]), np.diag(self.Lambda_vert[i][j]))
                tensor1 = tensor1.reshape(self.vd_hori[i][j-1] * self.vd_vert[i][j], self.vd_hori[i][j] * self.pd[i][j])
            # tensor2
            if j+1 == self.m-1:
                tensor2 = np.einsum('abc, bi -> iac', tensor2, np.diag(self.Lambda_vert[i][j+1]))
                tensor2 = tensor2.reshape(self.vd_vert[i][j+1], self.vd_hori[i][j] * self.pd[i][j+1])
            else:
                tensor2 = np.einsum('abcd, bi, cj -> jiad', tensor2, np.diag(self.Lambda_hori[i][j+1]), np.diag(self.Lambda_vert[i][j+1]))
                tensor2 = tensor2.reshape(self.vd_vert[i][j+1] * self.vd_hori[i][j+1], self.vd_hori[i][j] * self.pd[i][j+1])

        # i = 1 ~ n-2
        if 0 < i < self.n-1:
            # tensor1
            if j == 0:
                tensor1 = np.einsum('abcd, ai, cj -> ijbd', tensor1, np.diag(self.Lambda_vert[i-1][j]), np.diag(self.Lambda_vert[i][j]))
                tensor1 = tensor1.reshape(self.vd_vert[i-1][j] * self.vd_vert[i][j], self.vd_hori[i][j] * self.pd[i][j])
            else:
                tensor1 = np.rollaxis(tensor1, 2, 3)
                tensor1 = tensor1.reshape(self.vd_hori[i][j-1], self.vd_vert[i-1][j], self.vd_vert[i][j], self.vd_hori[i][j] * self.pd[i][j])
                tensor1 = np.einsum('abcd, ai, bj, ck -> ijkd', tensor1, np.diag(self.Lambda_hori[i][j-1]), np.diag(self.Lambda_vert[i-1][j]), np.diag(self.Lambda_vert[i][j]))
                tensor1 = tensor1.reshape(self.vd_hori[i][j-1] * self.vd_vert[i-1][j] * self.vd_vert[i][j], self.vd_hori[i][j] * self.pd[i][j])
            # tensor2
            if j+1 == self.m-1:
                tensor2 = np.einsum('abcd, ai, cj -> ijbd', tensor2, np.diag(self.Lambda_vert[i-1][j+1]), np.diag(self.Lambda_vert[i][j+1]))
                tensor2 = tensor2.reshape(self.vd_vert[i-1][j+1] * self.vd_vert[i][j+1], self.vd_hori[i][j] * self.pd[i][j+1])
            else:
                tensor2 = np.rollaxis(tensor2, 0, 3)
                tensor2 = tensor2.reshape(self.vd_vert[i-1][j+1], self.vd_hori[i][j+1], self.vd_vert[i][j+1], self.vd_hori[i][j] * self.pd[i][j+1])
                tensor2 = np.einsum('abcd, ai, bj, ck -> ijkd', tensor2, np.diag(self.Lambda_vert[i][j+1]), np.diag(self.Lambda_vert[i-1][j+1]), np.diag(self.Lambda_hori[i][j+1]))
                tensor2 = tensor2.reshape(self.vd_vert[i-1][j+1] * self.vd_hori[i][j+1] * self.vd_vert[i][j+1], self.vd_hori[i][j] * self.pd[i][j+1])
        
        # i = -1
        if i == self.n-1:
            # tensor1
            if j == 0:
                tensor1 = np.einsum('abc, bi -> iac', tensor1, np.diag(self.Lambda_vert[i-1][j]))
                tensor1 = tensor1.reshape(self.vd_vert[i-1][j], self.vd_hori[i][j] * self.pd[i][j])
            else:
                tensor1 = np.einsum('abcd, ai, cj -> ijbd', tensor1, np.diag(self.Lambda_hori[i][j-1]), np.diag(self.Lambda_vert[i-1][j]))
                tensor1 = tensor1.reshape(self.vd_hori[i][j-1] * self.vd_vert[i-1][j], self.vd_hori[i][j] * self.pd[i][j])
            # tensor2
            if j+1 == self.m-1:
                tensor2 = np.einsum('abc, bi -> iac', tensor2, np.diag(self.Lambda_vert[i-1][j+1]))
                tensor2 = tensor2.reshape(self.vd_vert[i-1][j+1], self.vd_hori[i][j] * self.pd[i][j+1])
            else:
                tensor2 = np.einsum('abcd, bi, cj -> jiad', tensor2, np.diag(self.Lambda_hori[i][j+1]), np.diag(self.Lambda_vert[i-1][j+1]))
                tensor2 = tensor2.reshape(self.vd_vert[i-1][j+1] * self.vd_hori[i][j+1], self.vd_hori[i][j] * self.pd[i][j+1])

        

        q1, r1 = np.linalg.qr(tensor1)
        q2, r2 = np.linalg.qr(tensor2)
        r1 = r1.reshape(q1.shape[1], self.vd_hori[i][j], self.pd[i][j])
        r2 = r2.reshape(q2.shape[1], self.vd_hori[i][j], self.pd[i][j+1])
        tensor = np.einsum('abcd, ijc, xyd, jy -> aibx', gate, r1, r2, np.diag(self.Lambda_hori[i][j]))
        tensor = tensor.reshape(self.pd[i][j] * q1.shape[1], self.pd[i][j+1] * q2.shape[1])
        r1, lm, r2 = svd_cut(tensor, cut_dim=cut_dim)
        self.Lambda_hori[i][j] = lm
        self.vd_hori[i][j] = lm.size
        r1 = r1.reshape(self.pd[i][j], q1.shape[1], self.vd_hori[i][j])
        r1 = np.rollaxis(r1, 0, 2)
        r2 = r2.reshape(self.vd_hori[i][j], self.pd[i][j+1], q2.shape[1])
        r2 = np.rollaxis(r2, 2, 0)
        tensor1 = np.tensordot(q1, r1, [[1], [0]])
        tensor2 = np.tensordot(q2, r2, [[1], [0]])


        # i = 0
        if i == 0:
            # tensor1
            if j == 0:
                tensor1 = tensor1.reshape(self.vd_vert[i][j], self.vd_hori[i][j], self.pd[i][j])
                tensor1 = np.einsum('iac, bi -> abc', tensor1, np.diag(1/self.Lambda_vert[i][j]))
            else:
                tensor1 = tensor1.reshape(self.vd_hori[i][j-1], self.vd_vert[i][j], self.vd_hori[i][j], self.pd[i][j])
                tensor1 = np.einsum('ijbd, ai, cj -> abcd', tensor1, np.diag(1/self.Lambda_hori[i][j-1]), np.diag(1/self.Lambda_vert[i][j]))
            # tensor2
            if j+1 == self.m-1:
                tensor2 = tensor2.reshape(self.vd_vert[i][j+1], self.vd_hori[i][j], self.pd[i][j+1])
                tensor2 = np.einsum('iac, bi -> abc', tensor2, np.diag(1/self.Lambda_vert[i][j+1]))
            else:
                tensor2 = tensor2.reshape(self.vd_vert[i][j+1], self.vd_hori[i][j+1], self.vd_hori[i][j], self.pd[i][j+1])
                tensor2 = np.einsum('jiad, bi, cj -> abcd', tensor2, np.diag(1/self.Lambda_hori[i][j+1]), np.diag(1/self.Lambda_vert[i][j+1]))

        # i = 1 ~ n-2
        if 0 < i < self.n-1:
            # tensor1
            if j == 0:
                tensor1 = tensor1.reshape(self.vd_vert[i-1][j], self.vd_vert[i][j], self.vd_hori[i][j], self.pd[i][j])
                tensor1 = np.einsum('ijbd, ai, cj -> abcd', tensor1, np.diag(1/self.Lambda_vert[i-1][j]), np.diag(1/self.Lambda_vert[i][j]))
            else:
                tensor1 = tensor1.reshape(self.vd_hori[i][j-1], self.vd_vert[i-1][j], self.vd_vert[i][j], self.vd_hori[i][j] * self.pd[i][j])
                tensor1 = np.einsum('ijkd, ai, bj, ck -> abcd', tensor1, np.diag(1/self.Lambda_hori[i][j-1]), np.diag(1/self.Lambda_vert[i-1][j]), np.diag(1/self.Lambda_vert[i][j]))
                tensor1 = tensor1.reshape(self.vd_hori[i][j-1], self.vd_vert[i-1][j], self.vd_vert[i][j], self.vd_hori[i][j], self.pd[i][j])
                tensor1 = np.rollaxis(tensor1, 3, 2)

            # tensor2
            if j+1 == self.m-1:
                tensor2 = tensor2.reshape(self.vd_vert[i-1][j+1], self.vd_vert[i][j+1], self.vd_hori[i][j], self.pd[i][j+1])
                tensor2 = np.einsum('ijbd, ai, cj -> abcd', tensor2, np.diag(1/self.Lambda_vert[i-1][j+1]), np.diag(1/self.Lambda_vert[i][j+1]))
            else:
                tensor2 = tensor2.reshape(self.vd_vert[i-1][j+1], self.vd_hori[i][j+1], self.vd_vert[i][j+1], self.vd_hori[i][j] * self.pd[i][j+1])
                tensor2 = np.einsum('ijkd, ai, bj, ck -> abcd', tensor2, np.diag(1/self.Lambda_vert[i][j+1]), np.diag(1/self.Lambda_vert[i-1][j+1]), np.diag(1/self.Lambda_hori[i][j+1]))
                tensor2 = tensor2.reshape(self.vd_vert[i-1][j+1], self.vd_hori[i][j+1], self.vd_vert[i][j+1], self.vd_hori[i][j], self.pd[i][j+1])
                tensor2 = np.rollaxis(tensor2, 3, 0)

        # i = -1
        if i == self.n-1:
            # tensor1
            if j == 0:
                tensor1 = tensor1.reshape(self.vd_vert[i-1][j], self.vd_hori[i][j], self.pd[i][j])
                tensor1 = np.einsum('iac, bi -> abc', tensor1, np.diag(1/self.Lambda_vert[i-1][j]))
            else:
                tensor1 = tensor1.reshape(self.vd_hori[i][j-1], self.vd_vert[i-1][j], self.vd_hori[i][j], self.pd[i][j])
                tensor1 = np.einsum('ijbd, ai, cj -> abcd', tensor1, np.diag(1/self.Lambda_hori[i][j-1]), np.diag(1/self.Lambda_vert[i-1][j]))
            # tensor2
            if j+1 == self.m-1:
                tensor2 = tensor2.reshape(self.vd_vert[i-1][j+1], self.vd_hori[i][j], self.pd[i][j+1])
                tensor2 = np.einsum('iac, bi -> abc', tensor2, np.diag(1/self.Lambda_vert[i-1][j+1]))
            else:
                tensor2 = tensor2.reshape(self.vd_vert[i-1][j+1], self.vd_hori[i][j+1], self.vd_hori[i][j], self.pd[i][j+1])
                tensor2 = np.einsum('jiad, bi, cj -> abcd', tensor2, np.diag(1/self.Lambda_hori[i][j+1]), np.diag(1/self.Lambda_vert[i-1][j+1]))

        self.tensors[i][j] = tensor1
        self.tensors[i][j+1] = tensor2
    
    def TEBD(self, hamiltonion, tau = 1e-4, cut_dim = -1, debug = False):
        """
        单步时间演化
        :param hamiltonion: 时间演化二体哈密顿量
        :param tau: 模拟步长
        :param cut_dim: 裁剪维数，如果设置tol则为初始裁剪维数
        :param times: 模拟次数

        """
        hamiltonion = expm(-1j * tau * hamiltonion) # e^(-iHt)
        if np.linalg.norm(np.imag(hamiltonion)) < 1e-15:
            hamiltonion = np.real(hamiltonion)
        hamiltonion = hamiltonion.reshape(self.pd[0][0], self.pd[0][0], self.pd[0][0], self.pd[0][0])
        for i in range(0, self.n):
            for j in range(0, self.m-1):
                self.evolve_gate(hamiltonion, (i, j), (i, j+1), cut_dim=cut_dim, debug=debug)
    
    def normalize(self):
        norm = self.inner(self)
        norm = np.power(norm, 1/(2 * self.n * self.m))
        for i in range(0, self.n):
            for j in range(0, self.m):
                self.tensors[i][j] = self.tensors[i][j]/norm

            
def svd_cut(matrix, cut_dim = -1):
    u, lm, v = np.linalg.svd(matrix, full_matrices=False)
    if 0 < cut_dim < lm.size:
        u = u[:, :cut_dim]
        lm = lm[:cut_dim]
        v = v[:cut_dim, :]
    return u, lm, v



