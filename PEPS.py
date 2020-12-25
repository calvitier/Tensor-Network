import copy
import numpy as np
import torch as tc
from scipy.linalg import expm

class peps:

    """
    ！！！重要变量！！！
    - n：行数
    - m：列数
    - shape: (n, m)
    - tensors: 二维list，储存PEPS中每个小张量
    - physdim: 物理指标矩阵，大小为shape
    - virtdim_horizontal: 水平虚拟指标矩阵，n * m-1
    - virtdim_vertical: 竖直虚拟指标矩阵，n-1 * m
    - is_GL: 1表示为Gamma-Lambda形式，0表示不为
    - Lambda：二维list，储存Gamma-Lambda形式时的Lambda对角阵，is_GL = 0时为空

    ！！！重要成员函数！！！
    - init_tensor：输入指定张量列表，需满足注中要求
    - init_rand: 随机初始化
    - peps2tensor: 收缩所有虚拟指标，返回MPS代表张量
    - peps2vec: 收缩所有虚拟指标，将MPS代表张量转化为向量输出
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
        self.is_GL = 0
        self.physdim = np.zeros(self.shape)
        self.virtdim_horizontal = np.zeros((self.n, self.m - 1))
        self.virtdim_vertical = np.zeros((self.n - 1, self.m))

        # 物理指标
        for i in range(0, self.n):
            for j in range(0, self.m):
                self.physdim[i][j] = self.tensors[i][j].shape[-1]

        # 虚拟指标
        # 第一行 i = 0
        self.virtdim_horizontal[0][0] = self.tensors[0][0].shape[0]
        self.virtdim_vertical[0][0] = self.tensors[0][0].shape[1]
        for j in range(1, self.m - 1):
            self.virtdim_horizontal[0][j] = self.tensors[0][j].shape[1]
            self.virtdim_vertical[0][j] = self.tensors[0][j].shape[2]
        self.virtdim_vertical[0][-1] = self.tensors[0][-1].shape[1]

        # i = 1 ~ n-2
        for i in range(1, self.n - 1):
            self.virtdim_horizontal[i][0] = self.tensors[i][0].shape[1]
            self.virtdim_vertical[i][0] = self.tensors[i][0].shape[2]
            for j in range(1, self.m - 1):
                self.virtdim_horizontal[i][j] = self.tensors[i][j].shape[2]
                self.virtdim_vertical[i][j] = self.tensors[i][j].shape[3]
            self.virtdim_vertical[i][-1] = self.tensors[i][-1].shape[2]

        # i = n-1
        self.virtdim_horizontal[-1][0] = self.tensors[0][0].shape[0]
        for j in range(1, self.m - 1):
            self.virtdim_horizontal[-1][j] = self.tensors[0][j].shape[1]

        
    @classmethod
    def init_rand(cls, physdim, virtdim, shape):
        """
        随机初始化

        """
        n = shape[0]
        m = shape[1]
        tensors = list()

        # i = 0
        tensor = [np.random.rand(virtdim, virtdim, physdim)]
        for _ in range(1, m - 1):
            tensor += [np.random.rand(virtdim, virtdim, virtdim, physdim)]
        tensor += [np.random.rand(virtdim, virtdim, physdim)]
        tensors += [tensor]

        # i = 1 ~ n-2
        for _ in range(1, n - 1):
            tensor = list()
            tensor += [np.random.rand(virtdim, virtdim, virtdim, physdim)]
            for __ in range(1, m - 1):
                tensor += [np.random.rand(virtdim, virtdim, virtdim, virtdim, physdim)]
            tensor += [np.random.rand(virtdim, virtdim, virtdim, physdim)]
            tensors += [tensor]

        # i = n-1
        tensor = [np.random.rand(virtdim, virtdim, physdim)]
        for _ in range(1, m - 1):
            tensor += [np.random.rand(virtdim, virtdim, virtdim, physdim)]
        tensor += [np.random.rand(virtdim, virtdim, physdim)]
        tensors += [tensor]

        return cls(tensors)

    @classmethod
    def init_tensors(cls, tensors):  
        """
        暂无格式检查
        """
        return cls(tensors)
