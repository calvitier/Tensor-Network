import numpy as np
import copy
import torch as tc
import Basicfun as bf
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

    def __init__(self, d, chi, length):
        """
        :param d: 物理指标维数
        :param chi: 辅助指标截断维数
        :param length: 张量个数

        :param d: physical bond dimension
        :param chi: virtual bond dimension cut-off
        :param length: number of tensors
        
        """
        self.d = d  # physical bond dimension
        self.chi = chi  # virtual bond dimension cut-off
        self.length = length  # number of tensors
        self.tensors = list()  # tensors in MPS
        self.physdim = list()  # physical bond dimensions
        self.virtdim = list()  # virtual bond dimensions
        self.center = -1  # 正交中心 当center为负数时，MPS非中心正交
        self.dtype = None
        self.init_tensors()

    def init_tensors(self, tensors = None):
        """
        :param tensors: 默认为空，进行随机初始化，或输入指定张量列表，需满足注1要求

        """
        if tensors is None:
            self.tensors = [np.random.randn(self.d, self.chi)]
            for n in range(1,self.length-1):
                self.tensors += [np.random.randn(self.chi, self.d, self.chi)]
            self.tensors += [np.random.randn(self.chi, self.d)]
            self.virtdim = [self.chi] * (self.length - 1)
        
        else:
            assert len(tensors) >= self.length
            for n in range(0,self.length-1):
                assert tensors[n].shape[-1] == tensors[n+1].shape[0]

            for n in range(0,self.length):
                self.tensors[n] = copy.deepcopy(tensors[n])
            
            for n in range(0,self.length-1):
                if tensors[n].shape[0] > self.chi:
                    self.cut_off(n)    
                else:
                    self.virtdim += [self.tensors[n].shape[0]]
        
        self.physdim = [self.d] * self.length


    def mps2tensor(self):
        tensor = self.tensors[0]
        for n in range(1, self.length):
            tensor_ = self.tensors[n]
            tensor = np.tensordot(tensor, tensor_, [[-1], [0]])
        return np.squeeze(tensor)

    def get_tensor(self, nt, if_copy=True):
        if if_copy:
            return copy.deepcopy(self.tensors[nt])
        else:
            return self.tensors[nt]
    

    def mps2vec(self):
        vec = self.mps2tensor().reshape((-1,))
        return vec

    
    def center_orth(self, center):
        assert 0 <= center < self.length


    def cut_off(self, n):
        self.virtdim += [self.chi]
        # 先写中心正交


    

