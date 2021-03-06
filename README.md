# Tensor-Network
这是一个本科双学位毕业设计，目标为利用张量网络模拟二维量子体系。

## v0.2简介
当前版本包含一维MPS态模拟和二维PEPS模拟，由`MPS`、`PEPS`、`Operator`组成。

## MPS
### mps类
* 重要变量
  * `length`: MPS长度
  * `tensors`: 为list，储存MPS中每个小张量
  * `center`: 正交中心，-1表示不为中心正交形式
  * `physdim`: 物理指标list
  * `virtdim`: 虚拟指标list

* 重要成员函数
  * `init_tensor`：输入指定张量列表，需满足注中要求
  * `init_rand`: 随机初始化
  * `mps2tensor`: 收缩所有虚拟指标，返回MPS代表张量
  * `mps2vec`: 收缩所有虚拟指标，将MPS代表张量转化为向量输出
  * `inner`: MPS态间做内积
  * `center_orth`：指定正交中心，对MPS进行中心正交化，或移动MPS正交中心至指定位置
  * `evolve_gate`: 演化一次门操作
  * `TEBD`: 时间演化模拟
  

### 其他
* `ground_state`: 返回相邻二体哈密顿量对应基态mps对象

## PEPS

### peps类

* 重要变量
  - `n`：行数 >= 2
  - `m`：列数 >= 2
  - `shape`: (n, m)
  - `tensors`: 二维list，储存PEPS中每个小张量
  - `pd`: 物理指标矩阵，大小为shape
  - `vd_hori`: 水平虚拟指标维数矩阵，n * m-1
  - `vd_vert`: 竖直虚拟指标维数矩阵，n-1 * m
  - `is_GL`: True表示为Gamma-Lambda形式，False表示不为
  - `Lambda_hori`：二维list，储存Gamma-Lambda形式时的Lambda对角阵，is_GL = False时为空，n * m-1
  - `Lambda_vert`：二维list，储存Gamma-Lambda形式时的Lambda对角阵，is_GL = False时为空，n-1 * m

* 重要成员函数
  - `init_tensor`：输入指定张量列表，需满足注中要求
  - `init_rand`: 随机初始化
  - `inner`: PEPS态间做内积
  - `to_Gamma_Lambda`：PEPS转化为Gamma-Lambda形式
  - `to_PEPS`：Gamma-Lambda形式转化为PEPS
  - `evolve_gate`: 演化一次门操作
  - `TEBD`: 时间演化模拟
## Operator
算符和哈密顿量初始化  

* `spin_operator`: 自旋算符，可扩充
* `heisenberg_hamilt`: 海森堡自旋哈密顿量
