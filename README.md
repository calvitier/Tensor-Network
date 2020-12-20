# Tensor-Network
这是一个本科双学位毕业设计，目标为利用张量网络模拟二维量子体系。

## v0.1简介
当前版本只包含一维MPS态模拟，由`MPS`、`Operator`组成。

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

## Operator
算符和哈密顿量初始化  

* `spin_operator`: 自旋算符，可扩充
* `heisenberg_hamilt`: 海森堡自旋哈密顿量
