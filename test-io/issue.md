# 对于从数据集读入二分图(doc（左节点, from）, word(右节点, to))问题的说明
## 负数方案不可行
我们想到把二分图的左节点设为负数，右节点设为正数

虽然vid为负数在数据类型上没有问题，但是实际上会出现内部错误

参考 GraphLite的实现

https://github.com/schencoding/GraphLite/blob/42f53d0c0cb295302fb1a9c4ac82dd999e0d410d/GraphLite-0.20/engine/Worker.cc

其中假设了from和to节点的vid都是从0开始连续递增的

如果，一共有cnt个worker(不含master), 那么节点将会被分配给 wid = vid mod cnt 的worker

且节点在该worker内部的index为 vid/cnt

如果worker需要输出vid，会把index还原为 vid, vid = index * cnt + wid

所以，如果给负数的index， 会有这样的问题或者其他内部问题，导致卡死。

## 采用连续编号
把数据集中的doc和word从0开始连续编号， word在doc后面连续。

doc会被读入为from， word会被读成to

把from和to都用addVertex()函数添加进去，初始化赋值为不同的vertex_data值

from节点的vertex_data.flag = IS_DOC, to节点的vertex_data.flag = IS_WORD

本目录下有相应的测试数据和输出结果。

代码见 commit:
 https://github.com/suxi1314/cgs_lda/tree/456d42d7aa06b4b125060b15e373770b7cd3c72f
