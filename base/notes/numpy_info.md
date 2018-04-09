### Numpy的使用手册
---
<font size=4>[1.创建数据](#1)</font><br>
<font size=4>[2.I/O操作](#2)</font><br>
<font size=4>[3.数据类型](#3)</font><br>
<font size=4>[4.帮助](#4)</font><br>
<font size=4>[5.数组的数学方法](#5)</font><br>
<font size=4>[6.数组的复制](#6)</font><br>
<font size=4>[7.数组的排序](#7)</font><br>
<font size=4>[8.子集构造、切片、索引](#8)</font><br>
<font size=4>[9.数组的操作](#9)</font><br>

---

<h4 id="1">1.创建数据</h4>

  ```python
  import numpy as np
  a = np.array([1,2,3])
  b = np.array([(1,2,3,4),(4,5,6)],dtype = Float)
  c = np.array([[(1,2,33,4),(5,6,7)],[(2,3,4),(7,8,9)]],dtype = Float)
  #初始化
  np.zeros((2,3)) #创建一个2行3列的元素为0的数组
  np.ones((2,3,4),dtype = np.int16) #创建2个元素为1的3行4列的矩阵
  d = np.arange(10,25,2) #创建一个10到25步距为2的数组
  np.linespace(0,2,9)  # 在指定的间隔内返回均匀间隔的数字
  e = np.full((3,2),7) #创建一个由常数填充的3行2列矩阵
  np.eye(6) #创建单位矩阵，函数中的参数n，则创建n * n的单位矩阵
  np.random.random((3,2)) #产生一个3行2列的随机数组
  np.empty((3,2))
  ```

<h4 id="2">2.I/O操作</h4>

  ```python
  import numpy as np
  np.save("A.npy",A)   #如果文件路径末尾没有扩展名.npy，该扩展名会被自动加上。
  f = np.load("A.npy")
  np.savez('A.npy') #将多个数组保存到一个文件中,第一个参数是文件名,其后的参数都是需要保存的数组
  np.loadtxt('a.txt') #加载txt文件
  np.genfromtxt('my_file.csv',delimiter=',')
  np.savetext('myarray.txt',a,delimiter=" ")
  ```

<h4 id="3">3.数据类型</h4>

  ```python
  import numpy as np
  np.bool #用一位存储的布尔类型（值为TRUE或FALSE）
  np.inti #由所在平台决定其精度的整数（一般为int32或int64）
  np.int8 #整数，范围为128至127
  np.int16 #整数，范围为32 768至32 767
  np.int32 #整数，范围为231至231 1
  np.int64 #整数，范围为263至263 1
  np.uint8 #无符号整数，范围为0至255
  np.uint16 #无符号整数，范围为0至65 535
  np.uint32 #无符号整数，范围为0至2321
  np.uint64 #无符号整数，范围为0至2641
  np.float16 #半精度浮点数（16位）：其中用1位表示正负号，5位表示指数，10位表示尾数
  np.float32 #单精度浮点数（32位）：其中用1位表示正负号，8位表示指数，23位表示尾数
  np.float64或float #双精度浮点数（64位）：其中用1位表示正负号，11位表示指数，52位表示尾数
  np.complex64 #复数，分别用两个32位浮点数表示实部和虚部
  np.complex128或complex #复数，分别用两个64位浮点数表示实部和虚部
  np.object
  np.string_
  np.unicode_
  ```

<h4 id="4">4.帮助及获得数组的属性</h4>

  ```python
  import numpy as np
  np.info(np.ndarry.dtype) #获得np.ndarry.dtype的帮助信息
  np_demo.shape #shape函数是numpy.core.fromnumeric中的函数，它的功能是读取矩阵的长度
  np_demo.ndim #数组的维度数
  np_demo.size #数组的尺寸
  np_demo.dtype #类型
  np_demo.dtype.name #数据类型名
  np_demo.astype(int)   #重新设置数据类型
  size(np_demo) #数组的长度
  ```

<h4 id="5">5.数组的数学方法</h4>

  ```python
  import numpy as np
  # 算术运算
  np_c = np_a - np_b
  np.subtract(a,b)
  np_C = np_a + np_b
  np.add(b,a)
  np_C = np_a / np_b
  np.divide(a,b)
  np_C = np_a * np_b
  np.exp(a) #以自然数e为底的指数函数
  np.sqrt(a) #求平方根
  np.sin(a)
  np.cos(a)
  np.log(a)
  e.dot(f) #数组e与数组f矩阵乘
  # 比较运算
  a == b
  a < b
  np.array_equal(a,b)
  # 聚合函数
  a.sum()
  a.min()
  a.max(axis=0)
  a.cumsum(axis=1) #计算轴向元素累加和，返回由中间结果组成的数组
  a.mean()
  a.median()
  a.corrcoef() #得到相关系数矩阵
  np.std(a) #标准方差
  ```

<h4 id="6">6.数组的复制</h4>

  ```python
  import numpy as np
  h = a.view()
  np.copy(a)
  h = a.copy()
  ```

<h4 id="7">7.数组的排序</h4>

  ```python
  import numpy as np
  a.sort()
  c.sort(axis=0)
  ```

<h4 id="8">8.子集构造、切片、索引</h4>

  ```python
  import numpy as np
  a[1]
  b[1,2] #构造子集
  a[0:2]
  b[1:8,1]
  b[:2]
  c[1,...]
  a[::-1]
  a[a<2] #boolean索引
  b[[1,0,1,0],[0,11,2,0]]
  b[[1,0,1,0]][:,[0,1,2,0]]
  ```

<h4 id="9">9.数组的操作</h4>

  ```python
  import numpy as np
  #置换阵列
  i = np.transpose(b)
  i.T  # 转置
  #修改数组形状
  b.ravel()
  g.reshape(3,-2)
  #添加移除元素
  h.resize((2,6))
  np.append(h,g)
  np.insert(a,1,5)
  np.delete(a,[1])
  #合并
  np.concatenate((a,b),axis=0)
  np.vstack((a,b))
  np.r_[e,f]
  np.hstack((e,f))
  np.column_stack((a,d))
  np.c_[a,d]
  #分裂
  np.hsplit(a,3)
  np.vsplit(c,2)
  ```
