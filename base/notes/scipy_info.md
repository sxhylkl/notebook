### Scipy的使用手册
---
<font size=4>[1.结合Numpy使用](#1)</font><br>
<font size=4>[2.帮助](#2)</font><br>
<font size=4>[3.线性代数](#3)</font><br>

---

<h4 id="1">1.结合Numpy使用</h4>

  ```python
  import numpy as np
  a = np.array([1,2,3])
  b = np.array([(1+5j,2j,3j),(4j,5j,6j)])
  c = np.array([[(1.5,2,3),(4,5,6)],[(3,2,1),(4,5,6)]])
  # 索引技巧
  np.mgrid[0:5,0:5]
  np.ogrid[0:2,0:2]
  np.r_[3,[0]*5,-1:1:10j]
  np.c_[b,c]
  # 形状操作 Shape Manipulation
  np.transpose(b)
  np.flatten()
  np.hstack((b,c))
  np.vstack((a,b))
  np.hsplit(c,2)
  np.vpslit(d,2)
  # 多项式 Polynomials
  from numpy import polyld
  p = polyld([3,4,5])
  # 矢量方法 Vectorizing Functions
  def my_fucntion(a):
    if a < 0:
      return a*2
    else:
      return a/2
  np.vectorize(my_function)
  # 类型变换 Type Handing
  np.real(b)
  np.imag(b)
  np.real_id_close(c,tol=1000)
  np.cast['f'](np.pi)
  # 其他有用的函数
  np.angle(b,deg=True)
  g = np.linspace(0,np.pi,num=5)
  g[3:] += np.pi
  np.unwrap(g)
  np.logspace(0,10,3)
  np.select([c<4],[c*2])
  misc.factorial(a)
  misc.comb(10,3,exact=True)
  misc.central_diff_weights(3)
  misc.derivative(my_function,1.0)
  ```

<h4 id="2">2.帮助</h4>

  ```python
  help(scipy.linalg.diagsvd)
  np.info(np.matrix)
  ```

<h4 id="3">3.线性代数</h4>

  ```python
  from scipy import linalg,sparse
  # 矩阵创建Creating Matrices
  A = np.matrix(np.random.random((2,2)))
  B = np.asmatrix(b)
  C = np.mat(np.random.random((10,5)))
  D = np.mat([[3,4],[5,6]])  
  # 基础矩阵的例程 Basic Matrix Routines
  # 相反的 Inverse
  A.I
  linalg.inv(A)
  # 反转 Transposition
  A.T
  A.H
  # Trace
  np.trace(A)
  # 标准、规范 Norm
  linalg.norm(A)
  linalg.norm(A,1)
  linalg.norm(A,np.inf)
  # Rank
  np.linalg.matrix_rank(C)
  # Determinant
  linalg.det(A)
  # 解决线性问题 Solving linear problems
  linalg.solve(A,b)
  E = np.mat(a).T
  linalg.lstsq(F,E)
  # Generalized Inverse
  linalg.pinv(C)
  linalg.pinv2(C)
  # 创建稀疏矩阵 Creating Sparse Matrices
  F = np.eye(3,k=1)
  G = np.mat(np.identity(2))
  C[C>0.5] = 0
  H = sparse.csr_matrix(C)
  I = sparse.csc_matrix(D)
  J = sparse.dok_matrix(A)
  E.todense()
  sparse.isspamatrix_csc(A)
  # 稀疏矩阵例程 Sparse Matrix Routine
  # 相反的 Inverse
  sparse.linalg.inv(I)
  # Normal
  sparse.linalg.norm(I)
  # Solving linear problems
  sparse.linalg.spsolve(H,T)
  # 稀疏矩阵函数 Sparse Matrix Fucntion
  sparse.linalg.expm(I)
  # 矩阵函数 Matrix Fucntions
  # 添加 Addition
  np.add(A,D)
  # 减法 Subtracion
  np.subtract(A,D)
  # 除法 Division
  np.divide(A,D)
  # 乘法 Multiplication
  A @ D
  np.multiply(D,A)
  np.dot(A,D)
  np.vdot(A,D)
  np.inner(A,D)
  np.outer(A,D)
  np.tensordot(A,D)
  np.kron(A,D)
  # 指数函数 Exponential Functions
  linalg.expm(A)
  linalg.expm2(A)
  linalg.expm3(D)
  # 对数函数 Exponential Fucntions
  linalg.logm(A)
  # 三角函数 Trigonometric Functions
  linalg.sinm(D)
  linalg.cosm(D)
  linalg.tanm(A)
  # 双曲三角函数 Hyperbolic Trigonometric Functions
  linalg.sinhm(D)
  linalg.coshm(D)
  linalg.tanhm(A)
  # 矩阵符号函数 Matrix Sign Function
  np.signm(A)
  # 矩阵平方根 Matirx Square Root
  linalg.sqrtm(A)
  # 任意函数 Arbitrary Functions
  linalg.funm(A,lambda x: x*x)
  # 分解 Decompositions
  # 特征向量 Eigenvalues and Eigenvectors
  la,v = linalg.eig(A)
  l1,l2 = la
  v[:,0]
  v[:,1]
  linalg.eigvals(A)
  # 奇异值分解 Singular Value Decomposition
  U,s,Vh = linalg.svd(B)
  M,N = B.shape
  Sig = linalg.diagsvd(s,M,N)
  # LU分解 LU Decomposition
  P,L,U = linalg.lu(C)
  # Sparse Matrix Decompositions
  la,v = sparse.linalg.eigs(F,1)
  sparse.linalg.svds(H,2)
  ```
