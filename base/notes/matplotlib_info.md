### Matplotlib的使用手册
---
<font size=4>[1.例子](#1)</font><br>
<font size=4>[2.准备数据](#2)</font><br>
<font size=4>[3.创建图](#3)</font><br>
<font size=4>[4.画图步骤](#4)</font><br>
<font size=4>[5.自定义图片](#5)</font><br>
<font size=4>[6.保存图片](#6)</font><br>
<font size=4>[7.显示图片](#7)</font><br>
<font size=4>[8.关闭、清理](#8)</font><br>

---

<h4 id="1">1.例子</h4>

  ```python
  import matplotlib.pyplot as plt
  x = [1,2,3,4] # 步骤1
  y = [10,20,25,30] # 步骤1
  flg = plt.figure() # 步骤2
  ax = flg.add_subplot(111) # 步骤3
  ax.plot(x,y,color='lightblue',linewidth=3) # 步骤3、4
  ax.scatter([2,4,6],[5,15,25],color='darkgreen',marker='^')
  ax.ser_xlim(1,6.5)
  plt.saveflg('foo.png')
  plt.show()   # 步骤6
  ```

<h4 id="2">2.准备数据</h4>

  ```python
  # 一维
  import numpy as np
  x = np.linspace(0,10,10)
  y = np.cos(x)
  z = np.sin(x)
  # 二维
  data = 2 * np.random.random((10,10))
  data2 = 3 * np.random.random((10,10))
  Y,X = np.mgrid[-3:3:100j,-3,3,100j]
  U = -1-X**2+Y
  V = 1+X-Y**2
  from matplotlib.cbook import get_sample_data
  img = np.load(get_sample_data('axes_grid/bivariate_normal.npy'))
  ```

<h4 id="3">3.创建图</h4>

  ```python
  import matplotlib.pyplot as plt
  # Figure
  flg = plt.figure()
  flg2 = plt.figure(figsize=plt.flgaspect(2.0))
  # Axes
  flg.add_axes()
  ax1 = flg.add_subplot(221)
  ax3 = flg.add_subplot(212)
  flg3,axes = plt.subplots(nrows=2,ncols=2)
  flg4,axes2 = plt.subplots(ncols=3)
  ```

<h4 id="4">4.画图步骤</h4>

  ```python
  # 1维
  flg,ax = plt.subplots()
  lines = ax.plot(x,y)
  ax.scatter(x,y)
  axes[0,0].bar([1,2,3],[4,5,6])
  axes[1,0].barh([0.5,1,2.5],[0,1,2])
  axes[1,1].axhline(0.45)
  axes[0,1].axvline(0.65)
  ax.fill(x,y,color='blue')
  ax.fill_between(x,y,color='yellow')
  # 2维
  flg,ax = plt.subplots()
  im = ax.imshow(img,cmap='gist_earth',interpolation='nearest',vmin=-2,vmax=2)
  axes2[0].pcolor(data2)
  axes2[0].pcolormesh(data)
  CS = plt.contour(Y,X,U)
  axes2[2].contourf(data1)
  axes2[2] = ax.clabel(CS)
  # 矢量场
  axes[0,1].arrow(0,0,0.5,0.5)
  axes[1,1].quiver(y,z)
  axes[0,1].streamplot(X,Y,U,V)
  # 数据分布
  ax1.hist(y)
  ax3.boxplot(y)
  ax3.violinplot(z)
  ```

<h4 id="5">5.自定义图片</h4>

  ```python
  # Color,Color Bars & Color Maps
  plt.plot(x,x,x,x**2,x,x**3)
  ax.plot(x,y,alpha=0.4)
  ax.plot(x,y,c='k')
  flg.colorbar(im,orientation='horizontal')
  im = ax.imshow(img,cmap='seismic')
  # Makers标识
  flg,ax = plt.subplots()
  ax.scatter(x,y,marker='.')
  ax.plot(x,y,marker='o')
  # Linestyles 线性绘制
  plt.plot(x,y,linewidth=4.0)
  plt.plot(x,y,ls='solid')
  plt.plot(x,y,ls='--')
  plt.plot(x,y,'--',x**2,y**2,'-.')
  plt.setp(lines,color='r',linewidth=4.0)
  # Text & Annotation
  ax.text(1,-2.1,'Example Grapy',style='italic')
  ax.annotate("Sine",xy=(8,0),xyciirds='data',xytext=(10.5,0),textcoords='data',arrowprops=dict(arrowstyle='->',connectionstyle='arc3'),)
  # MathText
  plt.title(r'$sigma_i=15$',fontsize=20)
  # Limits,Legends & Layouts
  # Limits & Autoscaling
  ax.margins(x=0.0,y=0.1)
  ax.axis('equal')
  ax.set(xlim=[0,10.5],ylim=[-1.5,1.5])
  ax.set_xlim(0,10.5)
  # Legends
  ax.set(title='An Example Axes',ylabel='Y-Axis',xlabel='X-Axis')
  ax.legend(loc='best')
  # Ticks
  ax.xaxis.set(ticks=range(1,5),ticklabels=[3,100,-12,'fool'])
  ax.tick_params(axis='y',direction='inout',length=10)
  # Subplot Spacing 间距
  fig3.subplots_adjust(wspace=0.5,hspace=0.3,left=0.125,right=0.9,top=0.9,bottom=0.1)
  fig.tight_layout()
  # Axis Spines
  ax1.spines['top'].set_visible(False)
  ax1.spines['bottom'].set_position(('outward',10))
  ```

<h4 id="6">6.保存图片</h4>

  ```python
  plt.savefig('foo.png')
  plt.saveflg('foo.png',transparent=True)
  ```

<h4 id="7">7.显示图片</h4>

  ```python
  plg.show()
  ```

<h4 id="8">8.关闭、清理</h4>

  ```python
  plt.cla()
  plt.clf()
  plt.close()
  ```
