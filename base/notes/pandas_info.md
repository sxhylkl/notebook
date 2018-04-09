### Pandas的使用手册
---
<font size=4>[1.创建DataFrame](#1)</font><br>
<font size=4>[2.数据整形](#2)</font><br>
<font size=4>[3.数据链接操作](#3)</font><br>
<font size=4>[4.行、列操作](#4)</font><br>
<font size=4>[5.数据概要](#5)</font><br>
<font size=4>[6.操作缺失值](#6)</font><br>
<font size=4>[7.创建新的列](#7)</font><br>
<font size=4>[8.分组](#8)</font><br>
<font size=4>[9.透视](#9)</font><br>
<font size=4>[10.合并数据集](#10)</font><br>

---

<h4 id="1">1.创建DataFrame</h4>

  ```python
  import Pandas as pd

  df = pd.DataFrame({'a':[4,5,6],'b':[7,8,9],'c':[10,11,12]},index=[1,2,3])

  df = pd.DataFrame({'a':[4,5,6],'b':[7,8,9],'c':[10,11,12]},index=[1,2,3],columns=['a','b','c'])

  df = pd.DataFrame({'a':[4,5,6],'b':[7,8,9],'c':[10,11,12]},
                    index=pd.MultiIndex.from_tuples([('d',1),('d',2),('e',2)]),
                    names = ['n','v'])))

  ```

<h4 id="2">2.数据整形</h4>

  ```python
  import Pandas as pd
  pd.melt(df)  # 列转行
  pd.concat([df1,df2]) # 合并df1和df2
  pd.concat([df1,df2],axis=1)
  df.pivot(columns='var',values='val') # 透视行转列
  df.sort_values('mpg')
  df.sort_values('mpg',ascending=False)
  df.rename(columns={'y':'year',})
  df.sort_index()
  df.reset_index()
  df.drop(['Length','Height'],axis=1)

  ```

<h4 id="3">3.数据链操作</h4>

  ```python
  import Pandas as pd
  df = (pd.melt(df).rename(columns={'variable':'var','value':'val',}).query('val >= 200'))
  ```

<h4 id="4">4.行、列操作</h4>

  ```python
  import Pandas as pd
  # 行操作
  df[df.Length>7]
  df.drop_duplicates()
  df.head()
  df.tail()
  df.sample(frac=0.5)
  df.sample(n=10)
  df.iloc(10:20)
  df.nlargest(n,'value')
  df.nsmallest(n,'value')
  # 列操作
  df[['width','length','species']]
  df['width']
  df.width
  df.filter(regex='regex')
  df.loc(:,'x2':'x4')
  df.iloc(:,[1,2,5])
  df.loc(df['a']>10,['a','c'])
  # 逻辑表达式
  # 正则表达式
  ```

<h4 id="5">5.数据概要</h4>

  ```python
  import Pandas as pd
  df['w'].value_counts() # 统计行数：每个不相等的变量的值
  len(df) # 行数
  df['w'].nunique() # 去重
  df.describe() # 基本的描述
  sum()
  count() # 计数
  median() # 中位数
  quantile([0.25,0.75]) # 分位数
  apply(function)
  min()
  max()
  mean() # 平均值
  var() # 方差
  std() # 标准差
  ```

<h4 id="6">6.操作缺失值</h4>

  ```python
  import Pandas as pd
  df.dropna() # 删除
  df.fillna(value) # 填充
  ```

<h4 id="7">7.创建新的列</h4>

  ```python
  import Pandas as pd
  df.assign(Area=lambda df:df.Length * df.Height)
  df['Volume'] = df.Length*df.Height*df.Depth
  pd.qcut(df.col,n,labels=False) # 分桶操作
  # 矢量函数
  max(axis=1)
  min(axis=1)
  abs() # 绝对值
  clip(lower=-10,upper=10)
  ```

<h4 id="8">8.分组</h4>

  ```python
  import Pandas as pd
  df.groupby(by='col')
  df.groupby(level='ind')
  # 函数
  size()
  agg(function)
  shift(1)
  shift(-1)
  rank(method='dense')
  rank(method='min')
  rank(method='first')
  rank(pct=True)
  cumsum()
  cummax()
  cummin()
  cumprod()
  ```

<h4 id="9">9.透视</h4>

  ```python
  import Pandas as pd
  df.plot.hist()
  df.plot.scatter(x='w',y='h')
  ```
<h4 id="10">10.合并数据集</h4>

  ```python
  import Pandas as pd
  pd.merge(adf,bdf,how='right',on='x1')
  pd.merge(adf,bdf,how='left',on='x1')
  pd.merge(adf,bdf,how='inner',on='x1')
  pd.merge(adf,bdf,how='outer',on='x1')
  # 过滤合并
  adf[adf.x1.isin(bdf.x1)]
  adf[~adf.x1.isin(bdf.x1)]  # 非
  #
  pd.merge(ydf,zdf)
  pd.merge(ydf,zdf,how='outer')
  pd.merge(ydf,zdf,how='outer',indicator=True)
    .query('_merge == "left_only"')
    .drop(['_merge'],axis=1)
  ```
