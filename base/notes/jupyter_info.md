### Jupyter Notebook安装与配置
---
<font size=4>[1.Jupyter安装与启动](#1)</font><br>
<font size=4>[2.配置](#2)</font><br>
<font size=4>[3.Win上bat快捷键启动](#3)</font><br>


---

<h4 id="1">1.Jupyter的安装与启动</h4>

  ```python
  # Python2
  pip2 install jupyter --user
  # Python3
  pip2 install jupyter --user
  #将Python2、Python3的环境添加到Jupyter的配置里
  python2 -m ipykernel install --user
  python3 -m ipykernel install --user
  # 启动
  ipython2 notebook
  ipython3 notebook
  Jupyter notebook #首选
  ```

<h4 id="2">2.配置</h4>

  ```python
  # 配置工作空间
  #1生成配置文件
  jupyter notebook --generate-config
  #2修改配置文件目录为 D:\WorkSpace
  c.NotebookApp.notebook_dir = u'D:\WorkSpace'
  ```

<h4 id="3">3.Win上bat快捷键启动</h4>

  ```
  创建一个后缀名为.bat的文件，内容如下：
  @echo off
  C:
  cd C:\Users\Andy
  jupyter notebook
  ```
