### Centos7安装
---
<font size=4>[1.虚拟机设置静态IP](#1)</font><br>
<font size=4>[2.关闭防火墙与网关](#2)</font><br>
<font size=4>[3.设置SSH登陆](#3)</font><br>
<font size=4>[4.更新系统内核及时间同步](#4)</font><br>
<font size=4>[5.安装依赖库](#5)</font><br>
<font size=4>[6.安装python环境](#6)</font><br>
<font size=4>[7.安装Java环境](#7)</font><br>
<font size=4>[8.Python依赖包安装](#8)</font><br>
---

<h4 id="1">1.虚拟机设置静态IP</h4>

  ```
  静态IP设置注意：虚拟机中的网关地址和子网掩码必须与主机的一致
  /etc/sysconfig/network-scripts/ifcfg-ens33网卡中内容
    TYPE=Ethernet
    BOOTPROTO=static
    NAME=ens33
    UUID=ef34f301-0bba-4501-90a2-bd53882487ed
    DEVICE=ens33
    ONBOOT=yes
    IPADDR=192.168.6.251
    GATEWAY=192.168.6.1
    NETMASK=255.255.254.0
    DNS1=8.8.8.8
    DNS2=114.114.114.114
    #NM_CONTROLLED=yes
  ```

<h4 id="2">2.关闭防火墙</h4>

  ```
  关闭防火墙
  systemctl stop firewalld
  systemctl disable firewalld
  systemctl status firewalld

  编辑/etc/selinux/config
  设置SELINUX=disabled
  ```

<h4 id="3">3.设置SSH登陆</h4>

  ```
  安装SSH
  yum install -y openssh-server
  设置SSH远程登陆
  修改/etc/ssh/sshd_config文件
  设置
  Port 22
  ListenAddress 0.0.0.0
  ListenAddress ::
  PermitRootLogin yes
  PasswordAuthentication yes
  ```

<h4 id="4">4.更新系统内核及时间同步</h4>

  ```
  date +%Z
  cp -f /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
  ntpdate cn.pool.ntp.org
  ```

<h4 id="5">5.安装依赖库</h4>

  ```
  yum -y install sudo vim
  yum -y install zlib-devel bzip2-devel ncurses-devel openssl-devel
  yum -y install make gcc-c++ cmake bison-devel swig patch sqlite-devel
  yum -y install readline readline-devel rsync nload ntp ntpdate wget
  yum -y install net-tools telnet firewalld
  yum -y install libjpeg-devel python-devel python-setuptools
  easy_install pip
  ```

<h4 id="6">6.安装python环境</h4>

  ```
  下载python3.6.4
  wget https://www.python.org/ftp/python/3.6.4/Python-3.6.4.tgz
  tar -zxf Python-3.6.4.tgz
  mkdir /usr/local/python3
  cd Python3.6.4
  ./configure --prefix=/usr/local/python3 --enable-optimizations
  make
  make install
  ln -s /usr/local/python3/bin/python3.6 /usr/bin/python3
  ln -s /usr/local/python3/bin/pip /usr/bin/pip3
  cd /usr/bin
  mv python python.bak
  mv python3 python
  编辑/etc/profile
  export PATH="$PATH:/usr/local/python3/bin"
  修改yum依赖的python
  vi /usr/bin/yum
  ```

<h4 id="7">7.安装Java环境</h4>

  ```
  下载
  wget http://download.oracle.com/otn-pub/java/jdk/8u171-b11/512cd62ec5174c3487ac17c61aaa89e8/jdk-8u171-linux-x64.tar.gz
  tar -zxvf jdk-8u171-linux-x64.tar.gz
  修改/etc/profile
  添加
  export JAVA_HOME=/home/soft/jdk1.8.0_171
  export CLASSPATH=.:$JAVA_HOME/jre/lib/rt.jar:$JAVA_HOME/lib/dt.jar:$JAVA_HOME/lib/tools.jar
  export PATH=$PATH:$JAVA_HOME/bin
  使配置文件生效
  source /etc/profile
  验证是否安装成功
  java -version
  ```

  <h4 id="8">8.Python依赖包安装</h4>

    ```
    将pip安装列表输出到文件中
    pip freeze >requirements.txt
    创建pip安装列表文件
    vi requirements.txt
    安装
    pip install -r requirements.txt
    安装列表
      autopep8
      beautifulsoup4
      celery
      Django
      Flask
      future
      gevent
      ggplot
      hive
      impyla
      ipython
      jieba
      Jinja2
      jupyter
      Keras
      luigi
      lxml
      Mako
      Markdown
      matplotlib
      multiprocess
      nltk
      nolearn
      notebook
      numba
      numpy
      pandas
      PyHive
      pymongo
      PyMySQL
      pyOpenSSL
      python-daemon
      pyzmq
      queuelib
      redis
      request
      sasl
      scikit-learn
      sciluigi
      scipy
      Scrapy
      scrapy-redis
      selenium
      setuptools
      simplegeneric
      simplejson
      six
      SQLAlchemy
      thrift
      thrift-sasl
      thriftpy
      tornado
      uuid
      Werkzeug
    ```
