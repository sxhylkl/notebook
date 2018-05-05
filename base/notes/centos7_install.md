### Centos7安装
---
<font size=4>[1.虚拟机设置静态IP](#1)</font><br>
<font size=4>[2.关闭防火墙与网关](#2)</font><br>
<font size=4>[3.设置SSH登陆](#3)</font><br>
<font size=4>[4.更新系统内核及时间同步](#4)</font><br>
<font size=4>[5.安装依赖库](#5)</font><br>
<font size=4>[6.安装python环境](#6)</font><br>
<font size=4>[7.安装Java环境](#7)</font><br>
<font size=4>[8.子集构造、切片、索引](#8)</font><br>
<font size=4>[9.数组的操作](#9)</font><br>

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

<h4 id="4">4.帮助</h4>

  - 创建数组

  ```python
  brew install 软件名
  import numpy as np
  ```

<h4 id="5">5.数组的数学方法</h4>

  - 创建数组

  ```python
  brew install 软件名
  import numpy as np
  ```

<h4 id="6">6.数组的复制</h4>

  - 创建数组

  ```python
  brew install 软件名
  import numpy as np
  ```

<h4 id="7">7.数组的排序</h4>

  - 创建数组

  ```python
  brew install 软件名
  import numpy as np
  ```

<h4 id="8">8.子集构造、切片、索引</h4>

  - 创建数组

  ```python
  brew install 软件名
  import numpy as np
  ```

<h4 id="9">9.数组的操作</h4>

- 创建数组

```python
brew install 软件名
import numpy as np
```
