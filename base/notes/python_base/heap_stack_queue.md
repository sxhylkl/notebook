## 堆、栈、队列

-----
### 堆

  #### 1.生成新的账号的RSA

  > #新建SSH key：

  >$ cd ~/.ssh     # 切换到~/.ssh
ssh-keygen -t rsa -C "xxxxx@email.com"  # 新建工作的SSH key

  >#设置名称为xxx_github_rsa

  >Enter file in which to save the key (/.ssh/id_rsa): xxx_github_rsa

  #### 2.新密钥添加到SSH agent中

  > ssh-add ~/.ssh/xxx_github_rsa

  如果出现Could not open a connection to your authentication agent的错误，就试着用以下命令：
  > ssh-agent bash

  > ssh-add ~/.ssh/xxx_github_rsa

  #### 3.修改config文件

  > #新建SSH key：

  > Host github2
  > HostName github.com
  > User git
  > IdentityFile ~/.ssh/xxx_github_rsa

  #### 4.添加到你的xxx_github_rsa.pub到github上的SSH Key

  > Settings

  > SSH and GPG keys------SSH keys

-----
### 栈


-----
### 队列
