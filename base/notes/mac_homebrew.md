## Homebrew

> MAC软件包管理工具、类似于linux中的yum

---

#### 安装

```shell
# MAC终端执行一下命令
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

```

#### 常用命令
-  搜索命令

  ```shell
  # MAC终端执行一下命令
  brew search 软件名
  ```

- 安装命令
  ```shell
  # MAC终端执行一下命令
  brew install 软件名
  ```

- 卸载命令
  ```shell
  # MAC终端执行一下命令
  brew remove 软件名
  ```

#### 软件安装
- homebrew-cask
  ```shell
  # MAC终端执行一下命令
  brew tap phinze/homebrew-cask
  brew install brew-cask
  ```

-  Java8
  ```shell
  # MAC终端执行一下命令
  brew tap caskroom/versions
  brew cask install java8
  ```

-  Scala
  ```shell
  # MAC终端执行一下命令
  brew install scala
  ```
