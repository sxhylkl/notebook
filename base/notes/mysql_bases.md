## MySQL基础

----

#### MySQL的基础
  + 表结构复制+表数据复制

  ``` SQL
  /* 表结构复制 */
  create table a like b;
  /* 表数据复制 */
  insert into b select * from a;
  ```
  + 索引的创建与删除

  ``` SQL
  /* 1-普通索引 */
  alter table table_name ADD INDEX index_name(column_list);
  /* 2-UNIQUE索引 */
  alter table table_name ADD UNIQUE(column_list);
  /* 3-PRIMARY KEY索引 */
  alter table table_name ADD PRIMARY KEY (column_list);
  /* 4-创建索引 */
  create index index_name on table_name(column_list);
  /* 5-创建索引 */
  create unique index index_name on table_name(column_list);
  /* 6-删除索引 */
  drop index index_name on table_name;
  /* 7-修改索引 */
  alter table table_name DROP INDEX index_name;
  /* 8-修改索引 */
  alter table table_name DROP PRIMARY KEY;
  /* 9-查看索引 */
  show index from table_name;
  /* 删除自增的主键索引，首先要删除自增，然后才能删除主键索引 */
  alter table table_name modify column_name int unsigned not null;
  alter table table_name drop PRIMARY KEY;
  /* 增加主键索引，实现自增 */
  alter table table_name add PRIMARY KEY (id);
  alter table table_name modify column_name int unsigned not null auto_increment;
  ```

  + 视图的创建、查看、删除  视图可以作为频繁查询的中间表使用

  ``` SQL
  /* 视图的创建 */
  create view view_name as select * from table_name where cloumn_name > 4;
  /* 视图查看 */
  show tables;
  /* 视图删除 */
  drop view view_name;
  ```

  + 内置函数

  ``` SQL
  /* 字符串函数 */
  concat(string1,string2); //连接字符串
  lcase(string1);  // 转换成小写
  ucase(string2);  // 转换成大写
  length(string1); // 计算字符串长度
  ltrim(string1); // 去除前端空格
  rtrim(string1); // 去除后端空格
  repeat(string1,count); 重复count次
  replace(string1,search_str,replace_str); //在string1中用replace_str替换search_str
  substring(string1,position[,length]); // 从string1的position位置开始，去length个字符
  space(count); //生成count个空格
   /* 数学函数 */
  bin(decimal_number) //十进制转二进制
  ceiling(number1) //向上取整
  floor(number1) //向下取整
  max(num1.num2) // 取最大值
  min(num1,num2) // 取最小值
  sqrt(number1) //开平方
  rand() //返回0-1内的随机值
  /*  日期函数 */
  curdate() //返回当前日期
  curtime() //返回当前时间
  now() // 返回当前的日期时间
  unix_rimestamp() // 返回当前date的unix日间戳
  from_unixtime() //返回Unix时间戳的日期值
  week(date) // 返回日期date为一年中的第几周
  year(date) //返回日期date的年份
  datediff(expr,expr2) //返回起始时间expr和结束时间expr2间天数
  ```

  + 预处理语句
  >所谓的预处理技术，最初也是由MySQL提出的一种减轻服务器压力的一种技术！

  >传统mysql处理流程

  > 1，  在客户端准备sql语句

  > 2，  发送sql语句到MySQL服务器

  > 3，  在MySQL服务器执行该sql语句

  > 4，  服务器将执行结果返回给客户端

  > 这样每条sql语句请求一次，mysql服务器就要接收并处理一次，当一个脚本文件对同一条语句反复执行多次的时候，mysql服务器压力会变大，所以出现mysql预处理，减轻服务器压力！

  > 预处理的基本策略：

  > 将sql语句强制一分为二：

  > 第一部分为前面相同的命令和结构部分

  > 第二部分为后面可变的数据部分

  > 在执行sql语句的时候，首先将前面相同的命令和结构部分发送给MySQL服务器，让MySQL服务器事先进行一次预处理（此时并没有真正的执行sql语句），而为了保证sql语句的结构完整性，在第一次发送sql语句的时候将其中可变的数据部分都用一个数据占位符来表示！比如问号？就是常见的数据占位符！

  ``` SQL
  /* 设置yuchuli预处理,传递一个数据作为一个where判断条件*/
  prepare yuchuli from 'select 8 from t1 where id>?';
  /* 设置一个变量 */
  set@i=1;
  /* 执行预处理 */
  execute yuchuli using @i;
  /* 删除预处理 */
  drop prepare yuchuli;
  ```

  + 事务处理
  > innodb引擎的表才具有事务处理机制

  ``` SQL
  /* 关闭自动提交功能 */
  set autocommit=0;
  /* 从表中删除一条记录 */
  delete from table_name where id=11;
  /* 此时做一个p1还原点 */
  savepoint p1;
  /* 再从表中删除一条数据 */
  delete from table_name where id=10;
  /* 再做一个p2还原点 */
  savepoint p2;
  /* 此时恢复p1还原点，当然后面的p2这些还原点自动会失效*/
  rollback to p1;
  /* 退回到最原始的还原点*/
  rollback;

  /* 修改表引擎 */
  alter table table_name engine=innodb;
  /* 事务回滚一定发生在commit之前，否则是没有意义的 */
  commit;
  ```

  + 存储

  ``` SQL
  /* 表结构复制 */
  create table a like b;
  /* 表数据复制 */
  insert into b select * from a;
  ```

  + 触发器

  ``` SQL
  /* 表结构复制 */
  create table a like b;
  /* 表数据复制 */
  insert into b select * from a;
  ```

  + 重排auto_increment值

  ``` SQL
  /* 表结构复制 */
  create table a like b;
  /* 表数据复制 */
  insert into b select * from a;
  ```

----
#### MySQL性能优化思路
  - SQL语言优化
  - myISAM表锁优化
  - 数据库优化
  - 服务器优化
