### HQL基本语法

---
<font size=4>[1.关系运算](#1)</font><br>
<font size=4>[2.数学运算](#2)</font><br>
<font size=4>[3.逻辑运算](#3)</font><br>
<font size=4>[4.数值计算](#4)</font><br>
<font size=4>[5.日期函数](#5)</font><br>
<font size=4>[6.条件函数](#6)</font><br>
<font size=4>[7.字符串函数](#7)</font><br>
<font size=4>[8.集合统计函数](#8)</font><br>
<font size=4>[9.复合类型构建操作](#9)</font><br>
<font size=4>[10.符合类型访问操作](#9)</font><br>
<font size=4>[11.复杂类型长度统计函数](#9)</font><br>
---

<h4 id="1">1.关系运算</h4>

  ```SQL
  A=B  --
  A<>B --如果表达式中有一个为NULL，则结果为NULL；如果A与B不相等，则为FALSE；否则为True
  A<B --如果表达式中有一个为NULL，则结果为NULL；如果A小于B，则为True；否则为FALSE
  A>B --如果表达式中有一个为NULL，则结果为NULL；如果A大于B，则为FTrue；否则为FALSE
  A<=B
  A>=B
  A IS NULL --如果A为NULL，则为True，否则为FALSE
  A IS NOT NULL --如果A为NULL，则为True，否则为FALSE
  A LIKE B
  A RLIKE B
  A REGEXP B
  ```

<h4 id="2">2.数学运算</h4>

  ```SQL
  A+B --int和double一般结果是Double
  A-B
  A*B
  A/B --结果为Double
  A%B -- 取余函数结果得数值类型等于A和B类型得最小父类型
  A&B -- 位与操作
  A|B -- 位或操作
  A^B -- 按位异或操作
  ~A --按位取反操作
  ```

<h4 id="3">3.逻辑运算</h4>

  ```SQL
  A AND B --逻辑与运算
  A OR B --逻辑或运算
  NOT A --逻辑非运算
  ```

<h4 id="4">4.数值计算</h4>

  ```SQL
  round(double a) --返回BIGINT 取整函数
  round(double a,int d) --返回指定精度为d的double类型
  floor(double a) --返回BIGINT 返回等于或者小于该double变量的最大的整数
  ceil(double a) -- 返回等于或者大于该Double变量的最小的函数
  ceiling(double a) -- 返回等于或者大于该Double变量的最小的函数
  rang(),rand(int seed) -- 返回值为Double  返回0到1范围的随机数。如果指定种子seed,则会等到一个稳定的随机数序列
  exp(double a) -- 返回double 返回自然对数e的a次方
  log10(double a) --返回值为double 返回以10为底额a的对数
  log10(double a) --返回值为double 返回以10为底额a的对数
  log2(double a) --返回以2为底的a的对数
  log(double base,double a) --返回以base为底的a的对数
  pow(double a,double b) -- 返回a的p次幂，与pow功能相同
  sqrt(double a) --返回a的平方根
  bin(BIGINT a) --返回a的二进制代码表示
  hex(BIGINT a) --如果变量是int类型，那么返回a的十六进制表示；如果变量是string类型，则返回该字符串的十六进制表示
  unhex(BIGINT a) --如果变量是int类型，那么返回a的十六进制表示；如果变量是string类型，则返回该字符串的十六进制表示
  conv(BIGINT num,int from_base,int to_base) --将数值num从from_base进制进化到to_base进制
  abs(double a) abs(int a) --返回数值a的绝对值
  pmod(int a,int b) pmod(double a,double b) --返回正的a除以b的余数
  sin()
  asin() --返回a的反正弦值
  cos()
  acos() --返回a的反余弦值
  positive() --返回a
  negative() --返回-a
  ```

<h4 id="5">5.日期函数</h4>

  ```SQL
  from_unixtime(bigint unixtime[,string format]) --转化UNIX时间戳到当时时区的时间格式
  >select from_unixtime(1323308888,'yyyyMMdd') from table;
  unix_timestamp() --获得当时时区的UNIX时间戳
  >select unix_timestamp() from table;
  unix_timestamp(string data,string pattern) #转换pattern格式的日期到unix时间戳。如果转化失败，则返回0
  >select unix_timestamp('2011-01-01 13:00:00','yyyyMMdd HH:mm:ss') from table
  to_date(string timestamp) --返回日期时间段中的日期部分
  >select to_date('2011-01-01 13:00:00') from table
  year(string date)
  >select year('2011-12-11 10:00:02') from table  --2011
  >select year('2011-12-11') from table --2011
  month(string date) --返回日期中的月份
  >select month('2011-12-11 10:00:02') from table  --12
  >select month('2011-12-11') from table --12
  day() --返回日期中的天数
  >select day('2011-12-11 10:00:02') from table  --10
  >select day('2011-12-11') from table --10
  hour(string date) --返回日期中的天数
  >select hour('2011-12-11 10:00:02') from table  --10
  minute()  --日期转分钟函数
  second()  --日期转秒函数
  weekofyear() --返回日期中在当前的周数
  >select weekofyear('2012-12-08 10:00:02') from table  --49
  datediff() --返回介绍日期减去开始日期的天数
  >select datediff('2018-12-12','2018-04-20') from table
  date_add() --返回开始日期startdate增加days天后的日期
  >select date_add('2018-12-08',10) from table --2018-12-18
  date_sub() --返回开始日期startdate减少days天数后的日期
  >select date_add('2018-12-08',10) from table --2018-11-28
  ```

<h4 id="6">6.条件函数</h4>

  ```SQL
  if(boolean testCondition,T valueTrue,T valureFalseOrNull)  --返回T 当条件testCondition为True时，返回valueTrue,否则返回valueFalseOrNull
  >select if(1=2,100,200) from table; --200
  >select if(1=1,100,200) from table; --100
  coalesce(T v1,T v2,...)  --返回T 返回参数中的第一个非空值；如果所有值都为NULL，那么返回NULL
  >select coalesce(null,'100','200') from table;   --100
  case() A when b then c[when d then e]* [else f]End --如果A等于b,那么返回c;如果a等于d,那么返回e,否则返回f
  ```

<h4 id="7">7.字符串函数</h4>

  ```SQL
  length() --返回字符串的长度
  reverse() --对字符串进行反转结果
  concat(string a,string b) --串联a和b
  concat_ws(string SEP,string A,string B...) --返回输入字符串连接后的结果，SEP表示个字符串间的分隔符
  substr() substring() --截取字符串
  upper() ucase() --进行大写转换
  lower() lcase() --进行小写转换
  trim() --除去字符串两边的空格
  ltrim() --去除左边的空格
  rtrim() --去除右边的空格
  regexp_replace(string A,string B,string c)  --正则表达式替换函数
  >
  regexp_extract(string subject,string pattern,int index) --正则表达式解析函数 将字符串subject按照pattern正则表达式的规则拆分，返回index指定的字符串
  >select regexp_extract('foothebar','foo(.*?)(bar)',1) from table; --the
  >select regexp_extract('foothebar','foo(.*?)(bar)',2) from table; --bar
  >select regexp_extract('foothebar','foo(.*?)(bar)',0) from table; --foothebar
  parse_url
  ```

<h4 id="8">8.集合统计函数</h4>

  ```SQL
  brew install 软件名
  import numpy as np
  ```

<h4 id="9">9.复合类型构建操作</h4>

```SQL
brew install 软件名
import numpy as np
```

<h4 id="10">10.符合类型访问操作</h4>

```SQL
brew install 软件名
import numpy as np
```

<h4 id="11">11.复杂类型长度统计函数</h4>

```SQL
brew install 软件名
import numpy as np
```
