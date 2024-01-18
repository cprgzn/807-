

# 统计学习理论——chapter 1：（风险）损失函数的泛函表示与本质

> 20年代，Hadamard观察到在一些（很一般的）情况下，求解（线性）算子方程
> $$
> \mathbf{A}f=F, \ \ \ \ f \in \mathscr{F}
> $$
> 的问题（寻找满足这一等式的函数$f∈\mathscr{F}$）是不适定的：即使方程存在唯一解，如果方程右边有一个微小的变动（如用$‖F−F_\delta‖<\delta$任意小的$F_\delta取代F$），也会导致解有很大的变化（即可能导致$‖f−f_\delta‖$很大）。
>
> 在这种情况下，如果方程右边的$F$是不准确的（如引入了$\delta$水平的噪声），那么使泛函
> $$
> R(f)=||\mathbf{A}f-F_\delta||^2
> $$
> 最小化的函数$f_\delta$并不能保证在$\delta\rightarrow0$时是方程真实解的一个好的近似。
>
> 60年代，Tikhonov发现，如果不是最小化泛函$R(f)$，而是最小化另一个称为正则化泛函的函数
> $$
> R^*(f)=||\mathbf{A}f-F_\delta||^2+\gamma(\delta)\Omega(\delta)
> $$
> 其中$\Omega(f)$是某个泛函数，$\gamma(\delta)$是某个适当的常数（它依赖于噪声的水平），那么我们就可以得到一个解序列，它在$\delta\rightarrow0$时收敛于我们希望的解。
>
> 在一定意义上，神经网络中的过拟合问题其实是不适定问题理论中称之为“错误结构”的现象，从解决不适定问题的理论中，人们得到了防止过学习的工具——正则化技术。

##  GSLM模型

1. 产生器$G$，产生随机向量$x\in \Bbb R^n$，它们是从固定但未知的概率分布$F(x)$中独立同分布$(i.i.d.)$产生的。
2. 训练器$S$，根据固定但是未知的条件分布$F(y|\mathbf x)$，对每个输入向量$\mathbf x$返回一个输出值$y$，我们只考虑值$y$是标量的情形。
3. 学习机器$LM$，它能够实现一定的函数集$f(\mathbf x,\alpha),\alpha∈\Lambda$，其中$\Lambda$是参数集合，我们只考虑值$f(\mathbf x,\alpha)$是标量的情形。

![image-20230903150031765](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20230903150031765.png)

学习的问题就是从给定的函数集$f(\mathbf x,\alpha),\alpha∈\Lambda$中选择能够最好地逼近训练器响应的函数。这种选择是基于训练集的，训练集由根据联合分布$F(x,\mathbf y)=F(x)F(y|\mathbf x)$抽取出的l个独立同分布样本$(\mathbf x_i,y_i),i=1,2,⋯,l$组成。

## 风险最小化问题

考虑在给定输入$\mathbf x$的情况下，学习机器的输出$f(\mathbf x,\alpha)$与真实响应$y$之间的差异（损失）$L(y,f(\mathbf x,\alpha))$。考虑损失的期望值
$$
R(\alpha)=\int L(y,f(\mathbf x,\alpha))\mathrm{d}F(\mathbf x,y)\tag{1-1}
$$
它就是风险泛函，学习的目标就是最小化风险泛函（即泛化误差）

## 三种主要的学习问题

1.模式识别

   $y$的输出只有两种取值$y = {0\ \mathrm{or}\ 1}$,学习机器输出也仅为0或1。考虑下面的损失函数：
$$
L(y,f(\mathbf x,\alpha))=
\begin{cases}
0,\ \ y=f(\mathbf x,\alpha)\\[2ex]
1,\ \ y\neq f(\mathbf x,\alpha)
\end{cases}\tag{1-2}
$$
​	在此条件下,最小化风险泛函$R(\alpha)$就是最小化错误率。

2.回归问题。

​	给定损失函数
$$
L(y,f(\mathbf x,\alpha))=(y-f(\mathbf x,\alpha))^2\tag{1-3}
$$
​	将其带入式（1-1）中，可得
$$
\begin{aligned}
R(\alpha)&=\int L(y,f(\mathbf x,\alpha))\mathrm{d}F(\mathbf x,y)\\
&=\int (y-f(\mathbf x,\alpha))^2\mathrm{d}F(\mathbf x,y)\\
&=\iint (y-f(\mathbf x,\alpha))^2\boldsymbol f(\mathbf x,y)\mathrm{d}\mathbf x\mathrm dy   \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \boldsymbol f(\mathbf x,y)是联合概率密度
\end{aligned}
$$
​	想要使得$R(\alpha)$最小，利用欧拉—拉格朗日方程计算$R(\alpha)$关于$f(\mathbf x,\alpha)$的导数，最后可以解得
$$
\begin{aligned}
f(\mathbf x,\alpha)&=\int y\boldsymbol f(y|\mathbf x)\mathrm dy\\
&=\int y \mathrm dF(y|\mathbf x)\\
&=\Bbb E_y[y|\mathbf x]
\end{aligned}
$$
​	这也透露回归问题的本质是逼近$y$的条件期望。

3.密度估计（Fisher-Wald表示）

​	最后，考虑从密度集函数$p(\mathbf x,\alpha),\alpha \in \Lambda$中估计密度函数的问题。这是一个无监督问题，只有$\mathbf x$,对这个问题，考虑下面的损失函数：
$$
L(p(\mathbf x,\alpha))=-\log p(\mathbf x,\alpha)\tag{1-4}
$$
将其带入式（1-1）中得：
$$
\begin{aligned}
R(\alpha)&=\int -\log p(\mathbf x,\alpha)\mathrm dF(\mathbf x)\\
&=\int -\log p(\mathbf x,\alpha)\boldsymbol f(\mathbf x)\mathrm d\mathbf x\\
&=\int \boldsymbol f(\mathbf x)\frac{\log \boldsymbol f(\mathbf x)}{\log p(\mathbf x,\alpha)}\mathrm d\mathbf x-\int f(\mathbf x)\log f(\mathbf x)\mathrm d\mathbf x
\end{aligned}
$$
上式中第二项与$p(\mathbf x,\alpha)$无关，第一项代表真实概率密度$\boldsymbol f(\mathbf x)$与估计概率密度$p(\mathbf x,\alpha)$之间的Kullback—Leibler距离，概率密度估计的本质是极小化真实概率密度$\boldsymbol f(\mathbf x)$与估计概率密度$p(\mathbf x,\alpha)$之间的Kullback—Leibler距离。

学习问题可以一般表示为：设有定义在空间$\mathbf z$上的概率测度$F(\mathbf z)$。考虑函数集合$Q(\mathbf z,\alpha),\alpha \in \Lambda$。学习的目标是最小化风险泛函
$$
R(\alpha)=\int Q(\mathbf z,\alpha)\mathrm{d}F(\mathbf z),\ \ \alpha \in \Lambda \tag{1-5}
$$
其中概率测度$F(\mathbf z)$未知，但给定了一定的独立同分布样本
$$
\mathbf z_1,\mathbf z_2,\mathbf z_3,\cdots,\mathbf z_l.\tag{1-6}
$$


前面讨论的都是这一一般问题的特例，接下来我们将探讨在学习问题中一般表式的结论

## 经验风险最小化归纳原则（ERM原则）

为在分布函数未知的情况下最小化（1-5）式的风险泛函，采用下面的归纳原则：

1.  把风险泛函$R(\alpha)$替换为所谓的经验风险泛函
   $$
   R_{emp}=\frac{1}{l}\sum_{i=1}^lQ(\mathbf z_i,\alpha)\tag{1-7}
   $$

2. 使用使（1-7）式最小的函数$Q(\mathbf z,\alpha_j)$去逼近使（1-5）式最小的函数$Q(\mathbf z,\alpha_k)$。这一原则被称为**经验风险最小化**原则，简称emp准则。

对于式（1-3）我们要进行的最小化泛函变成
$$
R_{emp}=\frac{1}{l}\sum_{i=1}^l(y_i-f(\mathbf x_i,\alpha))^2
$$
这就是最小二乘法。而如果对于式（1-4）
$$
R_{emp}=\frac{1}{l}\sum_{i=1}^l\ln p(\mathbf x_i,\alpha)
$$
最小话这一泛函对应最大似然方法。

## 统计学习理论的四个问题

1.一个基于EMP原则的学习过程具有一致性的条件是什么？

2.这个学习过程收敛速度有多快？

3.如何控制这个学习过程的收敛速度（推广能力）？

4.怎样构造能控制推广能力的算法？