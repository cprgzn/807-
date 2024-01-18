# 模式识别（Chapter 2）

## 一.医疗诊断问题

现有一病人的x光片，我们想判断其是否患有癌症。在此情况下，输入变量$\mathbf x$代表x光片,输出变量$t$代表病人是否患有癌症。（患有癌症记为$w_1$类，健康记为$w_2$类）

依据贝叶斯公式，当已知输入$\mathbf x$将其归类为$w_k$的概率应为
$$
P(w_k|\mathbf x)=\frac {P(\mathbf x|w_k)P(w_k)}{P(\mathbf x)}\tag{1-1}
$$
我们将$P(w_k|\mathbf x)$称为后验概率，$P(\mathbf x|w_k)$称为类条件概率,$P(w_k)$称为先验概率。$P(\mathbf x)$是一个常数（对所有的$w_k$都一样）



得出后验概率$P(w_k|\mathbf x)$之后再考虑分类问题，一个很自然的想法是将$\mathbf x$归类至后验概率最大的类别中。
$$
\begin{aligned}
t&=\mathrm{argmax_{w_k}\ \ P(\mathbf x|w_k)P(w_k)}\\
&=\mathrm{argmax_{w_k}\ \ P(\mathbf x,w_k)}
\end{aligned}
\tag{1-2}
$$


显然，有必要证明这种直觉是合理的。

## 二.贝叶斯错误率(二分类)

定义以下符号

1.平均错误率$Pr(error)$,为误分类的概率，目标是尽可能降低错误率。

2.决策区域$\mathcal R_1$:在此范围内$\mathbf x$被判为$w_1$类

3.决策区域$\mathcal R_2$:在此范围内$\mathbf x$被判为$w_2$类

4.全空间$\mathcal R$:所有决策区域之和（不考虑拒绝情况）

5.决策边界：决策区域相交的边界

有以上定义可以给出$Pr(error)$的表达式如下(二分类)
$$
\begin{aligned}
Pr(error)&=Pr(w_1,\mathbf x\in \mathcal R_2)+Pr(w_2,\mathbf x\in \mathcal R_1)\ \ \ \ \mathbf x属于w_1类却被判为w_2类以及\mathbf x属于w_2类却被判为w_1类\\
&=\int_{\mathcal R_2}P(\mathbf x,w_1)\mathrm{d\mathbf x}+\int_{\mathcal R_1}P(\mathbf x,w_2)\mathrm{d\mathbf x}\\
\end{aligned}
\tag{1-3}
$$
上式可以看出决定$Pr(error)$的正是决策区域$\mathcal {R_1，R_2}$,可以通过调整决策边界使$Pr(error)$达到最小。（决策边界可以是任意的，只要不考虑错误率）

易证得
$$
Pr(error)\geq\int_{\mathcal R} \mathrm{min(P(\mathbf x,w_1),P(\mathbf x,w_2))}\mathrm{d\mathbf x}\\
\Updownarrow\\
P(\mathbf x,w_1)\leq P(\mathbf x,w_2) \ \ if \mathbf\ \  x\in \mathcal R_2\\
P(\mathbf x,w_1)\geq P(\mathbf x,w_2) \ \ if \mathbf\ \  x\in \mathcal R_1\\
\tag{1-4}
$$
与（1-2）所得出的判别结果是一致的

![](E:\360MoveData\Users\Administrator\Desktop\Markdown\模式识别\未命名文件-导出.png)

阴影部分的面积就是$P(error)$

## 三.贝叶斯错误率（多分类）

在多分类的情况下计算分类正确率更为容易。

记$P(correct)$为分类正确的概率，总共有$c$个类别，类似的我们有
$$
\begin{aligned}
Pr(correct)&=Pr(w_1,\mathbf x\in \mathcal R_1)+Pr(w_2,\mathbf x\in \mathcal R_2)+\cdots\\
&=\sum_i^cPr(w_i,\mathbf x\in \mathcal R_i)\\
&=\sum_i^c\int_{\mathcal R_i}P(\mathbf x,w_i)\mathrm{d\mathbf x}\\
\end{aligned}
$$
易证得
$$
\begin{aligned}
Pr(correct)\leq \int_{\mathcal R}&\mathrm{max(P(\mathbf x,w_1),P(\mathbf x,w_2),\cdots)}\mathrm{d\mathbf x}\\
&\Updownarrow\\
if\ \ \ \ \  P(\mathbf x,w_j)\geq P(\mathbf x,w_i),&\forall \ \ i成立，\mathbf x\in \mathcal R_j
\end{aligned}
\tag{1-5}
$$
***所以最佳决策区域的选择使每个$\mathbf x$都归类至使其后验概率最大的类别中***

## 四.判别函数(二分类)

本质上都是比较$\mathbf x$的后验概率大小。

1.
$$
P(\mathbf x|w_1)P(w_1)\geq P(\mathbf x|w_2)P(w_2)\Rightarrow \mathbf x \in w_1\\
P(\mathbf x|w_1)P(w_1)\leq P(\mathbf x|w_2)P(w_2)\Rightarrow \mathbf x \in w_2\\
\tag{1-6}
$$
2.
$$
l_{12}=\frac{P(\mathbf x|w_1)}{P(\mathbf x|w_2)}\geq \frac{P(w_2)}{P(w_1)}=\theta_{12}\Rightarrow \mathbf x\in w_1\\
l_{12}=\frac{P(\mathbf x|w_1)}{P(\mathbf x|w_2)}\leq \frac{P(w_2)}{P(w_1)}=\theta_{12}\Rightarrow \mathbf x\in w_2\\
\tag{1-7}
$$
$l_{12},\theta_{12}$称为似然比与判别阈值。

## 五.最小风险损失

在许多应用中，我们的目标比最小化错误率更复杂，如果给健康的病人误诊为患病，结果可能仅仅是一些心理压力，想反，将癌症病人误诊为健康可能会导致病人错过最佳治疗时间而死亡。对于这两种情况造成的损失不应一视同仁。

我们可以通过损失函数来描绘这个问题。

对于$\mathbf x$的真实类别$w_k$,如果我们把它分类成$w_j$，造成一定损失$L_{kj}$,当$k=j$，即分类正确可认为$L_{kj}=0$

将$w_k$类误分类成$w_j$类的概率为
$$
\int_{\mathcal R_j}P(\mathbf x,w_k)\mathrm{dx}
$$
所造成的损失为
$$
\int_{\mathcal R_j}L_{kj}P(\mathbf x,w_k)\mathrm{dx}
$$
将所有的可能性相加得到总期望损失
$$
\begin{aligned}
\Bbb E(\mathcal R_j)&=\sum_k\sum_j\int_{\mathcal R_j}L_{kj}P(\mathbf x,w_k)\mathrm{dx}\\
&=\sum_j\int_{\mathcal R_j}\sum_kL_{kj}P(\mathbf x,w_k)\mathrm{dx}\\
&\geq \int_{\mathcal R}\mathrm{min}(\sum_kL_{k1}P(\mathbf x,w_k),\sum_kL_{k2}P(\mathbf x,w_k),\cdots)\mathrm{dx}
\end{aligned}
\tag{1-8}
$$
类比式（1-5）很容易得出判别式
$$
\sum_kL_{ki}P(\mathbf x,w_k)\leq \sum_kL_{kj}P(\mathbf x,w_k),\forall j,成立,\Rightarrow \mathbf x \in \mathcal R_i
$$
记$R_i=\sum_kL_{ki}P(\mathbf x,w_k)$,易知其意义就是所有类判别为$w_i$类的总损失

