第六章P129反向传播部分，$\delta$并不是损失函数对权值的导数，而是损失函数对未激活输出$x$的导数，书上所用的损失函数是$MSE$损失（不是交叉熵损失），最后一层的激活函数是$Sigmod$函数

第六章p155，$csvm$中约束条件应该去掉$\beta\geq0$这一条件（这个是等式约束）

第九章，P197页，$J_d$计算的不仅仅包括类间的距离，还包括类内的距离（甚至自己和自己的距离）

第十章P212，拉格朗日函数与约束不符，约束为$\mathrm {tr}(\mathbf w^TS_w\mathbf w)=1$

而拉格朗日函数却为
$$
\begin{aligned}
g(\mathbf w)&=J_1(\mathbf w)-\mathrm{tr}(\Lambda(\mathbf w^TS_w\mathbf w-I))\\
&\Updownarrow\ (拉格朗日函数实际的约束)\\
&\mathbf w^TS_w\mathbf w=I
\end{aligned}
$$
第十章P218页，各维新特征的方差并不是$\lambda_i$，$\lambda _i$实际上是各维新特征在先验概率下的加权方差。

