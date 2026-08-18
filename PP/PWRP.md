Previous-Layer Residual Write Probes，PWRP

这个方向与 ESP 的逻辑不同，我认为甚至更有意思。

1. 核心出发点

上一层 expert e
′
：

E
l−1,e
′
	​

(x)=W
l−1,e
′
d
	​

h
l−1,e
′
	​

(x).

展开：

E
l−1,e
′
	​

(x)=
j=1
∑
D
	​

h
e
′
,j
	​

(x)d
e
′
,j
	​

,

其中：

d
e
′
,j
	​

=W
l−1,e
′
d
	​

[:,j]∈R
d
.

所以每个 down_proj column：

d
e
′
,j
	​


都是一个真实写入 residual stream 的参数方向。

经典 FFN mechanistic analysis 把 FFN 的第一层方向解释为 pattern/key，而第二层对应方向解释成向 residual/output space 写出的 values；整个 FFN 输出就是这些 value directions 的组合。

因此：

W
l−1,e
′
d
	​

[:,j]
	​


虽然不是真实 hidden token，却比 Gaussian/random vector 多了一层非常明确的模型语义：

它是模型前一层本身学会写入 residual stream 的方向。

所以我们可以问：

这些上一层能够产生的 residual directions，如果继续传播到下一层，会触发下一层 expert 的哪些 channels？

2. 注意：它不等于真实下一层输入

这一点方法中必须说清楚。

真实过程大致是：

x
l
	​

=x
l−1
	​

+MoE
l−1
	​

(⋅)+后续/下一 block attention effects.

所以单个：

d
l−1,e
′
,j
	​


只是 residual update 的一个基向量，不是完整 hidden state。

因此 PWRP 的 claim 应该是：

model-derived residual basis direction
	​


而不是：

approximate real token.
	​

3. 候选池怎么构造

对于第 l 层，收集第 l−1 层所有 routed experts 的 down columns：

D
l−1
	​

={d
l−1,e
′
,j
	​

:e
′
=1,…,N,j=1,…,D}.

候选数：

ND.

如果模型有 shared expert，第一版不要加入 shared expert，只用 routed experts，避免额外结构因素。

而且一律从原始未剪枝 checkpoint提取，不能使用已经剪过的上一层，否则方法会变成 sequential-order dependent。

4. 为什么需要考虑正负方向

真实上一层输出系数：

h
e
′
,j
	​

(x)

可以正也可以负。

因此：

d
e
′
,j
	​


和：

−d
e
′
,j
	​


实际上都可能出现在 residual update 中。

对每一个候选 write vector d
c
	​

，先构造：

q
c
	​

=RMSNorm
l
	​

(d
c
	​

).

注意这里用的是当前第 l 层 MoE 前面的 RMSNorm 参数，因为我们希望把这个 residual direction 映射成当前 expert 接收输入时的尺度。

然后对目标 expert e：

a
c,e
	​

=r
l,e
⊤
	​

q
c
	​

.

使用绝对 affinity：

∣a
c,e
	​

∣.

并把方向朝向 target expert：

p
c,e
WR
	​

=sign(a
c,e
	​

)q
c
	​

	​


于是：

r
l,e
⊤
	​

p
c,e
WR
	​

=∣a
c,e
	​

∣.

这一步的含义是：

上一层这个 residual write direction 的两个可能符号中，选择更符合当前 target expert routing region 的那个方向。

5. 给每个 target expert 选哪些 write probes

对于目标：

(l,e),

从全部：

ND

个 previous-layer write candidates 中，根据：

A
c,e
	​

=
	​

r
l,e
⊤
	​

RMSNorm
l
	​

(d
c
	​

)
	​

	​


选最高的：

K+1
	​


个。

这样与 PP 完全匹配 probe budget：

X
l,e
WR
	​

=Top
K+1
	​

(A
c,e
	​

).

对应的 oriented probes：

p
1
WR
	​

,…,p
K+1
WR
	​

.
6. 为什么这里我建议用 raw router logit，而不是 Router Gram

因为候选已经不是 Router rows 了。

对于 arbitrary residual direction q
c
	​

，最自然的问题就是：

如果它真的作为当前层 hidden input，它会不会倾向进入 expert e？

当前原始 Router 的答案就是：

r
l,e
⊤
	​

q
c
	​

.

因此直接使用：

∣r
l,e
⊤
	​

q
c
	​

∣

比人为构造另一套 Gram 更直接。

Router bias 如果存在，对于固定 target expert 内比较不同 probes只是常数：

r
e
⊤
	​

q
c
	​

+b
e
	​

,

所以 MVP 排序可以忽略 bias。

7. 再把选中的 probes 送入目标 expert

对于 current expert e 的 channel j：

h
k,j
WR
	​

=SiLU((W
l,e
g
	​

[j,:])
⊤
p
k
WR
	​

)((W
l,e
u
	​

[j,:])
⊤
p
k
WR
	​

).

完全复用 PP score：

s
l,e,j
WR
	​

=MeanTopQ
k
	​

∣h
k,j
WR
	​

∣
	​


同样：

不乘 down norm；
不加新 importance；
不 reconstruction。

然后保护：

P
l,e
WR
	​

=Top
GD
	​

(s
l,e,j
WR
	​

).
	​


固定：

G=10%.

其余位置仍由 AIMER 填满。

8. 第一层怎么办

最前面的第一个 MoE layer 没有 previous-MoE write vectors。

MVP 不要设计新机制。

直接：

第一 MoE layer 使用原 PP10
	​


从第二个 MoE layer 开始使用 PWRP10。

因为所有实验都在同一个模型上，只有极少一层特殊处理，对诊断影响有限。

如果 PWRP 后续证明有效，再解决第一层统一性问题。

9. PWRP 的高效实现

不要真的对每个 target expert 单独遍历：

ND

个 vectors。

将上一层全部 down_proj columns 拼起来：

D
prev
=[W
l−1,1
d
	​

,W
l−1,2
d
	​

,…,W
l−1,N
d
	​

]∈R
d×ND
.

逐列使用当前 layer RMSNorm：

P
prev
=RMSNormRows/Cols
l
	​

(D
prev
).

当前 Router：

R
l
	​

∈R
N×d
.

一次 GEMM：

A
l
	​

=R
l
	​

P
prev
∈R
N×ND
.
	​


然后：

∣A
l
	​

∣

每一行直接 Top-(K+1)。

也就是说一次矩阵乘法就能同时得到这一层所有 target experts的 previous-write probe candidates。

如果 ND 太大，不需要一次存下来，可以按 previous expert 或 channel block 做 chunked Top-k。

10. PWRP 最小可行实验

仍然只跑：

方法	B6	B9
AIMER + PP10	已有	已有
AIMER + PWRP10	新跑	新跑

固定：

与 PP 相同 K+1 probe 数；
相同 Q；
G=10%；
NoDownNorm；
原 AIMER；
同位宽；
Quick9；
不补偿、不重构；
previous write candidate 使用原始 checkpoint；
第一 MoE 层 fallback PP10。

另外记录：

O
PP,WR
	​

=
0.1D
∣Top10%(PP)∩Top10%(PWRP)∣
	​

	​


即可。

第一轮不需要做：

PP5 + PWRP5；
限制 previous source expert；
previous AIMER filtering；
down-column magnitude weighting；
多层 write mixing。

只有 PWRP10 自己有正信号，才进入 Hybrid。