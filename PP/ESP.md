Expert Spectral Probes，ESP
1. 核心思想

当前 PP 的 probe 来源是 Router：

r
l,e
	​

→RMSNorm→p
l,e
	​

.

它表达的是：

Router 学到的“什么输入方向属于这个 expert”
	​


ESP 换一个完全不同的内部信息源：直接从 expert 自己的 W
g
	​

,W
up
	​

 中寻找它最强的输入敏感方向。

对 expert e，定义联合输入算子：

M
l,e
	​

=[
W
l,e
g
	​

W
l,e
u
	​

	​

]∈R
2D×d
.

对其做 SVD：

M
l,e
	​

=UΣV
⊤
.

右奇异向量

v
1
	​

,v
2
	​

,…∈R
d

就是 residual/input space 中的方向，并且

∥M
l,e
	​

v
i
	​

∥
2
	​

=σ
i
	​


说明 v
i
	​

 是能够让该 expert 的 gate/up 两个输入投影整体产生较大响应的方向。

等价地，不需要显式拼接：

C
l,e
	​

=(W
l,e
g
	​

)
⊤
W
l,e
g
	​

+(W
l,e
u
	​

)
⊤
W
l,e
u
	​

	​


然后求：

C
l,e
	​

v
i
	​

=λ
i
	​

v
i
	​

,λ
1
	​

≥λ
2
	​

≥⋯.

其中：

λ
i
	​

=σ
i
2
	​

.

因此 ESP 的逻辑是：

expert weights→dominant input directions→pseudo inputs→真实 SwiGLU response
	​


注意：奇异值不直接作为 channel importance。我们只是利用 eigenvectors 产生 probes。

2. 为什么只用 W
g
	​

,W
up
	​

，暂时不用 W
down
	​


这是我建议 MVP 故意这样设计的。

W
g
	​

,W
up
	​

 都直接读取 MoE 输入：

x∈R
d
,

所以它们的右奇异向量天然属于输入空间。

而 W
down
	​

 的列描述的是向 residual stream 写出的方向：

d
j
	​

=W
d
[:,j].

它属于 output/write geometry。

第一轮如果把：

W
g
	​

,W
up
	​

,W
down
	​


都混进去，我们无法判断 spectral probe 有效究竟来自哪里。

所以 ESP-v1：

只做 input-side spectral probes
	​


最干净。

3. SVD 的符号问题必须处理

特征向量有：

v
i
	​


和：

−v
i
	​


完全等价的符号不确定性。

但 SwiGLU：

SiLU(g
⊤
x)(u
⊤
x)

对：

x→−x

不是简单对称的，因此不能随机接受 SVD 返回的符号。

MVP 推荐：用当前 expert Router 只做 sign orientation

先分别构造：

q
i
+
	​

=RMSNorm
l
	​

(v
i
	​

),
q
i
−
	​

=RMSNorm
l
	​

(−v
i
	​

).

计算目标 expert 的 router logit：

a
i
+
	​

=r
l,e
⊤
	​

q
i
+
	​

,a
i
−
	​

=r
l,e
⊤
	​

q
i
−
	​

.

选择：

p
i
ESP
	​

=arg
q∈{q
i
+
	​

,q
i
−
	​

}
max
	​

r
l,e
⊤
	​

q
	​


实际上在无 bias RMSNorm 下基本就是给 v
i
	​

 选择一个朝向 target-router 的符号。

这样 Router：

不决定选择哪个 eigenvector；
不改变 spectral ranking；
只消除 eigenvector 的人为符号不确定性。

这一点很重要，也与 SSMoE 区分开。

4. ESP probe 数量怎么设置

不要第一轮引入新的超参数 sweep。

当前 PP 有：

K+1

个 probes。

ESP 直接取 top：

C=K+1
	​


个 eigenvectors：

v
1
	​

,…,v
K+1
	​

.

所以每个 expert 仍然拥有：

K+1

个 probes。

这样 PP 和 ESP 的 probe 数完全一致。

5. ESP 怎么产生 channel score

每个 spectral probe：

p
i
ESP
	​


直接送入目标 expert e。

对第 j 个 channel：

h
i,j
ESP
	​

=SiLU((W
g
[j,:])
⊤
p
i
ESP
	​

)((W
u
[j,:])
⊤
p
i
ESP
	​

).

完全复用已经有效的 PP aggregation：

s
j
ESP
	​

=MeanTopQ
i
	​

∣h
i,j
ESP
	​

∣
	​


这里：

不乘 W
down
	​

 列范数；
Q 与当前 PP 相同；
不加入 eigenvalue λ
i
	​

 weighting；
不加入 AIMER score；
不做 reconstruction。
6. 最终怎么和 AIMER 配合

和 PP baseline 完全相同。

ESP 保护：

P
l,e
ESP
	​

=Top
GD
	​

(s
j
ESP
	​

).

第一轮固定：

G=10%
	​


然后从非保护 channel 中按照现有 AIMER 排序补齐到固定宽度：

∣S
l,e
	​

∣=BD.

所以实验就是：

AIMER + ESP10
	​


不是：

PP+ESP

第一轮先测试 ESP 单独作为 protection signal 是否有价值。

7. ESP 最小可行实验

只跑两组新数据：

方法	B6	B9
AIMER + PP10	已有 baseline	已有 baseline
AIMER + ESP10	新跑	新跑

全部固定：

K,Q,G=10%,B6/B9,

Quick9、AIMER 公式、同位宽设置全部不变。

同时输出两个诊断

第一个：

O
PP,ESP
	​

=
0.1D
∣Top10%(PP)∩Top10%(ESP)∣
	​

	​


第二个，谱集中度：

R
e
spec
	​

=
∑
i
	​

λ
i
	​

∑
i=1
K+1
	​

λ
i
	​

	​

	​


用于判断 expert spectral space 是否真的被很少几个方向主导。

Stop condition

如果：

ESP10

明显低于 PP10，我就不建议继续：

ESP5+PP5；
spectral eigenvalue weighting；
加 W
down
	​

；
调 spectral rank。

只有 ESP10 至少接近 PP10，或者某个预算上有正向结果，才值得做：

PP5+ESP5.