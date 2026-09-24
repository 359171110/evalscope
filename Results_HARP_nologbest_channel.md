# HARP-nologbest 通道排序对照

本文档汇总 **HARP-nologbest** 通道排序变体的 `full8_v1` 结果。**Wanda 与 ENP 分开对照**，不放在同一张主表里。

共用层/专家分配：\((\gamma,\eta,\Delta)=(2,0.15,128)\)，Layer water-fill 为 \(1+\mathrm{CV}^2\)（NoLog），Layer-SP / Expert-SP，校准身份 CalibrationFree。只改 expert 内 channel 的保留顺序；分到的宽度不变。

\(z^{SP},z^{Act}\) 是 rank-normalized 到 \([0,1]\)（越大越重要）。\(N_e\) 来自 WikiText-2 train **128×2048** 的 `route_counts`。融合一律

\[
S_{e,c}=z^{SP}_{e,c}+w_e z^{Act}_{e,c}.
\]

两种 \(w_e\)：**fuse** 用 \(w_e=N_e/(N_e+512)\)；**cfuse** 用 \(w_e=1\)（\(N_e>0\)）否则 \(0\)。没有额外 \(\lambda\)。

Mean retained % 是八个任务 **score / Dense** 的不加权宏平均。Dense 为同一协议、同一 vLLM 栈上的未剪枝基座。


| Model    | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   |
| -------- | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ |
| Qwen3    | 0.9735 | 0.8709    | 0.7537     | 0.9492 | 0.8920   | 0.8552 | 0.9512    | 0.7880 |
| Qwen3.6  | 0.9831 | 0.8986    | 0.8713     | 0.9651 | 0.9480   | 0.8909 | 0.9573    | 0.8000 |
| DeepSeek | 0.7925 | 0.6622    | 0.5549     | 0.7058 | 0.2880   | 0.5613 | 0.5183    | 0.4680 |


每张对照表都带同一套 **HARP-nologbest**（Channel-SP）和 **HARP-nologbest-rand**（按分配宽度随机保留、不做排序）。rand 不是激活方法，只作无排序对照。

已完成：nologbest、wanda、fuse、wanda-cfuse、enp、enpfuse、enp-cfuse、rand（A / B，WikiText 校准）。C4 / MetaMath / Code-Alpaca 上的 HARP-wanda / HARP-enp / cfuse 与同构 Wanda / ENP 见 **E**。

原版 AIMER（整专家均匀 drop）不进入 A / B 主表，见 **C. 原版 AIMER 对照**。

HARP-nologbest 低分任务六轮重跑的 **per-dataset max** 见 **D**，不替换 A / B / C 里的单次分数。

**E** 按三个目标组织：校准域偏移、HARP 相对同构 Wanda/ENP、同一 HARP 内 cfuse 再提升。四个校准的 HARP 四方法与同构 Wanda/ENP full8 均已入表。四个校准上每个专家的 \(N_e\) 分布见 **F**。纯 HARP 的 per-token routed FFN 相对计算量见 **G**（G.1/G.2 用校准 cache 的 \(p\)；G.3 是 full8 generate 实测，含 HARP-nologbest 与 HSP \(\Delta=128\)）。**HARP-nologbest 宽度分配伪代码见 H**（A/B 只改 Channel 排序，不改 H 里的 \(K_{\ell,e}\)）。HARP-nologbest 的 \(\rho=1+\mathrm{CV}^2\) rank 分段见 **I**。

---

## A. Wanda 对照

| 名称 | token | Channel 规则 |
| --- | --- | --- |
| HARP-nologbest | `HARPNoLogBest` | Channel-SP prefix |
| HARP-nologbest-wanda | `HARPNoLogBestWanda` | Wanda；\(N_e=0\) 退回 Channel-SP |
| HARP-nologbest-fuse | `HARPNoLogBestFuse` | \(S=z^{SP}+w_e z^{Wanda}\)，\(w_e=N_e/(N_e+512)\) |
| HARP-nologbest-wanda-cfuse | `HARPNoLogBestWandaCFuse` | \(S=z^{SP}+w_e z^{Wanda}\)，\(w_e=1\) if \(N_e>0\) else \(0\) |
| HARP-nologbest-rand | `HARPNoLogBestRand` | 随机 prefix |

### Mean retained %

差值相对 HARP-nologbest（百分点）。


| Model    | Sparsity | nologbest | wanda | fuse | cfuse | rand | wanda Δ | fuse Δ | cfuse Δ | rand Δ |
| -------- | -------- | --------- | ----- | ---- | ----- | ---- | ------- | ------ | ------- | ------ |
| Qwen3    | 25%      | 94.2      | 93.0  | 94.6 | 94.4  | 93.1 | −1.1    | +0.5   | +0.2    | −1.1   |
| Qwen3    | 50%      | 68.7      | 63.1  | 71.5 | 71.3  | 61.8 | −5.5    | +2.8   | +2.6    | −6.9   |
| Qwen3.6  | 25%      | 96.5      | 96.3  | 95.9 | 96.5  | 95.6 | −0.2    | −0.6   | 0.0     | −0.9   |
| Qwen3.6  | 50%      | 85.5      | 68.4  | 82.3 | 79.7  | 79.0 | −17.2   | −3.2   | −5.8    | −6.5   |
| DeepSeek | 25%      | 68.5      | 17.5  | 78.7 | 81.5  | 63.1 | −51.0   | +10.2  | +13.0   | −5.4   |
| DeepSeek | 50%      | 37.6      | 8.7   | 34.1 | 32.2  | 36.0 | −28.9   | −3.5   | −5.4    | −1.6   |


### 读法

- 纯 Wanda 全面低于 Channel-SP，且在 Qwen3.6 50% 和 DeepSeek 上明显差于随机保留。校准激活单独当 channel 优先级，替代不了 Channel-SP。
- 两种融合都不是单向改进。Qwen3 两档和 DeepSeek 25% 高于 SP；Qwen3.6 50% 和 DeepSeek 50% 低于 SP。DeepSeek 25% 仍是最大正增益：fuse +10.2，**cfuse +13.0**。
- cfuse（有 token 就 \(w_e=1\)）相对 fuse（\(N/(N+512)\)）：Qwen3 几乎打平；DeepSeek 25% 更高；Qwen3.6 50% / DeepSeek 50% 更差，50% 上更接近 rand。把 Wanda 加满并没有稳定超过半加权。

### Qwen3 25%


| Method                | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| --------------------- | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense                 | 0.9735 | 0.8709    | 0.7537     | 0.9492 | 0.8920   | 0.8552 | 0.9512    | 0.7880 | —               |
| HARP-nologbest        | 0.9564 | 0.8304    | 0.7017     | 0.9439 | 0.8380   | 0.7966 | 0.8720    | 0.6960 | 94.2            |
| HARP-nologbest-wanda  | 0.9544 | 0.8408    | 0.6993     | 0.9280 | 0.8280   | 0.7858 | 0.8171    | 0.6960 | 93.0            |
| HARP-nologbest-fuse         | 0.9527 | 0.8387    | 0.7064     | 0.9416 | 0.8540   | 0.7918 | 0.8841    | 0.7000 | 94.6            |
| HARP-nologbest-wanda-cfuse  | 0.9549 | 0.8379    | 0.7072     | 0.9371 | 0.8400   | 0.7884 | 0.8902    | 0.6940 | 94.4            |
| HARP-nologbest-rand         | 0.9563 | 0.8197    | 0.7001     | 0.9189 | 0.8040   | 0.7881 | 0.8415    | 0.7220 | 93.1            |


Retained %：


| Method                | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| --------------------- | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| HARP-nologbest        | 98.2 | 95.3      | 93.1       | 99.4  | 93.9     | 93.1 | 91.7      | 88.3 | 94.2 |
| HARP-nologbest-wanda  | 98.0 | 96.5      | 92.8       | 97.8  | 92.8     | 91.9 | 85.9      | 88.3 | 93.0 |
| HARP-nologbest-fuse         | 97.9 | 96.3      | 93.7       | 99.2  | 95.7     | 92.6 | 92.9      | 88.8 | 94.6 |
| HARP-nologbest-wanda-cfuse  | 98.1 | 96.2      | 93.8       | 98.7  | 94.2     | 92.2 | 93.6      | 88.1 | 94.4 |
| HARP-nologbest-rand         | 98.2 | 94.1      | 92.9       | 96.8  | 90.1     | 92.2 | 88.5      | 91.6 | 93.1 |


### Qwen3 50%


| Method                | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| --------------------- | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense                 | 0.9735 | 0.8709    | 0.7537     | 0.9492 | 0.8920   | 0.8552 | 0.9512    | 0.7880 | —               |
| HARP-nologbest        | 0.8543 | 0.5905    | 0.5651     | 0.8188 | 0.4980   | 0.6099 | 0.4939    | 0.4220 | 68.7            |
| HARP-nologbest-wanda  | 0.8185 | 0.6984    | 0.5549     | 0.7612 | 0.4040   | 0.6109 | 0.2805    | 0.3220 | 63.1            |
| HARP-nologbest-fuse         | 0.8574 | 0.7143    | 0.5912     | 0.7908 | 0.4760   | 0.6024 | 0.5671    | 0.4480 | 71.5            |
| HARP-nologbest-wanda-cfuse  | 0.8549 | 0.7042    | 0.5809     | 0.7908 | 0.4980   | 0.6084 | 0.4939    | 0.4940 | 71.3            |
| HARP-nologbest-rand         | 0.8585 | 0.6128    | 0.5525     | 0.5951 | 0.3760   | 0.5599 | 0.3720    | 0.4180 | 61.8            |


Retained %：


| Method                | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| --------------------- | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| HARP-nologbest        | 87.8 | 67.8      | 75.0       | 86.3  | 55.8     | 71.3 | 51.9      | 53.6 | 68.7 |
| HARP-nologbest-wanda  | 84.1 | 80.2      | 73.6       | 80.2  | 45.3     | 71.4 | 29.5      | 40.9 | 63.1 |
| HARP-nologbest-fuse         | 88.1 | 82.0      | 78.4       | 83.3  | 53.4     | 70.4 | 59.6      | 56.9 | 71.5 |
| HARP-nologbest-wanda-cfuse  | 87.8 | 80.9      | 77.1       | 83.3  | 55.8     | 71.1 | 51.9      | 62.7 | 71.3 |
| HARP-nologbest-rand         | 88.2 | 70.4      | 73.3       | 62.7  | 42.2     | 65.5 | 39.1      | 53.0 | 61.8 |


Wanda 拉高 HellaSwag，HumanEval / MATH-500 掉得多，均值接近 rand。fuse 与 cfuse 均值几乎相同（71.5 / 71.3），都明显高于 SP；cfuse 的 HumanEval 没有 fuse 那么高（51.9 vs 59.6），MBPP 更高（62.7 vs 56.9）。

### Qwen3.6 25%


| Method                | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| --------------------- | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense                 | 0.9831 | 0.8986    | 0.8713     | 0.9651 | 0.9480   | 0.8909 | 0.9573    | 0.8000 | —               |
| HARP-nologbest        | 0.9805 | 0.8921    | 0.7948     | 0.9553 | 0.9160   | 0.8459 | 0.9329    | 0.7500 | 96.5            |
| HARP-nologbest-wanda  | 0.9718 | 0.8990    | 0.7893     | 0.9530 | 0.9280   | 0.8483 | 0.9268    | 0.7380 | 96.3            |
| HARP-nologbest-fuse         | 0.9577 | 0.8810    | 0.7972     | 0.9507 | 0.9200   | 0.8497 | 0.9390    | 0.7280 | 95.9            |
| HARP-nologbest-wanda-cfuse  | 0.9749 | 0.8919    | 0.7987     | 0.9507 | 0.9300   | 0.8505 | 0.9146    | 0.7520 | 96.5            |
| HARP-nologbest-rand         | 0.9732 | 0.8952    | 0.7908     | 0.9553 | 0.9180   | 0.8451 | 0.9207    | 0.7080 | 95.6            |


Retained %：


| Method                | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| --------------------- | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| HARP-nologbest        | 99.7 | 99.3      | 91.2       | 99.0  | 96.6     | 94.9 | 97.5      | 93.8 | 96.5 |
| HARP-nologbest-wanda  | 98.9 | 100.0     | 90.6       | 98.7  | 97.9     | 95.2 | 96.8      | 92.2 | 96.3 |
| HARP-nologbest-fuse         | 97.4 | 98.0      | 91.5       | 98.5  | 97.0     | 95.4 | 98.1      | 91.0 | 95.9 |
| HARP-nologbest-wanda-cfuse  | 99.2 | 99.3      | 91.7       | 98.5  | 98.1     | 95.5 | 95.5      | 94.0 | 96.5 |
| HARP-nologbest-rand         | 99.0 | 99.6      | 90.8       | 99.0  | 96.8     | 94.9 | 96.2      | 88.5 | 95.6 |


五组几乎打平；cfuse 与 SP 同为 96.5。

### Qwen3.6 50%


| Method                | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| --------------------- | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense                 | 0.9831 | 0.8986    | 0.8713     | 0.9651 | 0.9480   | 0.8909 | 0.9573    | 0.8000 | —               |
| HARP-nologbest        | 0.9557 | 0.8339    | 0.7403     | 0.9348 | 0.7620   | 0.7487 | 0.7134    | 0.5860 | 85.5            |
| HARP-nologbest-wanda  | 0.9307 | 0.7368    | 0.7048     | 0.6331 | 0.5280   | 0.6563 | 0.6585    | 0.2040 | 68.4            |
| HARP-nologbest-fuse         | 0.9453 | 0.8184    | 0.7403     | 0.8878 | 0.7320   | 0.7385 | 0.7317    | 0.4640 | 82.3            |
| HARP-nologbest-wanda-cfuse  | 0.9385 | 0.7470    | 0.7395     | 0.8143 | 0.6740   | 0.7247 | 0.7195    | 0.4960 | 79.7            |
| HARP-nologbest-rand         | 0.9411 | 0.7957    | 0.6882     | 0.8886 | 0.6680   | 0.6990 | 0.6341    | 0.4900 | 79.0            |


Retained %：


| Method                | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| --------------------- | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| HARP-nologbest        | 97.2 | 92.8      | 85.0       | 96.9  | 80.4     | 84.0 | 74.5      | 73.2 | 85.5 |
| HARP-nologbest-wanda  | 94.7 | 82.0      | 80.9       | 65.6  | 55.7     | 73.7 | 68.8      | 25.5 | 68.4 |
| HARP-nologbest-fuse         | 96.2 | 91.1      | 85.0       | 92.0  | 77.2     | 82.9 | 76.4      | 58.0 | 82.3 |
| HARP-nologbest-wanda-cfuse  | 95.5 | 83.1      | 84.9       | 84.4  | 71.1     | 81.3 | 75.2      | 62.0 | 79.7 |
| HARP-nologbest-rand         | 95.7 | 88.5      | 79.0       | 92.1  | 70.5     | 78.5 | 66.2      | 61.2 | 79.0 |


Wanda 在 GSM8K / MATH-500 / MBPP 上垮掉，低于 rand。fuse 拉回 SP 附近（82.3），仍低 3.2。cfuse（79.7）几乎贴着 rand（79.0），比半加权 fuse 更差：把 Wanda 加满会把 Qwen3.6 50% 往随机保留拖。

### DeepSeek 25%


| Method                | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| --------------------- | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense                 | 0.7925 | 0.6622    | 0.5549     | 0.7058 | 0.2880   | 0.5613 | 0.5183    | 0.4680 | —               |
| HARP-nologbest        | 0.7055 | 0.5219    | 0.5517     | 0.4807 | 0.1400   | 0.4618 | 0.1951    | 0.2060 | 68.5            |
| HARP-nologbest-wanda  | 0.1728 | 0.1630    | 0.2170     | 0.0159 | 0.0200   | 0.2307 | 0.0000    | 0.0180 | 17.5            |
| HARP-nologbest-fuse         | 0.7283 | 0.6097    | 0.5541     | 0.5709 | 0.1540   | 0.5079 | 0.2927    | 0.3020 | 78.7            |
| HARP-nologbest-wanda-cfuse  | 0.7254 | 0.6287    | 0.5517     | 0.5572 | 0.1620   | 0.5165 | 0.3598    | 0.3240 | 81.5            |
| HARP-nologbest-rand         | 0.6324 | 0.4876    | 0.5020     | 0.4352 | 0.0820   | 0.4520 | 0.2073    | 0.2340 | 63.1            |


Retained %：


| Method                | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| --------------------- | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| HARP-nologbest        | 89.0 | 78.8      | 99.4       | 68.1  | 48.6     | 82.3 | 37.6      | 44.0 | 68.5 |
| HARP-nologbest-wanda  | 21.8 | 24.6      | 39.1       | 2.3   | 6.9      | 41.1 | 0.0       | 3.8  | 17.5 |
| HARP-nologbest-fuse         | 91.9 | 92.1      | 99.9       | 80.9  | 53.5     | 90.5 | 56.5      | 64.5 | 78.7 |
| HARP-nologbest-wanda-cfuse  | 91.5 | 94.9      | 99.4       | 78.9  | 56.2     | 92.0 | 69.4      | 69.2 | 81.5 |
| HARP-nologbest-rand         | 79.8 | 73.6      | 90.5       | 61.7  | 28.5     | 80.5 | 40.0      | 50.0 | 63.1 |


纯 Wanda 不可用。两种融合都远高于 SP；cfuse（81.5）高于 fuse（78.7），主要来自 HumanEval / MBPP / HellaSwag。

### DeepSeek 50%


| Method                | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| --------------------- | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense                 | 0.7925 | 0.6622    | 0.5549     | 0.7058 | 0.2880   | 0.5613 | 0.5183    | 0.4680 | —               |
| HARP-nologbest        | 0.4614 | 0.3078    | 0.5170     | 0.0667 | 0.0240   | 0.3446 | 0.0366    | 0.0780 | 37.6            |
| HARP-nologbest-wanda  | 0.0400 | 0.1286    | 0.0489     | 0.0015 | 0.0040   | 0.1846 | 0.0000    | 0.0080 | 8.7             |
| HARP-nologbest-fuse         | 0.3230 | 0.2919    | 0.4949     | 0.1183 | 0.0440   | 0.2547 | 0.0488    | 0.0560 | 34.1            |
| HARP-nologbest-wanda-cfuse  | 0.2748 | 0.2703    | 0.4791     | 0.0940 | 0.0360   | 0.2529 | 0.0488    | 0.0720 | 32.2            |
| HARP-nologbest-rand         | 0.5011 | 0.3390    | 0.5264     | 0.0311 | 0.0180   | 0.3280 | 0.0061    | 0.0400 | 36.0            |


Retained %：


| Method                | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| --------------------- | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| HARP-nologbest        | 58.2 | 46.5      | 93.2       | 9.5   | 8.3      | 61.4 | 7.1       | 16.7 | 37.6 |
| HARP-nologbest-wanda  | 5.0  | 19.4      | 8.8        | 0.2   | 1.4      | 32.9 | 0.0       | 1.7  | 8.7  |
| HARP-nologbest-fuse         | 40.8 | 44.1      | 89.2       | 16.8  | 15.3     | 45.4 | 9.4       | 12.0 | 34.1 |
| HARP-nologbest-wanda-cfuse  | 34.7 | 40.8      | 86.3       | 13.3  | 12.5     | 45.1 | 9.4       | 15.4 | 32.2 |
| HARP-nologbest-rand         | 63.2 | 51.2      | 94.9       | 4.4   | 6.2      | 58.4 | 1.2       | 8.5  | 36.0 |


50% 上四组都弱。Wanda 仍崩溃。fuse / cfuse 都低于 SP 和 rand；cfuse 比半加权 fuse 再低 1.9。

---

## B. ENP 对照

| 名称 | token | Channel 规则 |
| --- | --- | --- |
| HARP-nologbest | `HARPNoLogBest` | Channel-SP prefix |
| HARP-nologbest-enp | `HARPNoLogBestENP` | ENP-COS；\(N_e=0\) 退回 Channel-SP |
| HARP-nologbest-enpfuse | `HARPNoLogBestENPFuse` | \(S=z^{SP}+w_e z^{ENP}\)，\(w_e=N_e/(N_e+512)\) |
| HARP-nologbest-enp-cfuse | `HARPNoLogBestENPCFuse` | \(S=z^{SP}+w_e z^{ENP}\)，\(w_e=1\) if \(N_e>0\) else \(0\) |
| HARP-nologbest-rand | `HARPNoLogBestRand` | 随机 prefix |

### Mean retained %

差值相对 HARP-nologbest（百分点）。


| Model    | Sparsity | nologbest | enp  | enpfuse | cfuse | rand | enp Δ | enpfuse Δ | cfuse Δ | rand Δ |
| -------- | -------- | --------- | ---- | ------- | ----- | ---- | ----- | --------- | ------- | ------ |
| Qwen3    | 25%      | 94.2      | 92.7 | 93.9    | 93.5  | 93.1 | −1.4  | −0.3      | −0.7    | −1.1   |
| Qwen3    | 50%      | 68.7      | 60.0 | 67.8    | 66.4  | 61.8 | −8.7  | −0.9      | −2.3    | −6.9   |
| Qwen3.6  | 25%      | 96.5      | 96.1 | 96.5    | 96.2  | 95.6 | −0.4  | 0.0       | −0.3    | −0.9   |
| Qwen3.6  | 50%      | 85.5      | 72.9 | 82.8    | 83.6  | 79.0 | −12.6 | −2.7      | −1.9    | −6.5   |
| DeepSeek | 25%      | 68.5      | 55.7 | 69.0    | 69.3  | 63.1 | −12.8 | +0.5      | +0.8    | −5.4   |
| DeepSeek | 50%      | 37.6      | 10.6 | 36.4    | 35.5  | 36.0 | −27.0 | −1.2      | −2.1    | −1.6   |


### 读法

- 纯 ENP 全面低于 Channel-SP。DeepSeek 50% 掉到 10.6，接近不可用；Qwen3.6 50% 掉 12.6，HumanEval retained 只有 31.8。
- ENP 在多数 50% 格子上低于或接近 rand（Qwen3 50%：60.0 vs 61.8；DeepSeek 50%：10.6 vs 36.0）。单独用 ENP-COS 排序也不如随机保留稳。
- 两种融合都把纯 ENP 的大回撤收回来，相对纯 SP 几乎没有正增益。enpfuse / cfuse 彼此接近：Qwen3 上半加权略好，Qwen3.6 50% 上 cfuse 略好（83.6 vs 82.8），DeepSeek 25% 都只有 +0.5 / +0.8。没有 Wanda 侧那种 DeepSeek 25% +10 以上的跳变。

### Qwen3 25%


| Method                  | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| ----------------------- | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense                   | 0.9735 | 0.8709    | 0.7537     | 0.9492 | 0.8920   | 0.8552 | 0.9512    | 0.7880 | —               |
| HARP-nologbest          | 0.9564 | 0.8304    | 0.7017     | 0.9439 | 0.8380   | 0.7966 | 0.8720    | 0.6960 | 94.2            |
| HARP-nologbest-enp      | 0.9529 | 0.8292    | 0.7111     | 0.9356 | 0.8120   | 0.7784 | 0.8171    | 0.6900 | 92.7            |
| HARP-nologbest-enpfuse   | 0.9518 | 0.8344    | 0.7056     | 0.9401 | 0.8320   | 0.7868 | 0.8598    | 0.7060 | 93.9            |
| HARP-nologbest-enp-cfuse | 0.9515 | 0.8348    | 0.7064     | 0.9386 | 0.8200   | 0.7851 | 0.8354    | 0.7100 | 93.5            |
| HARP-nologbest-rand      | 0.9563 | 0.8197    | 0.7001     | 0.9189 | 0.8040   | 0.7881 | 0.8415    | 0.7220 | 93.1            |


Retained %：


| Method                  | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| ----------------------- | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| HARP-nologbest          | 98.2 | 95.3      | 93.1       | 99.4  | 93.9     | 93.1 | 91.7      | 88.3 | 94.2 |
| HARP-nologbest-enp      | 97.9 | 95.2      | 94.3       | 98.6  | 91.0     | 91.0 | 85.9      | 87.6 | 92.7 |
| HARP-nologbest-enpfuse   | 97.8 | 95.8      | 93.6       | 99.0  | 93.3     | 92.0 | 90.4      | 89.6 | 93.9 |
| HARP-nologbest-enp-cfuse | 97.7 | 95.9      | 93.7       | 98.9  | 91.9     | 91.8 | 87.8      | 90.1 | 93.5 |
| HARP-nologbest-rand      | 98.2 | 94.1      | 92.9       | 96.8  | 90.1     | 92.2 | 88.5      | 91.6 | 93.1 |


### Qwen3 50%


| Method                  | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| ----------------------- | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense                   | 0.9735 | 0.8709    | 0.7537     | 0.9492 | 0.8920   | 0.8552 | 0.9512    | 0.7880 | —               |
| HARP-nologbest          | 0.8543 | 0.5905    | 0.5651     | 0.8188 | 0.4980   | 0.6099 | 0.4939    | 0.4220 | 68.7            |
| HARP-nologbest-enp      | 0.7852 | 0.6429    | 0.5604     | 0.7089 | 0.3120   | 0.5471 | 0.2866    | 0.3720 | 60.0            |
| HARP-nologbest-enpfuse   | 0.8427 | 0.6703    | 0.5620     | 0.7726 | 0.4100   | 0.5810 | 0.5122    | 0.4320 | 67.8            |
| HARP-nologbest-enp-cfuse | 0.8433 | 0.6856    | 0.5691     | 0.7892 | 0.4120   | 0.5799 | 0.4024    | 0.4020 | 66.4            |
| HARP-nologbest-rand      | 0.8585 | 0.6128    | 0.5525     | 0.5951 | 0.3760   | 0.5599 | 0.3720    | 0.4180 | 61.8            |


Retained %：


| Method                  | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| ----------------------- | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| HARP-nologbest          | 87.8 | 67.8      | 75.0       | 86.3  | 55.8     | 71.3 | 51.9      | 53.6 | 68.7 |
| HARP-nologbest-enp      | 80.7 | 73.8      | 74.4       | 74.7  | 35.0     | 64.0 | 30.1      | 47.2 | 60.0 |
| HARP-nologbest-enpfuse   | 86.6 | 77.0      | 74.6       | 81.4  | 46.0     | 67.9 | 53.8      | 54.8 | 67.8 |
| HARP-nologbest-enp-cfuse | 86.6 | 78.7      | 75.5       | 83.1  | 46.2     | 67.8 | 42.3      | 51.0 | 66.4 |
| HARP-nologbest-rand      | 88.2 | 70.4      | 73.3       | 62.7  | 42.2     | 65.5 | 39.1      | 53.0 | 61.8 |


纯 ENP 低于 rand。enpfuse 接近 SP（67.8）；cfuse 略低（66.4），HumanEval 明显弱于半加权（42.3 vs 53.8）。

### Qwen3.6 25%


| Method                  | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| ----------------------- | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense                   | 0.9831 | 0.8986    | 0.8713     | 0.9651 | 0.9480   | 0.8909 | 0.9573    | 0.8000 | —               |
| HARP-nologbest          | 0.9805 | 0.8921    | 0.7948     | 0.9553 | 0.9160   | 0.8459 | 0.9329    | 0.7500 | 96.5            |
| HARP-nologbest-enp      | 0.9749 | 0.9033    | 0.7940     | 0.9621 | 0.9220   | 0.8573 | 0.9146    | 0.7120 | 96.1            |
| HARP-nologbest-enpfuse   | 0.9735 | 0.8991    | 0.7972     | 0.9606 | 0.9240   | 0.8564 | 0.9146    | 0.7440 | 96.5            |
| HARP-nologbest-enp-cfuse | 0.9749 | 0.9030    | 0.7932     | 0.9560 | 0.9220   | 0.8599 | 0.8963    | 0.7420 | 96.2            |
| HARP-nologbest-rand      | 0.9732 | 0.8952    | 0.7908     | 0.9553 | 0.9180   | 0.8451 | 0.9207    | 0.7080 | 95.6            |


Retained %：


| Method                  | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| ----------------------- | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| HARP-nologbest          | 99.7 | 99.3      | 91.2       | 99.0  | 96.6     | 94.9 | 97.5      | 93.8 | 96.5 |
| HARP-nologbest-enp      | 99.2 | 100.5     | 91.1       | 99.7  | 97.3     | 96.2 | 95.5      | 89.0 | 96.1 |
| HARP-nologbest-enpfuse   | 99.0 | 100.1     | 91.5       | 99.5  | 97.5     | 96.1 | 95.5      | 93.0 | 96.5 |
| HARP-nologbest-enp-cfuse | 99.2 | 100.5     | 91.0       | 99.1  | 97.3     | 96.5 | 93.6      | 92.8 | 96.2 |
| HARP-nologbest-rand      | 99.0 | 99.6      | 90.8       | 99.0  | 96.8     | 94.9 | 96.2      | 88.5 | 95.6 |


enpfuse 与 SP 打平（96.5）；cfuse 96.2。

### Qwen3.6 50%


| Method                  | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| ----------------------- | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense                   | 0.9831 | 0.8986    | 0.8713     | 0.9651 | 0.9480   | 0.8909 | 0.9573    | 0.8000 | —               |
| HARP-nologbest          | 0.9557 | 0.8339    | 0.7403     | 0.9348 | 0.7620   | 0.7487 | 0.7134    | 0.5860 | 85.5            |
| HARP-nologbest-enp      | 0.9346 | 0.8299    | 0.7190     | 0.8832 | 0.5540   | 0.6911 | 0.3049    | 0.4280 | 72.9            |
| HARP-nologbest-enpfuse   | 0.9504 | 0.8315    | 0.7530     | 0.9409 | 0.7360   | 0.7456 | 0.6037    | 0.5180 | 82.8            |
| HARP-nologbest-enp-cfuse | 0.9465 | 0.8387    | 0.7498     | 0.9295 | 0.7100   | 0.7342 | 0.6951    | 0.5380 | 83.6            |
| HARP-nologbest-rand      | 0.9411 | 0.7957    | 0.6882     | 0.8886 | 0.6680   | 0.6990 | 0.6341    | 0.4900 | 79.0            |


Retained %：


| Method                  | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| ----------------------- | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| HARP-nologbest          | 97.2 | 92.8      | 85.0       | 96.9  | 80.4     | 84.0 | 74.5      | 73.2 | 85.5 |
| HARP-nologbest-enp      | 95.1 | 92.4      | 82.5       | 91.5  | 58.4     | 77.6 | 31.8      | 53.5 | 72.9 |
| HARP-nologbest-enpfuse   | 96.7 | 92.5      | 86.4       | 97.5  | 77.6     | 83.7 | 63.1      | 64.8 | 82.8 |
| HARP-nologbest-enp-cfuse | 96.3 | 93.3      | 86.1       | 96.3  | 74.9     | 82.4 | 72.6      | 67.2 | 83.6 |
| HARP-nologbest-rand      | 95.7 | 88.5      | 79.0       | 92.1  | 70.5     | 78.5 | 66.2      | 61.2 | 79.0 |


纯 ENP 的 HumanEval 掉到 31.8。enpfuse 高于 rand，仍低于 SP 2.7。cfuse（83.6）略高于半加权，HumanEval 拉回 72.6。

### DeepSeek 25%


| Method                  | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| ----------------------- | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense                   | 0.7925 | 0.6622    | 0.5549     | 0.7058 | 0.2880   | 0.5613 | 0.5183    | 0.4680 | —               |
| HARP-nologbest          | 0.7055 | 0.5219    | 0.5517     | 0.4807 | 0.1400   | 0.4618 | 0.1951    | 0.2060 | 68.5            |
| HARP-nologbest-enp      | 0.4865 | 0.3481    | 0.5185     | 0.3442 | 0.0680   | 0.4490 | 0.2134    | 0.2080 | 55.7            |
| HARP-nologbest-enpfuse   | 0.7122 | 0.5610    | 0.5399     | 0.4587 | 0.1060   | 0.4772 | 0.2683    | 0.1940 | 69.0            |
| HARP-nologbest-enp-cfuse | 0.7128 | 0.5522    | 0.5478     | 0.4799 | 0.1240   | 0.4745 | 0.2622    | 0.1700 | 69.3            |
| HARP-nologbest-rand      | 0.6324 | 0.4876    | 0.5020     | 0.4352 | 0.0820   | 0.4520 | 0.2073    | 0.2340 | 63.1            |


Retained %：


| Method                  | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| ----------------------- | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| HARP-nologbest          | 89.0 | 78.8      | 99.4       | 68.1  | 48.6     | 82.3 | 37.6      | 44.0 | 68.5 |
| HARP-nologbest-enp      | 61.4 | 52.6      | 93.4       | 48.8  | 23.6     | 80.0 | 41.2      | 44.4 | 55.7 |
| HARP-nologbest-enpfuse   | 89.9 | 84.7      | 97.3       | 65.0  | 36.8     | 85.0 | 51.8      | 41.5 | 69.0 |
| HARP-nologbest-enp-cfuse | 89.9 | 83.4      | 98.7       | 68.0  | 43.1     | 84.5 | 50.6      | 36.3 | 69.3 |
| HARP-nologbest-rand      | 79.8 | 73.6      | 90.5       | 61.7  | 28.5     | 80.5 | 40.0      | 50.0 | 63.1 |


enpfuse 略高于 SP（+0.5），cfuse +0.8；都远小于 Wanda 融合在同一格子上的增益。

### DeepSeek 50%


| Method                  | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| ----------------------- | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense                   | 0.7925 | 0.6622    | 0.5549     | 0.7058 | 0.2880   | 0.5613 | 0.5183    | 0.4680 | —               |
| HARP-nologbest          | 0.4614 | 0.3078    | 0.5170     | 0.0667 | 0.0240   | 0.3446 | 0.0366    | 0.0780 | 37.6            |
| HARP-nologbest-enp      | 0.1536 | 0.1428    | 0.1681     | 0.0045 | 0.0020   | 0.0688 | 0.0000    | 0.0000 | 10.6            |
| HARP-nologbest-enpfuse   | 0.3791 | 0.2952    | 0.5217     | 0.0872 | 0.0440   | 0.3433 | 0.0305    | 0.0480 | 36.4            |
| HARP-nologbest-enp-cfuse | 0.3872 | 0.2975    | 0.5138     | 0.0895 | 0.0220   | 0.3440 | 0.0305    | 0.0460 | 35.5            |
| HARP-nologbest-rand      | 0.5011 | 0.3390    | 0.5264     | 0.0311 | 0.0180   | 0.3280 | 0.0061    | 0.0400 | 36.0            |


Retained %：


| Method                  | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| ----------------------- | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| HARP-nologbest          | 58.2 | 46.5      | 93.2       | 9.5   | 8.3      | 61.4 | 7.1       | 16.7 | 37.6 |
| HARP-nologbest-enp      | 19.4 | 21.6      | 30.3       | 0.6   | 0.7      | 12.3 | 0.0       | 0.0  | 10.6 |
| HARP-nologbest-enpfuse   | 47.8 | 44.6      | 94.0       | 12.4  | 15.3     | 61.2 | 5.9       | 10.3 | 36.4 |
| HARP-nologbest-enp-cfuse | 48.9 | 44.9      | 92.6       | 12.7  | 7.6      | 61.3 | 5.9       | 9.8  | 35.5 |
| HARP-nologbest-rand      | 63.2 | 51.2      | 94.9       | 4.4   | 6.2      | 58.4 | 1.2       | 8.5  | 36.0 |


纯 ENP 崩溃。enpfuse / cfuse 与 SP / rand 同档，没有拉开。

---

## C. 原版 AIMER 对照

本节只比 **HARP-nologbest** 与论文原版 **AIMER**（`AIMER_Expert/`，token `AIMER`）。不改 A / B 的 Wanda、ENP、融合、rand 行。

AIMER 是 data-free 整专家剪枝：每层对 routed expert 拼 `gate`/`up`/`down`，按 \(S=\mathrm{MeanAbs}(w)/\mathrm{RMS}(w)\) 降序丢掉最高分的 \(k\) 个专家，各 MoE 层 \(k\) 相同；channel 宽度不变。不是 AIMER-Channel / Mix / Unify。与 HARP-nologbest 的剪枝粒度不同（专家 drop vs 通道 prefix），只作同协议 full8 对照。

差值相对 HARP-nologbest（百分点）。

### Mean retained %


| Model    | Sparsity | nologbest | AIMER | AIMER Δ |
| -------- | -------- | --------- | ----- | ------- |
| Qwen3    | 25%      | 94.2      | 95.4  | +1.2    |
| Qwen3    | 50%      | 68.7      | 71.8  | +3.1    |
| Qwen3.6  | 25%      | 96.5      | 97.6  | +1.1    |
| Qwen3.6  | 50%      | 85.5      | 72.4  | −13.1   |
| DeepSeek | 25%      | 68.5      | 64.0  | −4.5    |
| DeepSeek | 50%      | 37.6      | 23.7  | −13.9   |


### 读法

- 25%：Qwen3 / Qwen3.6 上 AIMER 略高（+1.2 / +1.1）；DeepSeek 上 AIMER 低 4.5。
- 50%：Qwen3 上 AIMER 高 3.1（HellaSwag / 代码拉高，GSM8K / MMLU 更差）；Qwen3.6 和 DeepSeek 上 AIMER 明显更差（−13.1 / −13.9），Qwen3.6 的 GSM8K 从 96.9 掉到 46.3。

### Qwen3 25%


| Method         | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| -------------- | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense          | 0.9735 | 0.8709    | 0.7537     | 0.9492 | 0.8920   | 0.8552 | 0.9512    | 0.7880 | —               |
| HARP-nologbest | 0.9564 | 0.8304    | 0.7017     | 0.9439 | 0.8380   | 0.7966 | 0.8720    | 0.6960 | 94.2            |
| AIMER          | 0.9513 | 0.8400    | 0.7222     | 0.9363 | 0.8600   | 0.7415 | 0.9146    | 0.7520 | 95.4            |


Retained %：


| Method         | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| -------------- | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| HARP-nologbest | 98.2 | 95.3      | 93.1       | 99.4  | 93.9     | 93.1 | 91.7      | 88.3 | 94.2 |
| AIMER          | 97.7 | 96.5      | 95.8       | 98.6  | 96.4     | 86.7 | 96.2      | 95.4 | 95.4 |


AIMER 高在 HellaSwag、WinoGrande、MATH-500、HumanEval、MBPP；MMLU 低 6.4。

### Qwen3 50%


| Method         | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| -------------- | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense          | 0.9735 | 0.8709    | 0.7537     | 0.9492 | 0.8920   | 0.8552 | 0.9512    | 0.7880 | —               |
| HARP-nologbest | 0.8543 | 0.5905    | 0.5651     | 0.8188 | 0.4980   | 0.6099 | 0.4939    | 0.4220 | 68.7            |
| AIMER          | 0.8213 | 0.7234    | 0.6298     | 0.7157 | 0.5380   | 0.5211 | 0.6037    | 0.4980 | 71.8            |


Retained %：


| Method         | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| -------------- | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| HARP-nologbest | 87.8 | 67.8      | 75.0       | 86.3  | 55.8     | 71.3 | 51.9      | 53.6 | 68.7 |
| AIMER          | 84.4 | 83.1      | 83.6       | 75.4  | 60.3     | 60.9 | 63.5      | 63.2 | 71.8 |


AIMER 赢在 HellaSwag（+15.3）和代码；GSM8K / MMLU / ARC 低于 HARP-nologbest。

### Qwen3.6 25%


| Method         | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| -------------- | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense          | 0.9831 | 0.8986    | 0.8713     | 0.9651 | 0.9480   | 0.8909 | 0.9573    | 0.8000 | —               |
| HARP-nologbest | 0.9805 | 0.8921    | 0.7948     | 0.9553 | 0.9160   | 0.8459 | 0.9329    | 0.7500 | 96.5            |
| AIMER          | 0.9798 | 0.8995    | 0.8350     | 0.9598 | 0.9080   | 0.8630 | 0.9451    | 0.7540 | 97.6            |


Retained %：


| Method         | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| -------------- | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| HARP-nologbest | 99.7 | 99.3      | 91.2       | 99.0  | 96.6     | 94.9 | 97.5      | 93.8 | 96.5 |
| AIMER          | 99.7 | 100.1     | 95.8       | 99.5  | 95.8     | 96.9 | 98.7      | 94.2 | 97.6 |


两边都接近 Dense。AIMER 主要高在 WinoGrande（+4.6）和 MMLU。

### Qwen3.6 50%


| Method         | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| -------------- | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense          | 0.9831 | 0.8986    | 0.8713     | 0.9651 | 0.9480   | 0.8909 | 0.9573    | 0.8000 | —               |
| HARP-nologbest | 0.9557 | 0.8339    | 0.7403     | 0.9348 | 0.7620   | 0.7487 | 0.7134    | 0.5860 | 85.5            |
| AIMER          | 0.9301 | 0.6847    | 0.7964     | 0.4473 | 0.6080   | 0.6204 | 0.6037    | 0.5900 | 72.4            |


Retained %：


| Method         | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| -------------- | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| HARP-nologbest | 97.2 | 92.8      | 85.0       | 96.9  | 80.4     | 84.0 | 74.5      | 73.2 | 85.5 |
| AIMER          | 94.6 | 76.2      | 91.4       | 46.3  | 64.1     | 69.6 | 63.1      | 73.8 | 72.4 |


HARP-nologbest 明显更好。AIMER 的 GSM8K 崩到 46.3；HellaSwag / MATH-500 / MMLU / HumanEval 也低一截。WinoGrande 和 MBPP 略高。

### DeepSeek 25%


| Method         | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| -------------- | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense          | 0.7925 | 0.6622    | 0.5549     | 0.7058 | 0.2880   | 0.5613 | 0.5183    | 0.4680 | —               |
| HARP-nologbest | 0.7055 | 0.5219    | 0.5517     | 0.4807 | 0.1400   | 0.4618 | 0.1951    | 0.2060 | 68.5            |
| AIMER          | 0.6438 | 0.4752    | 0.5193     | 0.5027 | 0.1220   | 0.4295 | 0.1829    | 0.1860 | 64.0            |


Retained %：


| Method         | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| -------------- | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| HARP-nologbest | 89.0 | 78.8      | 99.4       | 68.1  | 48.6     | 82.3 | 37.6      | 44.0 | 68.5 |
| AIMER          | 81.2 | 71.8      | 93.6       | 71.2  | 42.4     | 76.5 | 35.3      | 39.7 | 64.0 |


除 GSM8K 外 AIMER 全面更低。

### DeepSeek 50%


| Method         | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| -------------- | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense          | 0.7925 | 0.6622    | 0.5549     | 0.7058 | 0.2880   | 0.5613 | 0.5183    | 0.4680 | —               |
| HARP-nologbest | 0.4614 | 0.3078    | 0.5170     | 0.0667 | 0.0240   | 0.3446 | 0.0366    | 0.0780 | 37.6            |
| AIMER          | 0.2041 | 0.2066    | 0.4167     | 0.0356 | 0.0100   | 0.2661 | 0.0000    | 0.0060 | 23.7            |


Retained %：


| Method         | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| -------------- | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| HARP-nologbest | 58.2 | 46.5      | 93.2       | 9.5   | 8.3      | 61.4 | 7.1       | 16.7 | 37.6 |
| AIMER          | 25.8 | 31.2      | 75.1       | 5.0   | 3.5      | 47.4 | 0.0       | 1.3  | 23.7 |


AIMER 八项都更低；HumanEval / MBPP 接近零。

---

## D. HARP-nologbest 低分任务重跑（六轮 max）

单次主跑仍是 `202609162249`，A / B / C **未替换**。

协议仍是 `full8_v1` greedy（`temperature=0`）。只重跑该格 retained 低于格均值的任务（跳过 MMLU）。每格 6 轮，时间戳 `202609172126`–`202609172131`。下表 **max** 是这六轮之间的最高 score，不是 `max(主跑, 重跑)`。

两处重跑 max 低于主跑：DeepSeek 25% MATH-500（0.1380 vs 0.1400）、DeepSeek 50% MBPP（0.0720 vs 0.0780）。

「代入 mean」= 重跑任务换成六轮 max、其余任务仍用主跑，仅作对照，**未写入 A / B**。


| Model    | Sparsity | 重跑数据集 | 主跑 mean | 代入 mean | Δ |
| -------- | -------- | ---------- | --------- | --------- | --- |
| Qwen3    | 25%      | WinoGrande, MATH-500, HumanEval, MBPP | 94.2 | 94.8 | +0.6 |
| Qwen3    | 50%      | HellaSwag, MATH-500, HumanEval, MBPP | 68.7 | 69.3 | +0.7 |
| Qwen3.6  | 25%      | WinoGrande, MATH-500, MBPP | 96.5 | 97.1 | +0.6 |
| Qwen3.6  | 50%      | WinoGrande, MATH-500, HumanEval, MBPP | 85.5 | 86.0 | +0.5 |
| DeepSeek | 25%      | GSM8K, MATH-500, HumanEval, MBPP | 68.5 | 69.5 | +1.0 |
| DeepSeek | 50%      | GSM8K, MATH-500, HumanEval, MBPP | 37.6 | 38.2 | +0.6 |


### Score：主跑 vs 六轮 max

空格表示该任务未重跑。


| Model    | Sp  | 来源 | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   |
| -------- | --- | ---- | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ |
| Qwen3    | 25% | 主跑 | 0.9564 | 0.8304    | 0.7017     | 0.9439 | 0.8380   | 0.7966 | 0.8720    | 0.6960 |
| Qwen3    | 25% | max  |        |           | 0.7032     |        | 0.8600   |        | 0.8841    | 0.7040 |
| Qwen3    | 50% | 主跑 | 0.8543 | 0.5905    | 0.5651     | 0.8188 | 0.4980   | 0.6099 | 0.4939    | 0.4220 |
| Qwen3    | 50% | max  |        | 0.5958    |            |        | 0.5100   |        | 0.5183    | 0.4280 |
| Qwen3.6  | 25% | 主跑 | 0.9805 | 0.8921    | 0.7948     | 0.9553 | 0.9160   | 0.8459 | 0.9329    | 0.7500 |
| Qwen3.6  | 25% | max  |        |           | 0.8011     |        | 0.9300   |        |           | 0.7680 |
| Qwen3.6  | 50% | 主跑 | 0.9557 | 0.8339    | 0.7403     | 0.9348 | 0.7620   | 0.7487 | 0.7134    | 0.5860 |
| Qwen3.6  | 50% | max  |        |           | 0.7427     |        | 0.7740   |        | 0.7317    | 0.5920 |
| DeepSeek | 25% | 主跑 | 0.7055 | 0.5219    | 0.5517     | 0.4807 | 0.1400   | 0.4618 | 0.1951    | 0.2060 |
| DeepSeek | 25% | max  |        |           |            | 0.4920 | 0.1380   |        | 0.2134    | 0.2220 |
| DeepSeek | 50% | 主跑 | 0.4614 | 0.3078    | 0.5170     | 0.0667 | 0.0240   | 0.3446 | 0.0366    | 0.0780 |
| DeepSeek | 50% | max  |        |           |            | 0.0667 | 0.0340   |        | 0.0488    | 0.0720 |


### Retained %：重跑任务


| Model    | Sp  | 数据集 | 主跑 | 六轮 max | Δ |
| -------- | --- | ------ | ---- | -------- | --- |
| Qwen3    | 25% | WinoGrande | 93.1 | 93.3 | +0.2 |
| Qwen3    | 25% | MATH-500 | 93.9 | 96.4 | +2.5 |
| Qwen3    | 25% | HumanEval | 91.7 | 92.9 | +1.2 |
| Qwen3    | 25% | MBPP | 88.3 | 89.3 | +1.0 |
| Qwen3    | 50% | HellaSwag | 67.8 | 68.4 | +0.6 |
| Qwen3    | 50% | MATH-500 | 55.8 | 57.2 | +1.4 |
| Qwen3    | 50% | HumanEval | 51.9 | 54.5 | +2.6 |
| Qwen3    | 50% | MBPP | 53.6 | 54.3 | +0.7 |
| Qwen3.6  | 25% | WinoGrande | 91.2 | 91.9 | +0.7 |
| Qwen3.6  | 25% | MATH-500 | 96.6 | 98.1 | +1.5 |
| Qwen3.6  | 25% | MBPP | 93.8 | 96.0 | +2.2 |
| Qwen3.6  | 50% | WinoGrande | 85.0 | 85.2 | +0.2 |
| Qwen3.6  | 50% | MATH-500 | 80.4 | 81.6 | +1.2 |
| Qwen3.6  | 50% | HumanEval | 74.5 | 76.4 | +1.9 |
| Qwen3.6  | 50% | MBPP | 73.2 | 74.0 | +0.8 |
| DeepSeek | 25% | GSM8K | 68.1 | 69.7 | +1.6 |
| DeepSeek | 25% | MATH-500 | 48.6 | 47.9 | −0.7 |
| DeepSeek | 25% | HumanEval | 37.6 | 41.2 | +3.6 |
| DeepSeek | 25% | MBPP | 44.0 | 47.4 | +3.4 |
| DeepSeek | 50% | GSM8K | 9.5 | 9.5 | 0.0 |
| DeepSeek | 50% | MATH-500 | 8.3 | 11.8 | +3.5 |
| DeepSeek | 50% | HumanEval | 7.1 | 9.4 | +2.3 |
| DeepSeek | 50% | MBPP | 16.7 | 15.4 | −1.3 |

---

## E. 校准域偏移与 HARP 约束

本节要回答三件事，Wanda / ENP 始终分开。四个校准都是 128×2048：WikiText-2 train（A / B，同构见 `Results_HARP.md`）、C4 train、MetaMath、Code-Alpaca。

1. **校准域会进下游。** 同一方法（同构 Wanda/ENP，或 HARP 宽度下的 Wanda/ENP 通道），只换校准集，full8 均值和任务分数会变，而且不是整体平移。
2. **HARP 异构框架能约束这份校准信号。** 对照是同一校准上的同构 Wanda / 同构 ENP。HARP-wanda / HARP-enp 换层–专家水填宽度、通道仍跟激活走；cfuse 再加 Channel-SP。
3. **在已经使用 HARP 的前提下，cfuse 还能再抬。** 对照是同一校准、同一 HARP 宽度下的 HARP-nologbest-wanda / HARP-nologbest-enp。

记号：

| 名称                                | 含义                                                  | 校准                               |
| --------------------------------- | --------------------------------------------------- | -------------------------------- |
| 同构 Wanda / 同构 ENP                 | `Wanda` / `ENP`，均匀宽度                                | WikiText、C4、MetaMath、Code-Alpaca |
| HARP-wanda / HARP-enp             | `HARPNoLogBestWanda` / `HARPNoLogBestENP`           | 同上                               |
| HARP-wanda-cfuse / HARP-enp-cfuse | `HARPNoLogBestWandaCFuse` / `HARPNoLogBestENPCFuse` | 同上                               |

WikiText 目录身份是 `CalibrationFree` / `WikiText128x2048`；C4 / MetaMath / Code-Alpaca 是 `C4128x2048` / `MetaMath128x2048` / `CodeAlpaca128x2048`（`202609180117`）。只换 channel 激活统计；Layer-SP / Expert-SP 与 HARP 宽度不变。Mean retained % 与 A / B 相同。A / B 主表不改。

### E.1 同一方法、不同校准 → 下游 full8 不同

#### HARP-wanda / HARP-enp

| Model    | Sp  | Wanda Wiki | Wanda C4 | Wanda MM | Wanda CA | ENP Wiki | ENP C4 | ENP MM | ENP CA |
| -------- | --- | ---------- | -------- | -------- | -------- | -------- | ------ | ------ | ------ |
| Qwen3    | 25% | 93.0       | 93.5     | 91.9     | 93.8     | 92.7     | 92.3   | 93.7   | 93.8   |
| Qwen3    | 50% | 63.1       | 59.0     | 67.6     | 73.4     | 60.0     | 56.9   | 70.5   | 72.9   |
| Qwen3.6  | 25% | 96.3       | 96.1     | 96.5     | 96.1     | 96.1     | 95.9   | 97.5   | 96.4   |
| Qwen3.6  | 50% | 68.4       | 75.8     | 82.5     | 80.9     | 72.9     | 66.7   | 81.5   | 84.8   |
| DeepSeek | 25% | 17.5       | 19.2     | 22.1     | 21.5     | 55.7     | 65.7   | 62.1   | 64.2   |
| DeepSeek | 50% | 8.7        | 21.9     | 25.6     | 12.0     | 10.6     | 33.4   | 38.3   | 5.3    |

相对 WikiText 的 Δ（百分点）：

| Model    | Sp  | Wanda C4 | Wanda MM | Wanda CA | ENP C4 | ENP MM | ENP CA |
| -------- | --- | -------- | -------- | -------- | ------ | ------ | ------ |
| Qwen3    | 25% | +0.5     | −1.1     | +0.8     | −0.4   | +1.0   | +1.1   |
| Qwen3    | 50% | −4.1     | +4.5     | +10.3    | −3.1   | +10.5  | +12.9  |
| Qwen3.6  | 25% | −0.2     | +0.2     | −0.2     | −0.2   | +1.4   | +0.3   |
| Qwen3.6  | 50% | +7.4     | +14.1    | +12.5    | −6.2   | +8.6   | +11.9  |
| DeepSeek | 25% | +1.7     | +4.6     | +4.0     | +10.0  | +6.4   | +8.5   |
| DeepSeek | 50% | +13.2    | +16.9    | +3.3     | +22.8  | +27.7  | −5.3   |

#### 同构 Wanda / 同构 ENP

| Model    | Sp  | Wanda Wiki | Wanda C4 | Wanda MM | Wanda CA | ENP Wiki | ENP C4 | ENP MM | ENP CA |
| -------- | --- | ---------- | -------- | -------- | -------- | -------- | ------ | ------ | ------ |
| Qwen3    | 25% | 88.9       | 88.5     | 85.8     | 91.6     | 89.6     | 86.2   | 89.4   | 92.1   |
| Qwen3    | 50% | 53.0       | 49.9     | 63.1     | 66.2     | 49.2     | 43.0   | 64.7   | 66.4   |
| Qwen3.6  | 25% | 90.9       | 92.0     | 93.3     | 91.6     | 91.3     | 90.6   | 91.6   | 93.7   |
| Qwen3.6  | 50% | 54.0       | 52.8     | 62.6     | 60.5     | 57.5     | 52.0   | 68.8   | 64.2   |
| DeepSeek | 25% | 10.1       | 19.9     | 15.8     | 14.1     | 56.9     | 62.3   | 60.7   | 61.4   |
| DeepSeek | 50% | 18.0       | 14.7     | 14.5     | 6.1      | 11.3     | 31.7   | 36.8   | 3.9    |

同构同样随校准换方向：Qwen3 50% Wanda C4 49.9 vs MetaMath 63.1 vs Code-Alpaca 66.2；DeepSeek 50% ENP 在 MetaMath 36.8、在 Code-Alpaca 掉到 3.9。HARP-act 上同一件事更硬：Qwen3.6 50% Wanda 换 C4 +7.4、ENP −6.2；Qwen3 50% Wanda 在 C4 −4.1、MetaMath +4.5、Code-Alpaca +10.3。DeepSeek 50% ENP 的 HARP-act 从 Wiki 10.6 到 MetaMath 38.3，Code-Alpaca 只有 5.3。

#### 任务级别：不是整体平移

| 设置                     | 任务        | Wiki | C4   | MetaMath | Code-Alpaca |
| ---------------------- | --------- | ---- | ---- | -------- | ----------- |
| Qwen3 50% HARP-wanda   | GSM8K     | 80.2 | 69.8 | 94.8     | 75.9        |
| Qwen3 50% HARP-wanda   | MBPP      | 40.9 | 18.0 | 35.3     | 72.8        |
| Qwen3.6 50% HARP-wanda | MATH-500  | 55.7 | 48.1 | 79.7     | 64.6        |
| Qwen3.6 50% HARP-enp   | HumanEval | 31.8 | 36.3 | 52.9     | 78.3        |
| Qwen3.6 50% HARP-enp   | MBPP      | 53.5 | 21.0 | 60.5     | 76.0        |
| DeepSeek 50% HARP-enp  | mean      | 10.6 | 33.4 | 38.3     | 5.3         |

Code-Alpaca 把 Qwen 的代码任务拉起来，却把 DeepSeek 50% ENP 打崩。MetaMath 抬 GSM8K / MATH，C4 不一定。这就是校准过拟合 / 域偏移。

### E.2 HARP 异构相对同构 Wanda / ENP

每个校准一张均值表。稳定压过同构的是 **cfuse**（24/24 格 cfuse > 同构）。HARP-act 在 Qwen 上已经够用；DeepSeek 上只换宽度、通道仍纯激活不够（Wiki 50% Wanda 8.7 < 同构 18.0；C4 25% Wanda 19.2 与同构 19.9 打平偏弱）。

#### WikiText

| Model    | Sp  | 同构 Wanda | HARP-wanda | HARP-wanda-cfuse | 同构 ENP | HARP-enp | HARP-enp-cfuse |
| -------- | --- | -------- | ---------- | ---------------- | ------ | -------- | -------------- |
| Qwen3    | 25% | 88.9     | 93.0       | 94.4             | 89.6   | 92.7     | 93.5           |
| Qwen3    | 50% | 53.0     | 63.1       | 71.3             | 49.2   | 60.0     | 66.4           |
| Qwen3.6  | 25% | 90.9     | 96.3       | 96.5             | 91.3   | 96.1     | 96.2           |
| Qwen3.6  | 50% | 54.0     | 68.4       | 79.7             | 57.5   | 72.9     | 83.6           |
| DeepSeek | 25% | 10.1     | 17.5       | 81.5             | 56.9   | 55.7     | 69.3           |
| DeepSeek | 50% | 18.0     | 8.7        | 32.2             | 11.3   | 10.6     | 35.5           |

| Model    | Sp  | HARP-wanda − 同构 | cfuse − 同构 Wanda | HARP-enp − 同构 | cfuse − 同构 ENP |
| -------- | --- | --------------- | ---------------- | ------------- | -------------- |
| Qwen3    | 25% | +4.1            | +5.5             | +3.1          | +3.9           |
| Qwen3    | 50% | +10.1           | +18.3            | +10.8         | +17.2          |
| Qwen3.6  | 25% | +5.4            | +5.6             | +4.8          | +4.9           |
| Qwen3.6  | 50% | +14.4           | +25.7            | +15.4         | +26.1          |
| DeepSeek | 25% | +7.4            | +71.4            | −1.2          | +12.4          |
| DeepSeek | 50% | −9.3            | +14.2            | −0.7          | +24.2          |

#### C4

| Model    | Sp  | 同构 Wanda | HARP-wanda | HARP-wanda-cfuse | 同构 ENP | HARP-enp | HARP-enp-cfuse |
| -------- | --- | -------- | ---------- | ---------------- | ------ | -------- | -------------- |
| Qwen3    | 25% | 88.5     | 93.5       | 94.2             | 86.2   | 92.3     | 94.4           |
| Qwen3    | 50% | 49.9     | 59.0       | 71.4             | 43.0   | 56.9     | 65.5           |
| Qwen3.6  | 25% | 92.0     | 96.1       | 96.3             | 90.6   | 95.9     | 96.6           |
| Qwen3.6  | 50% | 52.8     | 75.8       | 82.0             | 52.0   | 66.7     | 77.1           |
| DeepSeek | 25% | 19.9     | 19.2       | 79.8             | 62.3   | 65.7     | 69.0           |
| DeepSeek | 50% | 14.7     | 21.9       | 35.3             | 31.7   | 33.4     | 37.1           |

| Model    | Sp  | HARP-wanda − 同构 | cfuse − 同构 Wanda | HARP-enp − 同构 | cfuse − 同构 ENP |
| -------- | --- | --------------- | ---------------- | ------------- | -------------- |
| Qwen3    | 25% | +5.0            | +5.7             | +6.1          | +8.2           |
| Qwen3    | 50% | +9.1            | +21.5            | +13.9         | +22.5          |
| Qwen3.6  | 25% | +4.1            | +4.3             | +5.3          | +6.0           |
| Qwen3.6  | 50% | +23.0           | +29.2            | +14.7         | +25.1          |
| DeepSeek | 25% | −0.7            | +59.9            | +3.4          | +6.7           |
| DeepSeek | 50% | +7.2            | +20.6            | +1.7          | +5.4           |

#### MetaMath

| Model    | Sp  | 同构 Wanda | HARP-wanda | HARP-wanda-cfuse | 同构 ENP | HARP-enp | HARP-enp-cfuse |
| -------- | --- | -------- | ---------- | ---------------- | ------ | -------- | -------------- |
| Qwen3    | 25% | 85.8     | 91.9       | 95.3             | 89.4   | 93.7     | 94.4           |
| Qwen3    | 50% | 63.1     | 67.6       | 76.7             | 64.7   | 70.5     | 74.7           |
| Qwen3.6  | 25% | 93.3     | 96.5       | 96.4             | 91.6   | 97.5     | 97.9           |
| Qwen3.6  | 50% | 62.6     | 82.5       | 85.0             | 68.8   | 81.5     | 86.5           |
| DeepSeek | 25% | 15.8     | 22.1       | 78.6             | 60.7   | 62.1     | 68.6           |
| DeepSeek | 50% | 14.5     | 25.6       | 39.9             | 36.8   | 38.3     | 39.4           |

| Model    | Sp  | HARP-wanda − 同构 | cfuse − 同构 Wanda | HARP-enp − 同构 | cfuse − 同构 ENP |
| -------- | --- | --------------- | ---------------- | ------------- | -------------- |
| Qwen3    | 25% | +6.1            | +9.5             | +4.3          | +5.0           |
| Qwen3    | 50% | +4.5            | +13.6            | +5.8          | +10.0          |
| Qwen3.6  | 25% | +3.2            | +3.1             | +5.9          | +6.3           |
| Qwen3.6  | 50% | +19.9           | +22.4            | +12.7         | +17.7          |
| DeepSeek | 25% | +6.3            | +62.8            | +1.4          | +7.9           |
| DeepSeek | 50% | +11.1           | +25.4            | +1.5          | +2.6           |

#### Code-Alpaca

| Model    | Sp  | 同构 Wanda | HARP-wanda | HARP-wanda-cfuse | 同构 ENP | HARP-enp | HARP-enp-cfuse |
| -------- | --- | -------- | ---------- | ---------------- | ------ | -------- | -------------- |
| Qwen3    | 25% | 91.6     | 93.8       | 94.2             | 92.1   | 93.8     | 94.1           |
| Qwen3    | 50% | 66.2     | 73.4       | 76.2             | 66.4   | 72.9     | 73.9           |
| Qwen3.6  | 25% | 91.6     | 96.1       | 96.6             | 93.7   | 96.4     | 96.8           |
| Qwen3.6  | 50% | 60.5     | 80.9       | 85.4             | 64.2   | 84.8     | 87.4           |
| DeepSeek | 25% | 14.1     | 21.5       | 78.7             | 61.4   | 64.2     | 66.2           |
| DeepSeek | 50% | 6.1      | 12.0       | 35.0             | 3.9    | 5.3      | 35.0           |

| Model    | Sp  | HARP-wanda − 同构 | cfuse − 同构 Wanda | HARP-enp − 同构 | cfuse − 同构 ENP |
| -------- | --- | --------------- | ---------------- | ------------- | -------------- |
| Qwen3    | 25% | +2.2            | +2.6             | +1.7          | +2.0           |
| Qwen3    | 50% | +7.2            | +10.0            | +6.5          | +7.5           |
| Qwen3.6  | 25% | +4.5            | +5.0             | +2.7          | +3.1           |
| Qwen3.6  | 50% | +20.4           | +24.9            | +20.6         | +23.2          |
| DeepSeek | 25% | +7.4            | +64.6            | +2.8          | +4.8           |
| DeepSeek | 50% | +5.9            | +28.9            | +1.4          | +31.1          |

### E.3 同一 HARP 下：cfuse 相对仅 Wanda/ENP 通道

Δ = cfuse − HARP-act。24 格里 23 格 ≥ 0。唯一例外是 Qwen3.6 25% MetaMath Wanda **−0.1**（96.4 vs 96.5），25% 饱和噪声，不翻论点。

| Model    | Sp  | Wiki W | C4 W  | MM W  | CA W  | Wiki E | C4 E  | MM E | CA E  |
| -------- | --- | ------ | ----- | ----- | ----- | ------ | ----- | ---- | ----- |
| Qwen3    | 25% | +1.4   | +0.7  | +3.4  | +0.4  | +0.8   | +2.1  | +0.7 | +0.3  |
| Qwen3    | 50% | +8.2   | +12.4 | +9.1  | +2.8  | +6.4   | +8.6  | +4.2 | +1.0  |
| Qwen3.6  | 25% | +0.2   | +0.2  | −0.1  | +0.5  | +0.1   | +0.7  | +0.4 | +0.4  |
| Qwen3.6  | 50% | +11.3  | +6.2  | +2.5  | +4.5  | +10.7  | +10.4 | +5.0 | +2.6  |
| DeepSeek | 25% | +64.0  | +60.6 | +56.5 | +57.2 | +13.6  | +3.3  | +6.5 | +2.0  |
| DeepSeek | 50% | +23.5  | +13.4 | +14.3 | +23.0 | +24.9  | +3.7  | +1.1 | +29.7 |

50% 和 DeepSeek 上 Channel-SP 把校准崩点托住：Wiki / C4 / MetaMath / Code-Alpaca 的 DeepSeek 25% Wanda 分别是 17.5→81.5、19.2→79.8、22.1→78.6、21.5→78.7。Code-Alpaca 上 DeepSeek 50% ENP 从 HARP-act 5.3 拉到 cfuse 35.0。增益小的格子是 HARP-act 已经接近饱和，不是反例。

### E.4 C4 HARP 明细

供核对原始分。WikiText HARP 明细在 A / B。

#### Wanda · Qwen3 25%

| Method                     | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| -------------------------- | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense                      | 0.9735 | 0.8709    | 0.7537     | 0.9492 | 0.8920   | 0.8552 | 0.9512    | 0.7880 | —               |
| HARP-nologbest-wanda       | 0.9568 | 0.8412    | 0.7040     | 0.9181 | 0.8100   | 0.7820 | 0.8720    | 0.7000 | 93.5            |
| HARP-nologbest-wanda-cfuse | 0.9569 | 0.8424    | 0.7096     | 0.9325 | 0.8180   | 0.7886 | 0.8841    | 0.7020 | 94.2            |

| Method                     | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| -------------------------- | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| HARP-nologbest-wanda       | 98.3 | 96.6      | 93.4       | 96.7  | 90.8     | 91.4 | 91.7      | 88.8 | 93.5 |
| HARP-nologbest-wanda-cfuse | 98.3 | 96.7      | 94.1       | 98.2  | 91.7     | 92.2 | 92.9      | 89.1 | 94.2 |

#### ENP · Qwen3 25%

| Method                   | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| ------------------------ | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense                    | 0.9735 | 0.8709    | 0.7537     | 0.9492 | 0.8920   | 0.8552 | 0.9512    | 0.7880 | —               |
| HARP-nologbest-enp       | 0.9479 | 0.8418    | 0.7206     | 0.9249 | 0.7680   | 0.7603 | 0.8476    | 0.6860 | 92.3            |
| HARP-nologbest-enp-cfuse | 0.9527 | 0.8493    | 0.7167     | 0.9371 | 0.8220   | 0.7855 | 0.8659    | 0.7180 | 94.4            |

| Method                   | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| ------------------------ | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| HARP-nologbest-enp       | 97.4 | 96.7      | 95.6       | 97.4  | 86.1     | 88.9 | 89.1      | 87.1 | 92.3 |
| HARP-nologbest-enp-cfuse | 97.9 | 97.5      | 95.1       | 98.7  | 92.2     | 91.8 | 91.0      | 91.1 | 94.4 |

#### Wanda · Qwen3 50%

| Method                     | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| -------------------------- | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense                      | 0.9735 | 0.8709    | 0.7537     | 0.9492 | 0.8920   | 0.8552 | 0.9512    | 0.7880 | —               |
| HARP-nologbest-wanda       | 0.8419 | 0.7369    | 0.5706     | 0.6626 | 0.3700   | 0.5911 | 0.2561    | 0.1420 | 59.0            |
| HARP-nologbest-wanda-cfuse | 0.8582 | 0.7060    | 0.5770     | 0.7726 | 0.4720   | 0.5991 | 0.5732    | 0.4800 | 71.4            |

| Method                     | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| -------------------------- | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| HARP-nologbest-wanda       | 86.5 | 84.6      | 75.7       | 69.8  | 41.5     | 69.1 | 26.9      | 18.0 | 59.0 |
| HARP-nologbest-wanda-cfuse | 88.2 | 81.1      | 76.6       | 81.4  | 52.9     | 70.1 | 60.3      | 60.9 | 71.4 |

#### ENP · Qwen3 50%

| Method                   | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| ------------------------ | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense                    | 0.9735 | 0.8709    | 0.7537     | 0.9492 | 0.8920   | 0.8552 | 0.9512    | 0.7880 | —               |
| HARP-nologbest-enp       | 0.7869 | 0.7196    | 0.5730     | 0.6566 | 0.2300   | 0.5212 | 0.2134    | 0.2940 | 56.9            |
| HARP-nologbest-enp-cfuse | 0.8334 | 0.7289    | 0.5856     | 0.7680 | 0.3900   | 0.5750 | 0.3659    | 0.3700 | 65.5            |

| Method                   | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| ------------------------ | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| HARP-nologbest-enp       | 80.8 | 82.6      | 76.0       | 69.2  | 25.8     | 60.9 | 22.4      | 37.3 | 56.9 |
| HARP-nologbest-enp-cfuse | 85.6 | 83.7      | 77.7       | 80.9  | 43.7     | 67.2 | 38.5      | 47.0 | 65.5 |

#### Wanda · Qwen3.6 25%

| Method                     | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| -------------------------- | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense                      | 0.9831 | 0.8986    | 0.8713     | 0.9651 | 0.9480   | 0.8909 | 0.9573    | 0.8000 | —               |
| HARP-nologbest-wanda       | 0.9738 | 0.8988    | 0.7893     | 0.9530 | 0.9260   | 0.8449 | 0.9146    | 0.7380 | 96.1            |
| HARP-nologbest-wanda-cfuse | 0.9747 | 0.8949    | 0.7987     | 0.9545 | 0.9340   | 0.8485 | 0.9146    | 0.7340 | 96.3            |

| Method                     | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| -------------------------- | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| HARP-nologbest-wanda       | 99.1 | 100.0     | 90.6       | 98.7  | 97.7     | 94.8 | 95.5      | 92.2 | 96.1 |
| HARP-nologbest-wanda-cfuse | 99.1 | 99.6      | 91.7       | 98.9  | 98.5     | 95.2 | 95.5      | 91.8 | 96.3 |

#### ENP · Qwen3.6 25%

| Method                   | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| ------------------------ | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense                    | 0.9831 | 0.8986    | 0.8713     | 0.9651 | 0.9480   | 0.8909 | 0.9573    | 0.8000 | —               |
| HARP-nologbest-enp       | 0.9729 | 0.9066    | 0.7877     | 0.9522 | 0.9080   | 0.8445 | 0.9085    | 0.7400 | 95.9            |
| HARP-nologbest-enp-cfuse | 0.9743 | 0.9076    | 0.7972     | 0.9538 | 0.9100   | 0.8475 | 0.9146    | 0.7640 | 96.6            |

| Method                   | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| ------------------------ | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| HARP-nologbest-enp       | 99.0 | 100.9     | 90.4       | 98.7  | 95.8     | 94.8 | 94.9      | 92.5 | 95.9 |
| HARP-nologbest-enp-cfuse | 99.1 | 101.0     | 91.5       | 98.8  | 96.0     | 95.1 | 95.5      | 95.5 | 96.6 |

#### Wanda · Qwen3.6 50%

| Method                     | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| -------------------------- | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense                      | 0.9831 | 0.8986    | 0.8713     | 0.9651 | 0.9480   | 0.8909 | 0.9573    | 0.8000 | —               |
| HARP-nologbest-wanda       | 0.9403 | 0.8426    | 0.7198     | 0.6717 | 0.4560   | 0.6701 | 0.7378    | 0.5180 | 75.8            |
| HARP-nologbest-wanda-cfuse | 0.9462 | 0.8114    | 0.7443     | 0.8840 | 0.7260   | 0.7171 | 0.7195    | 0.4840 | 82.0            |

| Method                     | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| -------------------------- | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| HARP-nologbest-wanda       | 95.6 | 93.8      | 82.6       | 69.6  | 48.1     | 75.2 | 77.1      | 64.8 | 75.8 |
| HARP-nologbest-wanda-cfuse | 96.2 | 90.3      | 85.4       | 91.6  | 76.6     | 80.5 | 75.2      | 60.5 | 82.0 |

#### ENP · Qwen3.6 50%

| Method                   | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| ------------------------ | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense                    | 0.9831 | 0.8986    | 0.8713     | 0.9651 | 0.9480   | 0.8909 | 0.9573    | 0.8000 | —               |
| HARP-nologbest-enp       | 0.9341 | 0.8426    | 0.7316     | 0.8605 | 0.4620   | 0.5837 | 0.3476    | 0.1680 | 66.7            |
| HARP-nologbest-enp-cfuse | 0.9470 | 0.8453    | 0.7332     | 0.8923 | 0.6620   | 0.7091 | 0.6280    | 0.2760 | 77.1            |

| Method                   | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| ------------------------ | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| HARP-nologbest-enp       | 95.0 | 93.8      | 84.0       | 89.2  | 48.7     | 65.5 | 36.3      | 21.0 | 66.7 |
| HARP-nologbest-enp-cfuse | 96.3 | 94.1      | 84.2       | 92.5  | 69.8     | 79.6 | 65.6      | 34.5 | 77.1 |

#### Wanda · DeepSeek 25%

| Method                     | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| -------------------------- | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense                      | 0.7925 | 0.6622    | 0.5549     | 0.7058 | 0.2880   | 0.5613 | 0.5183    | 0.4680 | —               |
| HARP-nologbest-wanda       | 0.1764 | 0.2112    | 0.1571     | 0.0538 | 0.0160   | 0.2638 | 0.0000    | 0.0500 | 19.2            |
| HARP-nologbest-wanda-cfuse | 0.7164 | 0.6051    | 0.5525     | 0.5595 | 0.1640   | 0.5072 | 0.3232    | 0.3200 | 79.8            |

| Method                     | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| -------------------------- | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| HARP-nologbest-wanda       | 22.3 | 31.9      | 28.3       | 7.6   | 5.6      | 47.0 | 0.0       | 10.7 | 19.2 |
| HARP-nologbest-wanda-cfuse | 90.4 | 91.4      | 99.6       | 79.3  | 56.9     | 90.4 | 62.4      | 68.4 | 79.8 |

#### ENP · DeepSeek 25%

| Method                   | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| ------------------------ | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense                    | 0.7925 | 0.6622    | 0.5549     | 0.7058 | 0.2880   | 0.5613 | 0.5183    | 0.4680 | —               |
| HARP-nologbest-enp       | 0.6677 | 0.4809    | 0.5478     | 0.4215 | 0.0860   | 0.4851 | 0.2256    | 0.2380 | 65.7            |
| HARP-nologbest-enp-cfuse | 0.6669 | 0.5172    | 0.5485     | 0.4723 | 0.1240   | 0.4740 | 0.2439    | 0.2300 | 69.0            |

| Method                   | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| ------------------------ | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| HARP-nologbest-enp       | 84.3 | 72.6      | 98.7       | 59.7  | 29.9     | 86.4 | 43.5      | 50.9 | 65.7 |
| HARP-nologbest-enp-cfuse | 84.2 | 78.1      | 98.8       | 66.9  | 43.1     | 84.4 | 47.1      | 49.1 | 69.0 |

#### Wanda · DeepSeek 50%

| Method                     | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| -------------------------- | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense                      | 0.7925 | 0.6622    | 0.5549     | 0.7058 | 0.2880   | 0.5613 | 0.5183    | 0.4680 | —               |
| HARP-nologbest-wanda       | 0.2728 | 0.2089    | 0.3496     | 0.0106 | 0.0220   | 0.2038 | 0.0000    | 0.0040 | 21.9            |
| HARP-nologbest-wanda-cfuse | 0.3698 | 0.3301    | 0.4507     | 0.1259 | 0.0360   | 0.2377 | 0.0732    | 0.0840 | 35.3            |

| Method                     | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| -------------------------- | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| HARP-nologbest-wanda       | 34.4 | 31.5      | 63.0       | 1.5   | 7.6      | 36.3 | 0.0       | 0.9  | 21.9 |
| HARP-nologbest-wanda-cfuse | 46.7 | 49.8      | 81.2       | 17.8  | 12.5     | 42.3 | 14.1      | 17.9 | 35.3 |

#### ENP · DeepSeek 50%

| Method                   | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| ------------------------ | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense                    | 0.7925 | 0.6622    | 0.5549     | 0.7058 | 0.2880   | 0.5613 | 0.5183    | 0.4680 | —               |
| HARP-nologbest-enp       | 0.3712 | 0.2695    | 0.4957     | 0.0569 | 0.0220   | 0.3722 | 0.0000    | 0.0400 | 33.4            |
| HARP-nologbest-enp-cfuse | 0.3893 | 0.2924    | 0.5051     | 0.0834 | 0.0440   | 0.3509 | 0.0305    | 0.0780 | 37.1            |

| Method                   | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| ------------------------ | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| HARP-nologbest-enp       | 46.8 | 40.7      | 89.3       | 8.1   | 7.6      | 66.3 | 0.0       | 8.5  | 33.4 |
| HARP-nologbest-enp-cfuse | 49.1 | 44.2      | 91.0       | 11.8  | 15.3     | 62.5 | 5.9       | 16.7 | 37.1 |

### E.5 MetaMath HARP 明细

供核对原始分。WikiText HARP 明细在 A / B。

#### Wanda · Qwen3 25%

| Method                     | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| -------------------------- | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense                      | 0.9735 | 0.8709    | 0.7537     | 0.9492 | 0.8920   | 0.8552 | 0.9512    | 0.7880 | —               |
| HARP-nologbest-wanda       | 0.9532 | 0.8455    | 0.6906     | 0.9484 | 0.8860   | 0.7759 | 0.7012    | 0.6660 | 91.9            |
| HARP-nologbest-wanda-cfuse | 0.9591 | 0.8435    | 0.7143     | 0.9545 | 0.8820   | 0.7811 | 0.8780    | 0.7000 | 95.3            |

| Method                     | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| -------------------------- | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| HARP-nologbest-wanda       | 97.9 | 97.1      | 91.6       | 99.9  | 99.3     | 90.7 | 73.7      | 84.5 | 91.9 |
| HARP-nologbest-wanda-cfuse | 98.5 | 96.9      | 94.8       | 100.6 | 98.9     | 91.3 | 92.3      | 88.8 | 95.3 |

#### ENP · Qwen3 25%

| Method                   | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| ------------------------ | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense                    | 0.9735 | 0.8709    | 0.7537     | 0.9492 | 0.8920   | 0.8552 | 0.9512    | 0.7880 | —               |
| HARP-nologbest-enp       | 0.9543 | 0.8530    | 0.7143     | 0.9507 | 0.8640   | 0.7601 | 0.8232    | 0.6840 | 93.7            |
| HARP-nologbest-enp-cfuse | 0.9543 | 0.8488    | 0.7190     | 0.9469 | 0.8800   | 0.7712 | 0.8354    | 0.6900 | 94.4            |

| Method                   | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| ------------------------ | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| HARP-nologbest-enp       | 98.0 | 97.9      | 94.8       | 100.2 | 96.9     | 88.9 | 86.5      | 86.8 | 93.7 |
| HARP-nologbest-enp-cfuse | 98.0 | 97.5      | 95.4       | 99.8  | 98.7     | 90.2 | 87.8      | 87.6 | 94.4 |

#### Wanda · Qwen3 50%

| Method                     | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| -------------------------- | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense                      | 0.9735 | 0.8709    | 0.7537     | 0.9492 | 0.8920   | 0.8552 | 0.9512    | 0.7880 | —               |
| HARP-nologbest-wanda       | 0.8354 | 0.7135    | 0.5967     | 0.8999 | 0.6740   | 0.5863 | 0.1890    | 0.2780 | 67.6            |
| HARP-nologbest-wanda-cfuse | 0.8512 | 0.7117    | 0.6148     | 0.9143 | 0.6920   | 0.6004 | 0.5122    | 0.5120 | 76.7            |

| Method                     | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| -------------------------- | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| HARP-nologbest-wanda       | 85.8 | 81.9      | 79.2       | 94.8  | 75.6     | 68.6 | 19.9      | 35.3 | 67.6 |
| HARP-nologbest-wanda-cfuse | 87.4 | 81.7      | 81.6       | 96.3  | 77.6     | 70.2 | 53.8      | 65.0 | 76.7 |

#### ENP · Qwen3 50%

| Method                   | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| ------------------------ | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense                    | 0.9735 | 0.8709    | 0.7537     | 0.9492 | 0.8920   | 0.8552 | 0.9512    | 0.7880 | —               |
| HARP-nologbest-enp       | 0.8323 | 0.6995    | 0.6243     | 0.9234 | 0.7000   | 0.5162 | 0.3171    | 0.3620 | 70.5            |
| HARP-nologbest-enp-cfuse | 0.8481 | 0.7060    | 0.6259     | 0.9174 | 0.6940   | 0.5790 | 0.4512    | 0.4460 | 74.7            |

| Method                   | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| ------------------------ | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| HARP-nologbest-enp       | 85.5 | 80.3      | 82.8       | 97.3  | 78.5     | 60.4 | 33.3      | 45.9 | 70.5 |
| HARP-nologbest-enp-cfuse | 87.1 | 81.1      | 83.0       | 96.6  | 77.8     | 67.7 | 47.4      | 56.6 | 74.7 |

#### Wanda · Qwen3.6 25%

| Method                     | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| -------------------------- | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense                      | 0.9831 | 0.8986    | 0.8713     | 0.9651 | 0.9480   | 0.8909 | 0.9573    | 0.8000 | —               |
| HARP-nologbest-wanda       | 0.9712 | 0.8949    | 0.7932     | 0.9568 | 0.9300   | 0.8468 | 0.9390    | 0.7400 | 96.5            |
| HARP-nologbest-wanda-cfuse | 0.9729 | 0.8904    | 0.8090     | 0.9598 | 0.9420   | 0.8478 | 0.9085    | 0.7340 | 96.4            |

| Method                     | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| -------------------------- | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| HARP-nologbest-wanda       | 98.8 | 99.6      | 91.0       | 99.1  | 98.1     | 95.0 | 98.1      | 92.5 | 96.5 |
| HARP-nologbest-wanda-cfuse | 99.0 | 99.1      | 92.8       | 99.5  | 99.4     | 95.2 | 94.9      | 91.8 | 96.4 |

#### ENP · Qwen3.6 25%

| Method                   | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| ------------------------ | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense                    | 0.9831 | 0.8986    | 0.8713     | 0.9651 | 0.9480   | 0.8909 | 0.9573    | 0.8000 | —               |
| HARP-nologbest-enp       | 0.9795 | 0.9077    | 0.8635     | 0.9606 | 0.9420   | 0.8495 | 0.9024    | 0.7360 | 97.5            |
| HARP-nologbest-enp-cfuse | 0.9769 | 0.8960    | 0.8619     | 0.9583 | 0.9420   | 0.8533 | 0.9390    | 0.7440 | 97.9            |

| Method                   | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| ------------------------ | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| HARP-nologbest-enp       | 99.6 | 101.0     | 99.1       | 99.5  | 99.4     | 95.4 | 94.3      | 92.0 | 97.5 |
| HARP-nologbest-enp-cfuse | 99.4 | 99.7      | 98.9       | 99.3  | 99.4     | 95.8 | 98.1      | 93.0 | 97.9 |

#### Wanda · Qwen3.6 50%

| Method                     | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| -------------------------- | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense                      | 0.9831 | 0.8986    | 0.8713     | 0.9651 | 0.9480   | 0.8909 | 0.9573    | 0.8000 | —               |
| HARP-nologbest-wanda       | 0.9346 | 0.8205    | 0.7309     | 0.9196 | 0.7560   | 0.6909 | 0.6646    | 0.5440 | 82.5            |
| HARP-nologbest-wanda-cfuse | 0.9456 | 0.8352    | 0.7293     | 0.9318 | 0.8040   | 0.7229 | 0.7256    | 0.5480 | 85.0            |

| Method                     | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| -------------------------- | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| HARP-nologbest-wanda       | 95.1 | 91.3      | 83.9       | 95.3  | 79.7     | 77.6 | 69.4      | 68.0 | 82.5 |
| HARP-nologbest-wanda-cfuse | 96.2 | 92.9      | 83.7       | 96.5  | 84.8     | 81.1 | 75.8      | 68.5 | 85.0 |

#### ENP · Qwen3.6 50%

| Method                   | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| ------------------------ | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense                    | 0.9831 | 0.8986    | 0.8713     | 0.9651 | 0.9480   | 0.8909 | 0.9573    | 0.8000 | —               |
| HARP-nologbest-enp       | 0.9408 | 0.8387    | 0.7419     | 0.9538 | 0.8260   | 0.7002 | 0.5061    | 0.4840 | 81.5            |
| HARP-nologbest-enp-cfuse | 0.9512 | 0.8390    | 0.7514     | 0.9484 | 0.8460   | 0.7269 | 0.7134    | 0.5740 | 86.5            |

| Method                   | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| ------------------------ | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| HARP-nologbest-enp       | 95.7 | 93.3      | 85.1       | 98.8  | 87.1     | 78.6 | 52.9      | 60.5 | 81.5 |
| HARP-nologbest-enp-cfuse | 96.8 | 93.4      | 86.2       | 98.3  | 89.2     | 81.6 | 74.5      | 71.8 | 86.5 |

#### Wanda · DeepSeek 25%

| Method                     | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| -------------------------- | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense                      | 0.7925 | 0.6622    | 0.5549     | 0.7058 | 0.2880   | 0.5613 | 0.5183    | 0.4680 | —               |
| HARP-nologbest-wanda       | 0.2001 | 0.2018    | 0.2510     | 0.0516 | 0.0240   | 0.2809 | 0.0000    | 0.0460 | 22.1            |
| HARP-nologbest-wanda-cfuse | 0.7286 | 0.6156    | 0.5659     | 0.5481 | 0.1600   | 0.5104 | 0.2805    | 0.2980 | 78.6            |

| Method                     | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| -------------------------- | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| HARP-nologbest-wanda       | 25.2 | 30.5      | 45.2       | 7.3   | 8.3      | 50.0 | 0.0       | 9.8  | 22.1 |
| HARP-nologbest-wanda-cfuse | 91.9 | 93.0      | 102.0      | 77.7  | 55.6     | 90.9 | 54.1      | 63.7 | 78.6 |

#### ENP · DeepSeek 25%

| Method                   | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| ------------------------ | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense                    | 0.7925 | 0.6622    | 0.5549     | 0.7058 | 0.2880   | 0.5613 | 0.5183    | 0.4680 | —               |
| HARP-nologbest-enp       | 0.6573 | 0.4746    | 0.5296     | 0.3844 | 0.0960   | 0.4704 | 0.1646    | 0.2020 | 62.1            |
| HARP-nologbest-enp-cfuse | 0.6930 | 0.5075    | 0.5359     | 0.4860 | 0.1120   | 0.4788 | 0.2195    | 0.2480 | 68.6            |

| Method                   | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| ------------------------ | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| HARP-nologbest-enp       | 82.9 | 71.7      | 95.4       | 54.5  | 33.3     | 83.8 | 31.8      | 43.2 | 62.1 |
| HARP-nologbest-enp-cfuse | 87.4 | 76.6      | 96.6       | 68.9  | 38.9     | 85.3 | 42.3      | 53.0 | 68.6 |

#### Wanda · DeepSeek 50%

| Method                     | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| -------------------------- | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense                      | 0.7925 | 0.6622    | 0.5549     | 0.7058 | 0.2880   | 0.5613 | 0.5183    | 0.4680 | —               |
| HARP-nologbest-wanda       | 0.3593 | 0.2107    | 0.4522     | 0.0114 | 0.0160   | 0.2104 | 0.0000    | 0.0080 | 25.6            |
| HARP-nologbest-wanda-cfuse | 0.4833 | 0.3231    | 0.5351     | 0.0758 | 0.0260   | 0.3487 | 0.0732    | 0.0780 | 39.9            |

| Method                     | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| -------------------------- | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| HARP-nologbest-wanda       | 45.3 | 31.8      | 81.5       | 1.6   | 5.6      | 37.5 | 0.0       | 1.7  | 25.6 |
| HARP-nologbest-wanda-cfuse | 61.0 | 48.8      | 96.4       | 10.7  | 9.0      | 62.1 | 14.1      | 16.7 | 39.9 |

#### ENP · DeepSeek 50%

| Method                   | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| ------------------------ | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense                    | 0.7925 | 0.6622    | 0.5549     | 0.7058 | 0.2880   | 0.5613 | 0.5183    | 0.4680 | —               |
| HARP-nologbest-enp       | 0.5313 | 0.3606    | 0.5146     | 0.0599 | 0.0240   | 0.3549 | 0.0061    | 0.0500 | 38.3            |
| HARP-nologbest-enp-cfuse | 0.5067 | 0.3528    | 0.5099     | 0.1008 | 0.0360   | 0.3497 | 0.0366    | 0.0480 | 39.4            |

| Method                   | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| ------------------------ | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| HARP-nologbest-enp       | 67.0 | 54.5      | 92.7       | 8.5   | 8.3      | 63.2 | 1.2       | 10.7 | 38.3 |
| HARP-nologbest-enp-cfuse | 63.9 | 53.3      | 91.9       | 14.3  | 12.5     | 62.3 | 7.1       | 10.3 | 39.4 |

### E.6 Code-Alpaca HARP 明细

供核对原始分。WikiText HARP 明细在 A / B。

#### Wanda · Qwen3 25%

| Method                     | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| -------------------------- | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense                      | 0.9735 | 0.8709    | 0.7537     | 0.9492 | 0.8920   | 0.8552 | 0.9512    | 0.7880 | —               |
| HARP-nologbest-wanda       | 0.9571 | 0.8376    | 0.7127     | 0.9234 | 0.8140   | 0.7764 | 0.8598    | 0.7220 | 93.8            |
| HARP-nologbest-wanda-cfuse | 0.9563 | 0.8436    | 0.7096     | 0.9348 | 0.8400   | 0.7833 | 0.8598    | 0.7100 | 94.2            |

| Method                     | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| -------------------------- | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| HARP-nologbest-wanda       | 98.3 | 96.2      | 94.6       | 97.3  | 91.3     | 90.8 | 90.4      | 91.6 | 93.8 |
| HARP-nologbest-wanda-cfuse | 98.2 | 96.9      | 94.1       | 98.5  | 94.2     | 91.6 | 90.4      | 90.1 | 94.2 |

#### ENP · Qwen3 25%

| Method                   | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| ------------------------ | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense                    | 0.9735 | 0.8709    | 0.7537     | 0.9492 | 0.8920   | 0.8552 | 0.9512    | 0.7880 | —               |
| HARP-nologbest-enp       | 0.9518 | 0.8262    | 0.6946     | 0.9340 | 0.8300   | 0.7620 | 0.8841    | 0.7280 | 93.8            |
| HARP-nologbest-enp-cfuse | 0.9603 | 0.8334    | 0.6882     | 0.9462 | 0.8620   | 0.7756 | 0.8537    | 0.7120 | 94.1            |

| Method                   | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| ------------------------ | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| HARP-nologbest-enp       | 97.8 | 94.9      | 92.2       | 98.4  | 93.0     | 89.1 | 92.9      | 92.4 | 93.8 |
| HARP-nologbest-enp-cfuse | 98.6 | 95.7      | 91.3       | 99.7  | 96.6     | 90.7 | 89.7      | 90.4 | 94.1 |

#### Wanda · Qwen3 50%

| Method                     | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| -------------------------- | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense                      | 0.9735 | 0.8709    | 0.7537     | 0.9492 | 0.8920   | 0.8552 | 0.9512    | 0.7880 | —               |
| HARP-nologbest-wanda       | 0.8523 | 0.7136    | 0.5738     | 0.7202 | 0.4580   | 0.5835 | 0.6951    | 0.5740 | 73.4            |
| HARP-nologbest-wanda-cfuse | 0.8616 | 0.7019    | 0.6014     | 0.8120 | 0.5260   | 0.5976 | 0.7073    | 0.5660 | 76.2            |

| Method                     | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| -------------------------- | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| HARP-nologbest-wanda       | 87.6 | 81.9      | 76.1       | 75.9  | 51.3     | 68.2 | 73.1      | 72.8 | 73.4 |
| HARP-nologbest-wanda-cfuse | 88.5 | 80.6      | 79.8       | 85.5  | 59.0     | 69.9 | 74.4      | 71.8 | 76.2 |

#### ENP · Qwen3 50%

| Method                   | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| ------------------------ | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense                    | 0.9735 | 0.8709    | 0.7537     | 0.9492 | 0.8920   | 0.8552 | 0.9512    | 0.7880 | —               |
| HARP-nologbest-enp       | 0.8376 | 0.6672    | 0.5722     | 0.7953 | 0.4060   | 0.5428 | 0.7012    | 0.6160 | 72.9            |
| HARP-nologbest-enp-cfuse | 0.8557 | 0.6688    | 0.5848     | 0.8241 | 0.4880   | 0.5614 | 0.6524    | 0.5740 | 73.9            |

| Method                   | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| ------------------------ | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| HARP-nologbest-enp       | 86.0 | 76.6      | 75.9       | 83.8  | 45.5     | 63.5 | 73.7      | 78.2 | 72.9 |
| HARP-nologbest-enp-cfuse | 87.9 | 76.8      | 77.6       | 86.8  | 54.7     | 65.6 | 68.6      | 72.8 | 73.9 |

#### Wanda · Qwen3.6 25%

| Method                     | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| -------------------------- | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense                      | 0.9831 | 0.8986    | 0.8713     | 0.9651 | 0.9480   | 0.8909 | 0.9573    | 0.8000 | —               |
| HARP-nologbest-wanda       | 0.9706 | 0.8958    | 0.7995     | 0.9507 | 0.9240   | 0.8431 | 0.9146    | 0.7400 | 96.1            |
| HARP-nologbest-wanda-cfuse | 0.9744 | 0.8947    | 0.8122     | 0.9545 | 0.9300   | 0.8477 | 0.9146    | 0.7460 | 96.6            |

| Method                     | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| -------------------------- | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| HARP-nologbest-wanda       | 98.7 | 99.7      | 91.8       | 98.5  | 97.5     | 94.6 | 95.5      | 92.5 | 96.1 |
| HARP-nologbest-wanda-cfuse | 99.1 | 99.6      | 93.2       | 98.9  | 98.1     | 95.2 | 95.5      | 93.2 | 96.6 |

#### ENP · Qwen3.6 25%

| Method                   | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| ------------------------ | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense                    | 0.9831 | 0.8986    | 0.8713     | 0.9651 | 0.9480   | 0.8909 | 0.9573    | 0.8000 | —               |
| HARP-nologbest-enp       | 0.9735 | 0.9040    | 0.7972     | 0.9583 | 0.9260   | 0.8492 | 0.9146    | 0.7400 | 96.4            |
| HARP-nologbest-enp-cfuse | 0.9757 | 0.9035    | 0.8051     | 0.9613 | 0.9300   | 0.8500 | 0.9268    | 0.7380 | 96.8            |

| Method                   | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| ------------------------ | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| HARP-nologbest-enp       | 99.0 | 100.6     | 91.5       | 99.3  | 97.7     | 95.3 | 95.5      | 92.5 | 96.4 |
| HARP-nologbest-enp-cfuse | 99.2 | 100.5     | 92.4       | 99.6  | 98.1     | 95.4 | 96.8      | 92.2 | 96.8 |

#### Wanda · Qwen3.6 50%

| Method                     | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| -------------------------- | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense                      | 0.9831 | 0.8986    | 0.8713     | 0.9651 | 0.9480   | 0.8909 | 0.9573    | 0.8000 | —               |
| HARP-nologbest-wanda       | 0.9371 | 0.8288    | 0.7332     | 0.8089 | 0.6120   | 0.6697 | 0.7683    | 0.5760 | 80.9            |
| HARP-nologbest-wanda-cfuse | 0.9416 | 0.8374    | 0.7364     | 0.9204 | 0.7280   | 0.7100 | 0.7988    | 0.5960 | 85.4            |

| Method                     | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| -------------------------- | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| HARP-nologbest-wanda       | 95.3 | 92.2      | 84.2       | 83.8  | 64.6     | 75.2 | 80.3      | 72.0 | 80.9 |
| HARP-nologbest-wanda-cfuse | 95.8 | 93.2      | 84.5       | 95.4  | 76.8     | 79.7 | 83.4      | 74.5 | 85.4 |

#### ENP · Qwen3.6 50%

| Method                   | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| ------------------------ | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense                    | 0.9831 | 0.8986    | 0.8713     | 0.9651 | 0.9480   | 0.8909 | 0.9573    | 0.8000 | —               |
| HARP-nologbest-enp       | 0.9403 | 0.8356    | 0.7403     | 0.9090 | 0.7340   | 0.7037 | 0.7500    | 0.6080 | 84.8            |
| HARP-nologbest-enp-cfuse | 0.9479 | 0.8330    | 0.7451     | 0.9318 | 0.7840   | 0.7400 | 0.7927    | 0.6360 | 87.4            |

| Method                   | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| ------------------------ | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| HARP-nologbest-enp       | 95.6 | 93.0      | 85.0       | 94.2  | 77.4     | 79.0 | 78.3      | 76.0 | 84.8 |
| HARP-nologbest-enp-cfuse | 96.4 | 92.7      | 85.5       | 96.5  | 82.7     | 83.1 | 82.8      | 79.5 | 87.4 |

#### Wanda · DeepSeek 25%

| Method                     | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| -------------------------- | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense                      | 0.7925 | 0.6622    | 0.5549     | 0.7058 | 0.2880   | 0.5613 | 0.5183    | 0.4680 | —               |
| HARP-nologbest-wanda       | 0.2145 | 0.2057    | 0.2881     | 0.0379 | 0.0140   | 0.2455 | 0.0000    | 0.0360 | 21.5            |
| HARP-nologbest-wanda-cfuse | 0.7106 | 0.6040    | 0.5770     | 0.5466 | 0.1480   | 0.5171 | 0.3110    | 0.3000 | 78.7            |

| Method                     | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| -------------------------- | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| HARP-nologbest-wanda       | 27.1 | 31.1      | 51.9       | 5.4   | 4.9      | 43.7 | 0.0       | 7.7  | 21.5 |
| HARP-nologbest-wanda-cfuse | 89.7 | 91.2      | 104.0      | 77.4  | 51.4     | 92.1 | 60.0      | 64.1 | 78.7 |

#### ENP · DeepSeek 25%

| Method                   | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| ------------------------ | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense                    | 0.7925 | 0.6622    | 0.5549     | 0.7058 | 0.2880   | 0.5613 | 0.5183    | 0.4680 | —               |
| HARP-nologbest-enp       | 0.6657 | 0.5376    | 0.5359     | 0.3465 | 0.0840   | 0.4810 | 0.2012    | 0.2300 | 64.2            |
| HARP-nologbest-enp-cfuse | 0.6570 | 0.5058    | 0.5383     | 0.4708 | 0.1140   | 0.4677 | 0.1768    | 0.2320 | 66.2            |

| Method                   | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| ------------------------ | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| HARP-nologbest-enp       | 84.0 | 81.2      | 96.6       | 49.1  | 29.2     | 85.7 | 38.8      | 49.1 | 64.2 |
| HARP-nologbest-enp-cfuse | 82.9 | 76.4      | 97.0       | 66.7  | 39.6     | 83.3 | 34.1      | 49.6 | 66.2 |

#### Wanda · DeepSeek 50%

| Method                     | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| -------------------------- | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense                      | 0.7925 | 0.6622    | 0.5549     | 0.7058 | 0.2880   | 0.5613 | 0.5183    | 0.4680 | —               |
| HARP-nologbest-wanda       | 0.0910 | 0.0940    | 0.2186     | 0.0008 | 0.0040   | 0.1560 | 0.0000    | 0.0060 | 12.0            |
| HARP-nologbest-wanda-cfuse | 0.2768 | 0.2659    | 0.4941     | 0.1319 | 0.0380   | 0.3028 | 0.0488    | 0.0980 | 35.0            |

| Method                     | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| -------------------------- | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| HARP-nologbest-wanda       | 11.5 | 14.2      | 39.4       | 0.1   | 1.4      | 27.8 | 0.0       | 1.3  | 12.0 |
| HARP-nologbest-wanda-cfuse | 34.9 | 40.2      | 89.0       | 18.7  | 13.2     | 53.9 | 9.4       | 20.9 | 35.0 |

#### ENP · DeepSeek 50%

| Method                   | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| ------------------------ | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense                    | 0.7925 | 0.6622    | 0.5549     | 0.7058 | 0.2880   | 0.5613 | 0.5183    | 0.4680 | —               |
| HARP-nologbest-enp       | 0.0465 | 0.0796    | 0.0284     | 0.0000 | 0.0000   | 0.1072 | 0.0000    | 0.0000 | 5.3             |
| HARP-nologbest-enp-cfuse | 0.3360 | 0.2628    | 0.5288     | 0.0970 | 0.0340   | 0.3373 | 0.0183    | 0.0640 | 35.0            |

| Method                   | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| ------------------------ | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| HARP-nologbest-enp       | 5.9  | 12.0      | 5.1        | 0.0   | 0.0      | 19.1 | 0.0       | 0.0  | 5.3  |
| HARP-nologbest-enp-cfuse | 42.4 | 39.7      | 95.3       | 13.7  | 11.8     | 60.1 | 3.5       | 13.7 | 35.0 |

### E.7 同构 Wanda / ENP 明细（C4 / MetaMath / Code-Alpaca）

WikiText 同构在 `Results_HARP.md`。这里只写三个新校准。

#### C4

#### Qwen3 25%

| Method | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| ------ | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense  | 0.9735 | 0.8709    | 0.7537     | 0.9492 | 0.8920   | 0.8552 | 0.9512    | 0.7880 | —               |
| Wanda  | 0.9512 | 0.8340    | 0.6693     | 0.8552 | 0.7360   | 0.7852 | 0.7195    | 0.6740 | 88.5            |
| ENP    | 0.9470 | 0.8374    | 0.6740     | 0.8628 | 0.6900   | 0.7626 | 0.6585    | 0.6300 | 86.2            |

| Method | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| ------ | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| Wanda  | 97.7 | 95.8      | 88.8       | 90.1  | 82.5     | 91.8 | 75.6      | 85.5 | 88.5 |
| ENP    | 97.3 | 96.2      | 89.4       | 90.9  | 77.4     | 89.2 | 69.2      | 79.9 | 86.2 |

#### Qwen3 50%

| Method | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| ------ | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense  | 0.9735 | 0.8709    | 0.7537     | 0.9492 | 0.8920   | 0.8552 | 0.9512    | 0.7880 | —               |
| Wanda  | 0.8064 | 0.6079    | 0.5454     | 0.4898 | 0.2120   | 0.5526 | 0.1159    | 0.1720 | 49.9            |
| ENP    | 0.7540 | 0.6542    | 0.5296     | 0.2388 | 0.0820   | 0.4133 | 0.1341    | 0.1900 | 43.0            |

| Method | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| ------ | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| Wanda  | 82.8 | 69.8      | 72.4       | 51.6  | 23.8     | 64.6 | 12.2      | 21.8 | 49.9 |
| ENP    | 77.5 | 75.1      | 70.3       | 25.2  | 9.2      | 48.3 | 14.1      | 24.1 | 43.0 |

#### Qwen3.6 25%

| Method | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| ------ | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense  | 0.9831 | 0.8986    | 0.8713     | 0.9651 | 0.9480   | 0.8909 | 0.9573    | 0.8000 | —               |
| Wanda  | 0.9603 | 0.8971    | 0.8114     | 0.8567 | 0.7940   | 0.8266 | 0.8902    | 0.6980 | 92.0            |
| ENP    | 0.9673 | 0.9008    | 0.7695     | 0.9105 | 0.8040   | 0.8327 | 0.8049    | 0.6500 | 90.6            |

| Method | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| ------ | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| Wanda  | 97.7 | 99.8      | 93.1       | 88.8  | 83.8     | 92.8 | 93.0      | 87.2 | 92.0 |
| ENP    | 98.4 | 100.2     | 88.3       | 94.3  | 84.8     | 93.5 | 84.1      | 81.2 | 90.6 |

#### Qwen3.6 50%

| Method | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| ------ | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense  | 0.9831 | 0.8986    | 0.8713     | 0.9651 | 0.9480   | 0.8909 | 0.9573    | 0.8000 | —               |
| Wanda  | 0.9050 | 0.7229    | 0.6717     | 0.1547 | 0.1500   | 0.5251 | 0.4207    | 0.3060 | 52.8            |
| ENP    | 0.9065 | 0.8133    | 0.6914     | 0.5102 | 0.1140   | 0.4360 | 0.1646    | 0.1820 | 52.0            |

| Method | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| ------ | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| Wanda  | 92.1 | 80.4      | 77.1       | 16.0  | 15.8     | 58.9 | 43.9      | 38.2 | 52.8 |
| ENP    | 92.2 | 90.5      | 79.4       | 52.9  | 12.0     | 48.9 | 17.2      | 22.7 | 52.0 |

#### DeepSeek 25%

| Method | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| ------ | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense  | 0.7925 | 0.6622    | 0.5549     | 0.7058 | 0.2880   | 0.5613 | 0.5183    | 0.4680 | —               |
| Wanda  | 0.1925 | 0.1925    | 0.2573     | 0.0334 | 0.0180   | 0.2559 | 0.0000    | 0.0120 | 19.9            |
| ENP    | 0.6748 | 0.4762    | 0.5485     | 0.3518 | 0.0660   | 0.4705 | 0.2256    | 0.2000 | 62.3            |

| Method | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| ------ | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| Wanda  | 24.3 | 29.1      | 46.4       | 4.7   | 6.2      | 45.6 | 0.0       | 2.6  | 19.9 |
| ENP    | 85.1 | 71.9      | 98.8       | 49.8  | 22.9     | 83.8 | 43.5      | 42.7 | 62.3 |

#### DeepSeek 50%

| Method | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| ------ | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense  | 0.7925 | 0.6622    | 0.5549     | 0.7058 | 0.2880   | 0.5613 | 0.5183    | 0.4680 | —               |
| Wanda  | 0.1691 | 0.1715    | 0.1373     | 0.0099 | 0.0140   | 0.2104 | 0.0000    | 0.0100 | 14.7            |
| ENP    | 0.3402 | 0.2650    | 0.4933     | 0.0318 | 0.0240   | 0.3594 | 0.0000    | 0.0240 | 31.7            |

| Method | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| ------ | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| Wanda  | 21.3 | 25.9      | 24.7       | 1.4   | 4.9      | 37.5 | 0.0       | 2.1  | 14.7 |
| ENP    | 42.9 | 40.0      | 88.9       | 4.5   | 8.3      | 64.0 | 0.0       | 5.1  | 31.7 |

#### MetaMath

#### Qwen3 25%

| Method | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| ------ | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense  | 0.9735 | 0.8709    | 0.7537     | 0.9492 | 0.8920   | 0.8552 | 0.9512    | 0.7880 | —               |
| Wanda  | 0.9546 | 0.8372    | 0.6953     | 0.9401 | 0.8560   | 0.7841 | 0.4024    | 0.5620 | 85.8            |
| ENP    | 0.9507 | 0.8487    | 0.6977     | 0.9431 | 0.8500   | 0.7801 | 0.5854    | 0.6300 | 89.4            |

| Method | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| ------ | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| Wanda  | 98.1 | 96.1      | 92.3       | 99.0  | 96.0     | 91.7 | 42.3      | 71.3 | 85.8 |
| ENP    | 97.7 | 97.5      | 92.6       | 99.4  | 95.3     | 91.2 | 61.5      | 79.9 | 89.4 |

#### Qwen3 50%

| Method | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| ------ | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense  | 0.9735 | 0.8709    | 0.7537     | 0.9492 | 0.8920   | 0.8552 | 0.9512    | 0.7880 | —               |
| Wanda  | 0.8399 | 0.6761    | 0.5635     | 0.8749 | 0.5980   | 0.5909 | 0.1220    | 0.1940 | 63.1            |
| ENP    | 0.8391 | 0.6642    | 0.6117     | 0.9052 | 0.5860   | 0.5351 | 0.2012    | 0.2300 | 64.7            |

| Method | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| ------ | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| Wanda  | 86.3 | 77.6      | 74.8       | 92.2  | 67.0     | 69.1 | 12.8      | 24.6 | 63.1 |
| ENP    | 86.2 | 76.3      | 81.2       | 95.4  | 65.7     | 62.6 | 21.2      | 29.2 | 64.7 |

#### Qwen3.6 25%

| Method | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| ------ | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense  | 0.9831 | 0.8986    | 0.8713     | 0.9651 | 0.9480   | 0.8909 | 0.9573    | 0.8000 | —               |
| Wanda  | 0.9642 | 0.8897    | 0.7680     | 0.9356 | 0.8640   | 0.8262 | 0.8902    | 0.7000 | 93.3            |
| ENP    | 0.9743 | 0.8976    | 0.7845     | 0.9568 | 0.9100   | 0.8360 | 0.7378    | 0.6200 | 91.6            |

| Method | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| ------ | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| Wanda  | 98.1 | 99.0      | 88.1       | 96.9  | 91.1     | 92.7 | 93.0      | 87.5 | 93.3 |
| ENP    | 99.1 | 99.9      | 90.0       | 99.1  | 96.0     | 93.8 | 77.1      | 77.5 | 91.6 |

#### Qwen3.6 50%

| Method | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| ------ | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense  | 0.9831 | 0.8986    | 0.8713     | 0.9651 | 0.9480   | 0.8909 | 0.9573    | 0.8000 | —               |
| Wanda  | 0.8771 | 0.6442    | 0.6614     | 0.5474 | 0.4140   | 0.5832 | 0.4512    | 0.4100 | 62.6            |
| ENP    | 0.9200 | 0.8104    | 0.7017     | 0.9098 | 0.6800   | 0.6342 | 0.1707    | 0.2460 | 68.8            |

| Method | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| ------ | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| Wanda  | 89.2 | 71.7      | 75.9       | 56.7  | 43.7     | 65.5 | 47.1      | 51.2 | 62.6 |
| ENP    | 93.6 | 90.2      | 80.5       | 94.3  | 71.7     | 71.2 | 17.8      | 30.8 | 68.8 |

#### DeepSeek 25%

| Method | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| ------ | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense  | 0.7925 | 0.6622    | 0.5549     | 0.7058 | 0.2880   | 0.5613 | 0.5183    | 0.4680 | —               |
| Wanda  | 0.1155 | 0.1477    | 0.1697     | 0.0288 | 0.0080   | 0.2497 | 0.0000    | 0.0340 | 15.8            |
| ENP    | 0.6350 | 0.4527    | 0.5170     | 0.3639 | 0.0700   | 0.4562 | 0.2012    | 0.2260 | 60.7            |

| Method | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| ------ | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| Wanda  | 14.6 | 22.3      | 30.6       | 4.1   | 2.8      | 44.5 | 0.0       | 7.3  | 15.8 |
| ENP    | 80.1 | 68.4      | 93.2       | 51.6  | 24.3     | 81.3 | 38.8      | 48.3 | 60.7 |

#### DeepSeek 50%

| Method | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| ------ | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense  | 0.7925 | 0.6622    | 0.5549     | 0.7058 | 0.2880   | 0.5613 | 0.5183    | 0.4680 | —               |
| Wanda  | 0.1922 | 0.1343    | 0.1744     | 0.0068 | 0.0040   | 0.2052 | 0.0000    | 0.0060 | 14.5            |
| ENP    | 0.5347 | 0.3369    | 0.5130     | 0.0387 | 0.0260   | 0.3282 | 0.0122    | 0.0400 | 36.8            |

| Method | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| ------ | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| Wanda  | 24.3 | 20.3      | 31.4       | 1.0   | 1.4      | 36.6 | 0.0       | 1.3  | 14.5 |
| ENP    | 67.5 | 50.9      | 92.4       | 5.5   | 9.0      | 58.5 | 2.4       | 8.5  | 36.8 |

#### Code-Alpaca

#### Qwen3 25%

| Method | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| ------ | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense  | 0.9735 | 0.8709    | 0.7537     | 0.9492 | 0.8920   | 0.8552 | 0.9512    | 0.7880 | —               |
| Wanda  | 0.9535 | 0.8273    | 0.6851     | 0.8802 | 0.7600   | 0.7825 | 0.8354    | 0.7220 | 91.6            |
| ENP    | 0.9566 | 0.8264    | 0.6701     | 0.9212 | 0.7620   | 0.7778 | 0.8537    | 0.7200 | 92.1            |

| Method | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| ------ | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| Wanda  | 97.9 | 95.0      | 90.9       | 92.7  | 85.2     | 91.5 | 87.8      | 91.6 | 91.6 |
| ENP    | 98.3 | 94.9      | 88.9       | 97.1  | 85.4     | 90.9 | 89.7      | 91.4 | 92.1 |

#### Qwen3 50%

| Method | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| ------ | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense  | 0.9735 | 0.8709    | 0.7537     | 0.9492 | 0.8920   | 0.8552 | 0.9512    | 0.7880 | —               |
| Wanda  | 0.8326 | 0.6601    | 0.5470     | 0.5929 | 0.3080   | 0.5786 | 0.6037    | 0.5300 | 66.2            |
| ENP    | 0.8323 | 0.6200    | 0.5446     | 0.6194 | 0.2740   | 0.5372 | 0.6585    | 0.5840 | 66.4            |

| Method | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| ------ | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| Wanda  | 85.5 | 75.8      | 72.6       | 62.5  | 34.5     | 67.7 | 63.5      | 67.3 | 66.2 |
| ENP    | 85.5 | 71.2      | 72.3       | 65.3  | 30.7     | 62.8 | 69.2      | 74.1 | 66.4 |

#### Qwen3.6 25%

| Method | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| ------ | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense  | 0.9831 | 0.8986    | 0.8713     | 0.9651 | 0.9480   | 0.8909 | 0.9573    | 0.8000 | —               |
| Wanda  | 0.9622 | 0.8880    | 0.7695     | 0.8453 | 0.7980   | 0.8239 | 0.9024    | 0.7160 | 91.6            |
| ENP    | 0.9718 | 0.9052    | 0.7877     | 0.9333 | 0.8560   | 0.8336 | 0.8780    | 0.6960 | 93.7            |

| Method | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| ------ | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| Wanda  | 97.9 | 98.8      | 88.3       | 87.6  | 84.2     | 92.5 | 94.3      | 89.5 | 91.6 |
| ENP    | 98.9 | 100.7     | 90.4       | 96.7  | 90.3     | 93.6 | 91.7      | 87.0 | 93.7 |

#### Qwen3.6 50%

| Method | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| ------ | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense  | 0.9831 | 0.8986    | 0.8713     | 0.9651 | 0.9480   | 0.8909 | 0.9573    | 0.8000 | —               |
| Wanda  | 0.9160 | 0.7527    | 0.6740     | 0.2214 | 0.2520   | 0.6165 | 0.5366    | 0.4420 | 60.5            |
| ENP    | 0.9245 | 0.7915    | 0.6867     | 0.6414 | 0.4340   | 0.6184 | 0.0366    | 0.5400 | 64.2            |

| Method | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| ------ | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| Wanda  | 93.2 | 83.8      | 77.4       | 22.9  | 26.6     | 69.2 | 56.1      | 55.2 | 60.5 |
| ENP    | 94.0 | 88.1      | 78.8       | 66.5  | 45.8     | 69.4 | 3.8       | 67.5 | 64.2 |

#### DeepSeek 25%

| Method | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| ------ | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense  | 0.7925 | 0.6622    | 0.5549     | 0.7058 | 0.2880   | 0.5613 | 0.5183    | 0.4680 | —               |
| Wanda  | 0.1308 | 0.1052    | 0.1413     | 0.0273 | 0.0180   | 0.2173 | 0.0000    | 0.0300 | 14.1            |
| ENP    | 0.6728 | 0.4999    | 0.5430     | 0.3389 | 0.0780   | 0.4719 | 0.1341    | 0.2240 | 61.4            |

| Method | ARC  | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| ------ | ---- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| Wanda  | 16.5 | 15.9      | 25.5       | 3.9   | 6.2      | 38.7 | 0.0       | 6.4  | 14.1 |
| ENP    | 84.9 | 75.5      | 97.9       | 48.0  | 27.1     | 84.1 | 25.9      | 47.9 | 61.4 |

#### DeepSeek 50%

| Method | ARC    | HellaSwag | WinoGrande | GSM8K  | MATH-500 | MMLU   | HumanEval | MBPP   | Mean retained % |
| ------ | ------ | --------- | ---------- | ------ | -------- | ------ | --------- | ------ | --------------- |
| Dense  | 0.7925 | 0.6622    | 0.5549     | 0.7058 | 0.2880   | 0.5613 | 0.5183    | 0.4680 | —               |
| Wanda  | 0.0392 | 0.0597    | 0.0213     | 0.0023 | 0.0060   | 0.1578 | 0.0000    | 0.0040 | 6.1             |
| ENP    | 0.0316 | 0.0286    | 0.0213     | 0.0008 | 0.0020   | 0.1023 | 0.0000    | 0.0000 | 3.9             |

| Method | ARC | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | HumanEval | MBPP | Mean |
| ------ | --- | --------- | ---------- | ----- | -------- | ---- | --------- | ---- | ---- |
| Wanda  | 4.9 | 9.0       | 3.8        | 0.3   | 2.1      | 28.1 | 0.0       | 0.9  | 6.1  |
| ENP    | 4.0 | 4.3       | 3.8        | 0.1   | 0.7      | 18.2 | 0.0       | 0.0  | 3.9  |

## F. 校准路由：每个专家被打到的 \(N_e\) 分布

HARP 融合里的 \(N_e\) 就是校准 cache 上、每个 `(layer, expert)` 的 **route_counts**：该专家在 128×2048 上被 native router 选中的 token 次数。\(N_e=0\) 的专家没有激活统计，Wanda/ENP 通道分退回 weight-L2（同构）或 Channel-SP（HARP）。四个校准都是同一套 cache 协议，只换文本域。

Qwen3 / Qwen3.6 上 Wanda 与 ENP 的 `route_counts` **逐格相同**（同一 cache、同一 router，Spearman = 1，\(\max|\Delta N_e|=0\)）。下面 Qwen 表只写一份，标 Wanda=ENP。DeepSeek 上 Wanda 按 routed token 计数、ENP-COS 的 \(N_e\) 不同（同校准 Spearman 只有 0.13–0.33），分开写。

有效专家数 \(n_{\mathrm{eff}}=1/\sum_e p_e^2\)，\(p_e=N_e/\sum N\)，对层取平均。Gini 是把所有 `(layer, expert)` 槽位摊平后的基尼系数。Top-10% 是 \(N_e\) 最大的 10% 槽位占总路由次数的比例。

### F.1 覆盖与集中度（Wanda；Qwen 与 ENP 相同）

| Model | Cal | Experts/layer | Unseen slots | Unseen % | Median \(N_e\) | Gini | Mean \(n_{\mathrm{eff}}\) | \(n_{\mathrm{eff}}/E\) | Top-10% share | Max unseen/layer |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3 | Wiki | 128 | 81 | 1.3 | 8778 | 0.623 | 47.6 | 37.2% | 41.0% | 5 |
| Qwen3 | C4 | 128 | 30 | 0.5 | 7854 | 0.608 | 52.2 | 40.8% | 37.9% | 5 |
| Qwen3 | MetaMath | 128 | 63 | 1.0 | 11206 | 0.559 | 54.6 | 42.7% | 37.5% | 7 |
| Qwen3 | Code-Alpaca | 128 | 61 | 1.0 | 10140 | 0.577 | 50.4 | 39.4% | 40.4% | 7 |
| Qwen3.6 | Wiki | 256 | 6 | 0.1 | 4527 | 0.578 | 97.8 | 38.2% | 40.4% | 3 |
| Qwen3.6 | C4 | 256 | 0 | 0.0 | 4340 | 0.567 | 112.5 | 43.9% | 37.2% | 0 |
| Qwen3.6 | MetaMath | 256 | 2 | 0.0 | 5362 | 0.547 | 103.9 | 40.6% | 38.3% | 1 |
| Qwen3.6 | Code-Alpaca | 256 | 3 | 0.0 | 5091 | 0.559 | 96.9 | 37.9% | 41.2% | 1 |
| DeepSeek | Wiki | 64 | 0 | 0.0 | 23430 | 0.226 | 55.1 | 86.1% | 18.3% | 0 |
| DeepSeek | C4 | 64 | 0 | 0.0 | 22983 | 0.274 | 51.3 | 80.2% | 20.8% | 0 |
| DeepSeek | MetaMath | 64 | 0 | 0.0 | 21616 | 0.315 | 48.2 | 75.3% | 22.8% | 0 |
| DeepSeek | Code-Alpaca | 64 | 0 | 0.0 | 20985 | 0.362 | 44.2 | 69.1% | 26.1% | 0 |

槽位数：Qwen3 48×128=6144，Qwen3.6 40×256=10240，DeepSeek 26×64=1664。每层路由次数固定：Qwen3 \(128\times2048\times8=2{,}097{,}152\)，Qwen3.6 \(128\times2048\times8=2{,}097{,}152\)（256 专家均摊所以均值 \(N_e\) 减半），DeepSeek 均值 \(N_e=24{,}576\)。

读法：

- Qwen 校准覆盖不均匀：大约 10% 的专家槽吃掉 37–41% 的路由，\(n_{\mathrm{eff}}\) 只有专家数的四成。Wiki 上 Qwen3 还有 81 个完全没被打到的槽（最多一层 5 个）；C4 未见到的更少（30），MetaMath / Code-Alpaca 约 60。
- DeepSeek 没有 \(N_e=0\)，但域越偏、分布越尖：Wiki Gini 0.226 / \(n_{\mathrm{eff}}=55.1\)，Code-Alpaca Gini 0.362 / \(n_{\mathrm{eff}}=44.2\)。这和 E.1 里 DeepSeek 50% ENP 在 Code-Alpaca 崩掉、在 MetaMath 却不崩是同一件事的路由侧。
- 没被打到的专家集合几乎不共用：Qwen3 上 Wiki 与 C4/MM/CA 的 unseen Jaccard 只有 0.10–0.11。换校准不是少打同一批专家，是 **另一批专家变冷**。

### F.2 DeepSeek：Wanda 与 ENP 的 \(N_e\) 不是同一张图

| Cal | Wanda median | ENP median | Wanda Gini | ENP Gini | Wanda \(n_{\mathrm{eff}}\) | ENP \(n_{\mathrm{eff}}\) | Spearman(Wanda, ENP) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Wiki | 23430 | 23286 | 0.226 | 0.231 | 55.1 | 54.6 | 0.326 |
| C4 | 22983 | 22357 | 0.274 | 0.286 | 51.3 | 50.6 | 0.220 |
| MetaMath | 21616 | 22318 | 0.315 | 0.285 | 48.2 | 50.5 | 0.128 |
| Code-Alpaca | 20985 | 21495 | 0.362 | 0.331 | 44.2 | 46.6 | 0.223 |

两边都没有 unseen，但哪些专家热、哪些专家冷对不齐。所以 DeepSeek 上「同一校准、Wanda vs ENP」在 E 节里下游差很大，不只是打分公式不同，**校准观测的专家分配本身就不同**。

### F.3 \(N_e\) 直方图（Wanda 槽位数）

桶：0 / 1–100 / 101–1k / 1k–10k / 10k–50k / >50k。

| Model | Cal | 0 | 1–100 | 101–1k | 1k–10k | 10k–50k | >50k |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3 | Wiki | 81 | 707 | 722 | 1732 | 2483 | 419 |
| Qwen3 | C4 | 30 | 390 | 924 | 1976 | 2370 | 454 |
| Qwen3 | MetaMath | 63 | 283 | 716 | 1810 | 2947 | 325 |
| Qwen3 | Code-Alpaca | 61 | 131 | 699 | 2152 | 2713 | 388 |
| Qwen3.6 | Wiki | 6 | 343 | 1401 | 5730 | 2657 | 103 |
| Qwen3.6 | C4 | 0 | 170 | 1416 | 5632 | 2967 | 55 |
| Qwen3.6 | MetaMath | 2 | 274 | 1380 | 5746 | 2742 | 96 |
| Qwen3.6 | Code-Alpaca | 3 | 236 | 1218 | 6287 | 2343 | 153 |
| DeepSeek | Wiki | 0 | 0 | 3 | 74 | 1556 | 31 |
| DeepSeek | C4 | 0 | 0 | 1 | 135 | 1469 | 59 |
| DeepSeek | MetaMath | 0 | 0 | 6 | 193 | 1372 | 93 |
| DeepSeek | Code-Alpaca | 0 | 0 | 12 | 270 | 1269 | 113 |

Qwen 的长尾在 Wiki / C4 更重（>50k 槽更多）；MetaMath 把质量往 10k–50k 推，极端热专家变少。DeepSeek 主体都在 10k–50k，Code-Alpaca 把更多专家推进 >50k，同时 1k–10k 的冷专家也变多——更两极化。

### F.4 跨校准：同一专家的 \(N_e\) 排位保不保

对 Wanda 的摊平 \(N_e\) 向量，相对 Wiki 的 Spearman / Pearson。Qwen 的 ENP 与 Wanda 相同，不另列。

| Model | vs | Spearman | Pearson | Unseen Jaccard |
| --- | --- | ---: | ---: | ---: |
| Qwen3 | C4 | 0.877 | 0.740 | 0.11 |
| Qwen3 | MetaMath | 0.152 | −0.068 | 0.10 |
| Qwen3 | Code-Alpaca | −0.242 | −0.236 | 0.11 |
| Qwen3.6 | C4 | 0.731 | 0.556 | 0.00 |
| Qwen3.6 | MetaMath | 0.263 | 0.081 | 0.00 |
| Qwen3.6 | Code-Alpaca | 0.010 | −0.005 | 0.00 |
| DeepSeek | C4 | 0.220 | 0.192 | — |
| DeepSeek | MetaMath | 0.294 | 0.219 | — |
| DeepSeek | Code-Alpaca | 0.259 | 0.201 | — |

C4 与 Wiki 都是通用网页/维基，Qwen 的专家热度还对得上（Spearman 0.73–0.88）。MetaMath / Code-Alpaca 则把专家排序打乱：Qwen3 对 Code-Alpaca 已经是 **负相关**。DeepSeek 即便 C4 也对不齐 Wiki（0.22）。这就是 E.1「换校准、下游不是平移」在路由层上的对应物：激活剪枝跟的是一份 **域特异的专家–通道观测**，不是对同一批专家的无偏估计。

HARP cfuse 在 \(N_e=0\) 时退回 Channel-SP，在 \(N_e>0\) 时用结构分约束激活分，所以能挡住这份校准特异路由；它不能让四个校准的 \(N_e\) 变得一样。

统计来源：`/data/xinpeigao/evalscope_results/_artifacts/{wanda,enp}/<model>/statistics.pt`（Wiki），以及 `{wanda,enp}_{c4,metamath,codealpaca}/<model>/statistics.pt`。

## G. 纯 HARP per-token routed FFN

只统计 **routed-expert FFN**：每个 MoE 层里被 top-\(k\) 选中的专家，其 gate / up / down 三个投影。不计入 attention、router、shared expert / shared MLP、non-MoE 层、embedding、LM head。HARP 只改 routed-expert 的 intermediate 宽度，所以相对 dense 的比值就是实际被剪掉的那一块。

gate/up/down 都正比于宽度，精确 FLOPs 用 \(3 d_{\mathrm{model}} K_{\ell,e}\) 代替 \(K_{\ell,e}\)，**比值不变**。

### 记号

层 \(\ell\)、专家 \(e\)：dense 宽度 \(D\)，HARP 宽度 \(K_{\ell,e}\)，top-\(k\)。token \(t\) 的 routed 集合 \(S_t(\ell)\)。

\[
C_{\ell,t}^{\mathrm{dense}}=kD,\qquad
C_{\ell,t}^{\mathrm{HARP}}=\sum_{e\in S_t(\ell)}K_{\ell,e},\qquad
r_{\ell,t}=\frac{1}{kD}\sum_{e\in S_t(\ell)}K_{\ell,e}.
\]

按路由频率加权：\(p_{\ell,e}=N_{\ell,e}/T_\ell\)，\(\sum_e p_{\ell,e}=k\)。层平均

\[
r_\ell=\frac{1}{kD}\sum_e p_{\ell,e}K_{\ell,e}.
\]

各 MoE 层 \(D\)、\(k\)、token 数相同，层平均与全局 FLOPs 加权等价：

\[
\mathrm{RelFFN}=\frac{\sum_{\ell,e}p_{\ell,e}K_{\ell,e}}{D\cdot L_{\mathrm{MoE}}\cdot k}
=\frac{\sum_{\ell,e}N_{\ell,e}K_{\ell,e}}{D\sum_{\ell,e}N_{\ell,e}}.
\]

\(K_{\ell,e}\) 来自 **HARP-nologbest** 的 `v2_rank_adaptive_{25|50}pct.pt`（CalibrationFree 分配）。通道排序（Wanda / ENP / fuse / rand）**不改宽度**，RelFFN 相同。同构 Wanda/ENP 每专家都是 \(K=D\cdot(1-s)\)，RelFFN 就是名义保留比。

\(p_{\ell,e}\) 用 native router 在四个 128×2048 校准 cache 上的 `route_counts`（HARP 不改 router）。没有 full8 评测集上的路由 dump；换域等于换 \(p\)。Qwen 上 Wanda 与 ENP 的 \(N_e\) 相同；DeepSeek 用 Wanda 计数，ENP 加权与下表差 \(\le 0.3\) 个百分点。

名义预算：25% sparsity 保留 75% 宽度，50% 保留 50%。源宽度 \(D\)：Qwen3 768 / Qwen3.6 512 / DeepSeek 1408；\(k=8,8,6\)。HARP 层内均值锁在预算上，所以 **Budget RelFFN** \(=0.75\) / \(0.50\)。Route-weighted RelFFN 可以偏离它：热专家若更宽，实际计算高于名义稀疏度。

### G.1 RelFFN（相对 dense routed FFN）

| Model | Sparsity | Budget | Wiki | C4 | MetaMath | Code-Alpaca |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3 | 25 | 75.0 | 80.3 | 80.2 | 72.9 | 74.4 |
| Qwen3 | 50 | 50.0 | 55.3 | 55.2 | 47.9 | 49.4 |
| Qwen3.6 | 25 | 75.0 | 80.7 | 81.0 | 71.1 | 72.8 |
| Qwen3.6 | 50 | 50.0 | 55.7 | 56.0 | 46.1 | 47.8 |
| DeepSeek | 25 | 75.0 | 74.3 | 74.3 | 74.7 | 74.1 |
| DeepSeek | 50 | 50.0 | 49.3 | 49.3 | 49.7 | 49.1 |

相对 dense 的节省 \(=1-\mathrm{RelFFN}\)。Wiki 上纯 HARP：Qwen3 25/50 省 **19.7% / 44.7%**，Qwen3.6 **19.3% / 44.3%**，DeepSeek **25.7% / 50.7%**。

### G.2 相对预算：路由有没有打到宽专家

\(\mathrm{RelFFN}-\mathrm{Budget}\)（百分点）。正值 = 实际 FFN 比名义稀疏度更贵。同一套三档 \(\{\mathrm{budget}\pm 128\}\)，25 与 50 的偏离相同。Pearson 是各层 \(N_e\) 与 \(K_e\) 相关再平均。

| Model | Wiki Δ | C4 Δ | MetaMath Δ | Code-Alpaca Δ | Wiki Pearson | C4 | MM | CA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3 | +5.3 | +5.2 | −2.1 | −0.6 | +0.29 | +0.31 | −0.13 | −0.03 |
| Qwen3.6 | +5.7 | +6.0 | −3.9 | −2.2 | +0.22 | +0.25 | −0.15 | −0.08 |
| DeepSeek | −0.7 | −0.7 | −0.3 | −0.9 | −0.22 | −0.21 | −0.06 | −0.18 |

读法：

- 名义 50% 不是 token 级 50%。Qwen 在 Wiki/C4 上 Channel-SP 给的宽专家更常被打到（Pearson > 0），per-token routed FFN 大约是 dense 的 **55–56%**，比预算多花约 5 个百分点。
- MetaMath / Code-Alpaca 把流量推向窄专家，Qwen 的 RelFFN 落到预算以下（50% 档 46–49%）。
- DeepSeek 的宽度与路由略负相关，RelFFN 贴着预算略低（50% 档 49.1–49.7%）。异构宽度几乎没改它的 token 级 FFN。
- 这不是评测分数。它只回答「这个 token 经过 HARP 时，routed FFN 算了多少」。A/B 主表的 Wiki 分数应对 Wiki RelFFN；E 节换校准应对相应列。

实现：`HARP/rel_ffn.py`。HSP Δ=64 / Δ=128 的 full8 分数对照见 `Results_HARP_ablations.md`。full8 轨迹上的 RelFFN（HARP-nologbest 与 HSP \(\Delta=128\)）见 **G.3**。

### G.3 full8 generate RelFFN：HARP-nologbest vs HSP \(\Delta=128\)

G.1/G.2 的 \(p_{\ell,e}\) 来自校准 cache 的 `route_counts`（Wiki / C4 / MetaMath / Code-Alpaca），宽度来自 HARP-nologbest profile，**没有**在稀疏 checkpoint 上再 decode。G.3 是另一件事：在对应 padded checkpoint 上，对 frozen `full8_v1` 的 31382 条 greedy 轨迹 **重新 generate**（temperature 0，`enable_thinking=false`），用 vLLM `enable_return_routed_experts` 收 native top-\(k\)，再按同一套 RelFFN 公式加权 \(K_{\ell,e}\)。

HARP 宽度来自 `v2_rank_adaptive_{25|50}pct.pt`，评测目录 `HARPNoLogBest_202609162249`。HSP 宽度来自 `hsp_{25|50}pct_hetero.pt`（Expert-SP，\(\Delta=128\)，同一全局预算），评测目录 `HSPDelta128_202609131611`（Qwen3.6 50% 为已有的 `202609130330`）。通道排序不改宽度，所以 G.3 的 HARP 列也代表 nologbest 的 Wanda/ENP/fuse/rand 宽度。

表中 RelFFN / Budget 为相对 dense routed FFN 的百分数。Δ = RelFFN − Budget（百分点）；正值 = 比名义稀疏度更贵。Pearson 是各层 \(N_e\) 与 \(K_e\) 相关再平均。

| Model | Sparsity | Budget | HARP RelFFN | HSP RelFFN | HARP Δ | HSP Δ | HARP Pearson | HSP Pearson |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3 | 25 | 75.0 | 78.1 | 77.7 | +3.1 | +2.7 | +0.27 | +0.27 |
| Qwen3 | 50 | 50.0 | 52.3 | 51.9 | +2.3 | +1.9 | +0.21 | +0.20 |
| Qwen3.6 | 25 | 75.0 | 77.8 | 77.6 | +2.8 | +2.6 | +0.15 | +0.15 |
| Qwen3.6 | 50 | 50.0 | 52.9 | 52.6 | +2.9 | +2.6 | +0.15 | +0.15 |
| DeepSeek | 25 | 75.0 | 75.2 | 75.2 | +0.2 | +0.2 | +0.12 | +0.12 |
| DeepSeek | 50 | 50.0 | 50.3 | 50.2 | +0.3 | +0.2 | +0.15 | +0.15 |

相对 dense 的节省 \(=1-\mathrm{RelFFN}\)。full8 上 HARP：Qwen3 25/50 省 **21.9% / 47.7%**，Qwen3.6 **22.3% / 47.1%**，DeepSeek **24.8% / 49.7%**。HSP \(\Delta=128\) 再多省大约 0.1–0.4 个百分点（宽度档更碎，热专家略不那么宽）。

读法：

- full8 混合轨迹比 G.1 的 Wiki cache 更贴预算。Wiki 上 HARP Qwen3/Qwen3.6 50% 是 55.3 / 55.7；full8 generate 落到 **52.3 / 52.9**。数学/代码把流量推向窄专家，把 Wiki 上那 +5 个百分点的「宽专家溢价」削掉一半左右。DeepSeek 两边都贴 50%。
- HARP 与 HSP \(\Delta=128\) 在 full8 上几乎同一条 RelFFN。Qwen 差 0.3–0.4 个百分点，DeepSeek 打平。Expert-SP 改的是谁宽谁窄，不是全局预算；token 级 routed FFN 主要由 \(p\) 和层内均值锁预算决定。
- 这不是下游分数。A/B 主表仍应对 G.1 的 Wiki RelFFN；G.3 回答的是「跑 full8 时，这两套异构宽度实际算了多少 routed FFN」。

实现：`HARP/full8_rel_ffn.py`（`--mode generate`）。产物：

- HARP：`_artifacts/harp_score_ablation/HARPNoLogBest/<model>/full8_rel_ffn_{25|50}pct_generate.json`
- HSP：`_artifacts/hsp_ablation/HSPDelta128/<model>/full8_rel_ffn_{25|50}pct_generate.json`

## H. HARP-nologbest 伪代码

本节写的是 **HARP-nologbest 本体**（token `HARPNoLogBest`）：CalibrationFree、Layer-SP / Expert-SP / Channel-SP，Layer water-fill 用 \(1+\mathrm{CV}^2\)（实现里 `waterfill_score='ratio'`），\((\gamma,\eta,\Delta)=(2,0.15,128)\)。A/B 的 Wanda / ENP / fuse / rand **只替换第 4 步的通道排序**，\(K_{\ell,e}\) 与这里相同。

权重只来自 routed expert 的 gate / up / down。对向量 \(\Theta\in\mathbb{R}^N\)，参与比

\[
\rho(\Theta)=N\|\Theta\|_2^2/\|\Theta\|_1^2=1+\mathrm{CV}^2
\]

（\(\|\Theta\|_1=0\) 时 \(\rho=0\)）。下面 water-fill 和三处排序都用 \(\rho\)。排名缓存里存的是 \(S=\log\rho\)，NoLogBest 在 water-fill 里立刻做 \(\exp(S)=\rho\)；Expert-SP / Channel-SP 只比大小，单调 \(\log\) 不改变名次。算法本身没有留下 \(\log\)。

档宽：名义均值 \(K_{\mathrm{mid}}=D(1-s)\)，\(K_{\mathrm{low}}=K_{\mathrm{mid}}-\Delta\)，\(K_{\mathrm{high}}=K_{\mathrm{mid}}+\Delta\)。全局约束 \(\sum_{\ell,e}K_{\ell,e}=L E K_{\mathrm{mid}}\)。\(D\)：Qwen3 768 / Qwen3.6 512 / DeepSeek 1408。

实现：`HARP/allocate_v2_rank_adaptive_widths`（`rank_adaptive_core.py`）+ Channel-SP prefix（`CSP.csp_core.rank_channels_by_csp`）。

```text
Input: dense MoE checkpoint; sparsity s ∈ {0.25, 0.50}
       γ = 2, η = 0.15, Δ = 128
Output: widths K[ℓ, e] ∈ {K_low, K_mid, K_high}
        retained channel index prefix of length K[ℓ, e]

# 0. 档宽
K_mid  ← D · (1 − s)
K_low  ← K_mid − Δ
K_high ← K_mid + Δ

# 1. CalibrationFree 结构分（只看权重；不用 log）
for each MoE layer ℓ:
    for each expert e:
        Θ_{ℓ,e} ← concat(gate, up, down) of expert e
        ρ_exp[ℓ, e] ← ρ(Θ_{ℓ,e})                         # Expert-SP
        for each intermediate channel c:
            ρ_ch[ℓ, e, c] ← ρ(slice of gate/up/down at c)  # Channel-SP
    ρ_layer[ℓ] ← ρ(concat of all routed expert weights in ℓ)  # Layer-SP

# 2. Layer-SP water-fill（质量就是 ρ = 1+CV²）
w[ℓ] ← (ρ_layer[ℓ] / ∑_{ℓ'} ρ_layer[ℓ'])^γ
t[ℓ] ← K_low + (K_mid − K_low) · w[ℓ] / mean(w)
t     ← project t onto [K_low, K_high]^L with ∑_ℓ t[ℓ] = L · K_mid
         (surplus to high-ρ_layer layers first; deficit from low-ρ_layer last)
B[ℓ] ← t[ℓ] · E                          # 层通道预算

# 3. Expert-SP 三档搜索 + 余量传递
π ← layers sorted by ρ_layer descending
R ← 0
for ℓ in π:
    avail ← B[ℓ] + R
    (n_high, n_mid, n_low) ← argmax extra width
         extra = n_high(K_high−K_low) + n_mid(K_mid−K_low)
         s.t. n_high K_high + n_mid K_mid + n_low K_low ≤ avail
              n_high + n_mid + n_low = E
              each count ≥ ηE if feasible, else drop the floor
         tie-break: more even (n_high, n_mid, n_low)
    assign K[ℓ, ·] by Expert-SP order: top n_high → K_high,
         next n_mid → K_mid, rest → K_low
    R ← avail − ∑_e K[ℓ, e]
# 3b. discrete closure（只升档，不降档）
# 第 3 步每层都 ≤ avail，且余量向下传递，故 ∑ K ≤ L E K_mid，缺口
# missing = L E K_mid − ∑ K 必为 Δ 的非负倍数。每次把一名专家升一档
# （K_low→K_mid 或 K_mid→K_high），补 Δ 通道。
U ← missing / Δ
candidates ← {(ℓ, e) | K[ℓ,e] < K_high}
sort candidates lexicographically by
    (rank_ℓ, rank_{ℓ,e}):
      rank_ℓ     = Layer-SP 降序名次（ρ_layer 最高的层 = 0）
      rank_{ℓ,e} = 该层内 Expert-SP 降序名次（ρ_exp 最高的专家 = 0）
      同名次用 stable 下标；元组再按 (ℓ, e) 打破并列
promote the first U candidates by +Δ
# 现在 ∑_{ℓ,e} K[ℓ,e] = L E K_mid，取值仍只在 {K_low, K_mid, K_high}

# 4. Channel-SP prefix（nologbest 本体）
for each (ℓ, e):
    keep the top-K[ℓ,e] channels under ρ_ch[ℓ, e, ·]
    (stable argsort; lower index wins ties)
```

Wanda / ENP / fuse / cfuse / rand：第 1–3 步冻结，只改第 4 步的通道序。fuse / cfuse 用 rank-normalize 后的 \(z^{SP},z^{Act}\) 与文首的 \(w_e\)；\(N_e=0\) 的 Wanda/ENP 单路退回 Channel-SP。

Head change-point（32 次 0.5% 扰动）只写进 profile 诊断，**不**决定 \(n_{\mathrm{high}}\)。

## I. HARP-nologbest rank（\(\rho=1+\mathrm{CV}^2\)）

与 `HARP/rank_analysis_report.md` 同一套问题：有没有稳定 head、特殊对象是不是由下层少数对象驱动、HARP 该怎么用。
分数换成 NoLogBest 实际使用的 \(\rho=N\|\Theta\|_2^2/\|\Theta\|_1^2=1+\mathrm{CV}^2\)（排名缓存里的 \(S=\log\rho\) 先做 \(\exp\)）。
Expert / Channel 的名次与 \(\log\) 相同；change point、Cohen's \(d\)、2σ 和消融用的是 \(\rho\) 的数值，所以和旧报告不必相同。
2σ 只作辅助。Channel 分段只在前 25% rank 里找 head。
消融按 \(\rho\) 重算上一级分数，不是剩余分数的均值。
扰动：每个序列 32 次、相对噪声 0.5%；稳定 = 分界位移 \(\le 2\) 的比例 \(\ge 75\%\)。
模型只含 Qwen3、Qwen3.6、DeepSeek。

### I.1 三层 rank 的核心分段

#### Qwen3

| 层级 | 是否有自然分段 | 分界 rank | 分界前后差异 | 扰动后是否稳定 |
| --- | --- | ---: | ---: | --- |
| Layer | 有弱分段但不稳定 | 3 | d=2.728 | 是（within_two=1.000） |
| Expert | 有稳定 head | 7 （29/48 层有 head） | d=3.013 | 是 （mean within_two=0.956） |
| Channel | 有弱分段但不稳定 | 39 （2889/6144 expert 有 head） | d=2.966；d@10%=2.227 | 是（pooled within_two=1.000） |

#### Qwen3.6

| 层级 | 是否有自然分段 | 分界 rank | 分界前后差异 | 扰动后是否稳定 |
| --- | --- | ---: | ---: | --- |
| Layer | 有弱分段但不稳定 | 2 | d=3.268 | 是（within_two=1.000） |
| Expert | 有弱分段但不稳定 | 13 （24/40 层有 head） | d=2.796 | 是 （mean within_two=0.796） |
| Channel | 有弱分段但不稳定 | 26 （4579/10240 expert 有 head） | d=2.755；d@10%=1.896 | 是（pooled within_two=1.000） |

#### DeepSeek

| 层级 | 是否有自然分段 | 分界 rank | 分界前后差异 | 扰动后是否稳定 |
| --- | --- | ---: | ---: | --- |
| Layer | 有稳定 head | 4 | d=2.275 | 是（within_two=1.000） |
| Expert | 有弱分段但不稳定 | 4 （16/26 层有 head） | d=2.567 | 否 （mean within_two=0.712） |
| Channel | 有弱分段但不稳定 | 71 （636/1664 expert 有 head） | d=2.632；d@10%=2.142 | 是（pooled within_two=1.000） |

### I.2 特殊对象

定义：候选特殊对象 = rank head（top 10%）且该层序列有 gap / change point。没有 gap 时仍列出 top rank，但不把它当成可保护对象。

| Model | Special layers | Special experts | Special channels |
| --- | --- | --- | --- |
| Qwen3 | L0, L47, L37, L41, L25（弱分段 k=3；Layer 0 outlier；top10%=L0, L47, L37, L41, L25） | L0 top10% n=13（有额外 head gap，稳定，k=10）；L47 top10% n=13（有额外 head gap，稳定，k=7）；L37 top10% n=13（无额外 head gap，稳定，k=7）；L41 top10% n=13（有额外 head gap，稳定，k=7）；L25 top10% n=13（有额外 head gap，稳定，k=7） | 每个 expert 的 top 10% prefix；head gap 比例 2889/6144；top10% 平均 ρ 质量=0.104 |
| Qwen3.6 | L0, L39, L33, L32（仅端点尖峰；Layer 0 outlier；top10%=L0, L39, L33, L32） | L0 top10% n=26（有额外 head gap，稳定，k=13）；L39 top10% n=26（有额外 head gap，稳定，k=13）；L33 top10% n=26（有额外 head gap，稳定，k=13）；L32 top10% n=26（有额外 head gap，稳定，k=13） | 每个 expert 的 top 10% prefix；head gap 比例 4579/10240；top10% 平均 ρ 质量=0.105 |
| DeepSeek | L11, L10, L8（有明显 gap；top10%=L11, L10, L8） | L11 top10% n=7（有额外 head gap，稳定，k=5）；L10 top10% n=7（有额外 head gap，不稳定，k=4）；L8 top10% n=7（无额外 head gap，稳定，k=4） | 每个 expert 的 top 10% prefix；head gap 比例 636/1664；top10% 平均 ρ 质量=0.102 |

### I.3 层级归因（按 \(\rho\) 重算）

#### I.3.1 Channel → Expert

对每个候选特殊 layer 的 top 10% expert：去掉该 expert 的 top 1%/5%/10% channel 后，用剩余 channel 的 L1/L2 重算 Expert \(\rho\)。
逐 expert 明细在 `_artifacts/harp_rank_analysis_ratio/<model>/core_questions.json`；这里只保留计数和每层最高 \(\rho\) 的一行。

| Model | n | channel-driven | mixed | distributed-channel |
| --- | ---: | ---: | ---: | ---: |
| Qwen3 | 65 | 36 | 24 | 5 |
| Qwen3.6 | 104 | 33 | 39 | 32 |
| DeepSeek | 21 | 17 | 4 | 0 |

##### Qwen3

| Expert | 原 \(\rho\) | 去 top 1% channel | 去 top 5% | 去 top 10% | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| L0E35（层内最高） | 2.8486 | 2.8252 (r1) | 2.7760 (r3) | 2.7172 (r4) | distributed-channel |
| L47E2（层内最高） | 1.7072 | 1.6826 (r2) | 1.6381 (r18) | 1.6216 (r38) | channel-driven |
| L37E62（层内最高） | 1.7355 | 1.7291 (r1) | 1.7189 (r1) | 1.7020 (r3) | mixed |
| L41E42（层内最高） | 1.7602 | 1.7499 (r1) | 1.7227 (r1) | 1.7006 (r3) | channel-driven |
| L25E61（层内最高） | 1.7049 | 1.6915 (r1) | 1.6571 (r5) | 1.6405 (r10) | channel-driven |

汇总：channel-driven

##### Qwen3.6

| Expert | 原 \(\rho\) | 去 top 1% channel | 去 top 5% | 去 top 10% | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| L0E25（层内最高） | 9.1011 | 9.5048 (r1) | 11.7884 (r1) | 18.1751 (r1) | distributed-channel |
| L39E200（层内最高） | 2.1748 | 2.1659 (r1) | 2.1772 (r1) | 2.2075 (r1) | distributed-channel |
| L33E75（层内最高） | 1.8326 | 1.8270 (r1) | 1.8138 (r1) | 1.8028 (r1) | mixed |
| L32E123（层内最高） | 1.8063 | 1.7998 (r1) | 1.7928 (r1) | 1.7850 (r1) | mixed |

汇总：mixed

##### DeepSeek

| Expert | 原 \(\rho\) | 去 top 1% channel | 去 top 5% | 去 top 10% | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| L11E49（层内最高） | 1.6265 | 1.6241 (r1) | 1.6153 (r1) | 1.6108 (r1) | channel-driven |
| L10E33（层内最高） | 1.6023 | 1.6010 (r1) | 1.5990 (r4) | 1.5974 (r5) | mixed |
| L8E10（层内最高） | 1.6053 | 1.6016 (r2) | 1.5964 (r10) | 1.5923 (r18) | channel-driven |

汇总：channel-driven

#### I.3.2 Expert → Layer

对每个候选特殊 layer：去掉 top-1 / top 5% / top 10% expert 后，用剩余 expert 权重的 L1/L2 重算 Layer \(\rho\)。
只有相对其余层超过 2σ 的层才进入 majority；其余 top-10% 层标为 `not-outlier`。

##### Qwen3

| Layer | 原 Layer \(\rho\) | 去 top-1 expert | 去 top-5% | 去 top-10% | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| L0 | 1.7409 | 1.7356 | 1.7055 | 1.6831 | distributed |
| L47 | 1.6151 | 1.6143 | 1.6112 | 1.6092 | not-outlier |
| L37 | 1.6129 | 1.6120 | 1.6086 | 1.6063 | not-outlier |
| L41 | 1.6121 | 1.6110 | 1.6076 | 1.6057 | not-outlier |
| L25 | 1.6087 | 1.6080 | 1.6053 | 1.6037 | not-outlier |

汇总：distributed

##### Qwen3.6

| Layer | 原 Layer \(\rho\) | 去 top-1 expert | 去 top-5% | 去 top-10% | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| L0 | 2.7268 | 2.7193 | 2.6590 | 2.6073 | distributed |
| L39 | 1.6236 | 1.6219 | 1.6162 | 1.6125 | not-outlier |
| L33 | 1.6186 | 1.6178 | 1.6134 | 1.6106 | not-outlier |
| L32 | 1.6172 | 1.6165 | 1.6130 | 1.6104 | not-outlier |

汇总：distributed

##### DeepSeek

| Layer | 原 Layer \(\rho\) | 去 top-1 expert | 去 top-5% | 去 top-10% | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| L11 | 1.5933 | 1.5928 | 1.5921 | 1.5915 | distributed |
| L10 | 1.5906 | 1.5904 | 1.5898 | 1.5892 | not-outlier |
| L8 | 1.5902 | 1.5900 | 1.5893 | 1.5887 | not-outlier |

汇总：distributed

### I.4 核心结论

| Model | Layer rank | Expert rank | Channel rank | Channel→Expert | Expert→Layer | HARP 启发 |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3 | 有弱分段但不稳定 | 有稳定 head | 有弱分段但不稳定 | channel-driven | distributed | Layer quantile；Expert 按 rank 分档；Channel 用 top prefix |
| Qwen3.6 | 有弱分段但不稳定 | 有弱分段但不稳定 | 有弱分段但不稳定 | mixed | distributed | Layer quantile；Expert 按 rank 分档；Channel 用 top prefix |
| DeepSeek | 有稳定 head | 有弱分段但不稳定 | 有弱分段但不稳定 | channel-driven | distributed | Layer quantile；Expert 按 rank 分档；Channel 用 top prefix |

读法：

- Qwen3：Layer「有弱分段但不稳定」，Expert「有稳定 head」，Channel「有弱分段但不稳定」。2σ 层：L0=distributed。 Channel→Expert 汇总为 channel-driven。
- Qwen3.6：Layer「有弱分段但不稳定」，Expert「有弱分段但不稳定」，Channel「有弱分段但不稳定」。2σ 层：L0=distributed。 Channel→Expert 汇总为 mixed。
- DeepSeek：Layer「有稳定 head」，Expert「有弱分段但不稳定」，Channel「有弱分段但不稳定」。2σ 层：L11=distributed。 Channel→Expert 汇总为 channel-driven。

分数来自 `_artifacts/harp/<model>/harp_rankings.pt`，分析前 \(\exp\) 成 \(\rho\)。
明细：`_artifacts/harp_rank_analysis_ratio/<model>/core_questions.json`。

## Run ID

目录：`/data/xinpeigao/evalscope_results/{MODEL}_{25|50}_vllm_CalibrationFree_full8_v1_{TOKEN}_{TS}_42`

模型目录名：`Qwen330BA3BInstruct`、`Qwen3.6-35B-A3B`、`DeepSeek-V2-Lite-Chat`。


| Method                  | token                    | timestamp    |
| ----------------------- | ------------------------ | ------------ |
| HARP-nologbest          | `HARPNoLogBest`          | 202609162249 |
| HSP \(\Delta=128\)      | `HSPDelta128`            | 202609131611（Qwen3.6 50%：`202609130330`） |
| HARP-nologbest 低分重跑     | `HARPNoLogBest`          | 202609172126–202609172131（D 节，未替换主跑） |
| HARP-nologbest-wanda    | `HARPNoLogBestWanda`     | 202609162324 |
| HARP-nologbest-fuse         | `HARPNoLogBestFuse`         | 202609170046 |
| HARP-nologbest-wanda-cfuse  | `HARPNoLogBestWandaCFuse`   | 202609171243 |
| HARP-nologbest-enp          | `HARPNoLogBestENP`          | 202609170046 |
| HARP-nologbest-enpfuse      | `HARPNoLogBestENPFuse`      | 202609171226 |
| HARP-nologbest-enp-cfuse    | `HARPNoLogBestENPCFuse`     | 202609171243 |
| HARP-nologbest-rand         | `HARPNoLogBestRand`         | 202609170158 |
| AIMER（原版 Expert）        | `AIMER`                    | 202608300140 |


C 节 AIMER 路径与 `scripts/csp_hsp_rerun_sources.json` / `Results_csp_hsp.md` 一致，不混入 A / B。

D 节重跑时间戳：`202609172126`–`202609172131`（六轮），方法 token 仍是 `HARPNoLogBest`，目录与主跑分开。

E 节 C4 / MetaMath / Code-Alpaca 目录：`/data/xinpeigao/evalscope_results/{MODEL}_{25|50}_vllm_{C4128x2048|MetaMath128x2048|CodeAlpaca128x2048}_full8_v1_{TOKEN}_202609180117_42`。

HARP token：`HARPNoLogBestWanda`、`HARPNoLogBestWandaCFuse`、`HARPNoLogBestENP`、`HARPNoLogBestENPCFuse`。同构 token：`Wanda`、`ENP`。WikiText 同构见 `Results_HARP.md`（`WikiText128x2048`）。

F 节 \(N_e\) 来自上述 `_artifacts/{wanda,enp}[_c4|_metamath|_codealpaca]/<model>/statistics.pt` 的 `route_counts`。

G.1/G.2 的 \(K_{\ell,e}\) 来自 `_artifacts/harp_score_ablation/HARPNoLogBest/<model>/v2_rank_adaptive_{25|50}pct.pt`；\(p_{\ell,e}\) 与 F 节同一套 `route_counts`。

G.3 full8 generate：HARP checkpoint `HARPNoLogBest/<model>/checkpoint_{25|50}`，HSP checkpoint `hsp_ablation/HSPDelta128/<model>/checkpoint_{25|50}`；\(p_{\ell,e}\) 来自该 checkpoint 上对 31382 条 full8 样本的 greedy routed-expert dump。
