# RWKV 系 Recurrent LLM における State-Level Screening
## Slot-Based Absolute Relevance Read/Write: Optimized Research Draft

**版**: optimized-draft-v3  
**対象**: RWKV-7 系、または固定サイズ recurrent state を持つ efficient sequence model  
**実装対応**: 本リポジトリの `rwkv7m` JAX/Flax reference implementation  
**主張の強さ**: 本稿は研究仮説であり、性能改善や memory hygiene は実験で検証されるべきである。

---

## 1. Abstract

本稿は、RWKV 系 recurrent language model の compressed state に、Multiscreen 的な absolute relevance screening を移植する設計を提案する。

標準 softmax attention は候補集合上で相対重みを作るため、全候補が無関係でも総和 1 の重みが必ず割り当てられる。これに対して state-level screening は、固定個数の state slot を独立に評価し、閾値を超えた slot だけを read / write する。

中心仮説は次である。

> RWKV 系の固定サイズ recurrent state を slot 化し、absolute relevance によって read / write を制御すれば、token-to-token attention を復活させずに、長文 recall、無関係 state 読み出し抑制、長期 memory contamination 低減を改善できる可能性がある。

この仮説は、PPL だけでなく associative recall、long-context retrieval、long-form consistency、slot-level causal intervention によって検証される必要がある。

---

## 2. Background

### 2.1 Softmax Attention の相対配分性

Transformer は attention による sequence transduction を導入した。標準 attention は query に対し key 集合の相対重みを softmax で作る。この仕組みは候補内比較には強いが、「読むべき候補が存在しない」という状態を表現しにくい。

Multiscreen はこの問題に対し、bounded similarity と明示的閾値による screening を提案した。候補ごとに独立した採否判定を行い、softmax の sum-to-one 制約から離れる。

### 2.2 RWKV-7 と State-Based Memory

RWKV-7 “Goose” は固定サイズ state を用いる recurrent / state-based sequence model である。token 履歴を明示保存するのではなく、state evolution により情報を圧縮する。

したがって RWKV 系の memory 問題は、「どの token を見るか」ではなく、「現在の compressed state のうち何を読み、どこへ書くか」として定式化できる。

### 2.3 Efficient Model の Recall 問題

Mamba、RetNet、DeltaNet 系の研究は、attention 以外の長系列 modeling において state propagation、selective update、in-context retrieval が重要であることを示している。Zoology、Lost in the Middle、RULER も、長文 context 能力を PPL だけで評価する危うさを示している。

本設計はこの流れに沿い、RWKV 系 state access を slot 単位で観測・制御可能にする。

---

## 3. Proposed Method

### 3.1 Slot State

各 screened layer `l` に固定個数 `M` の slot bank を持たせる。

```text
S_l = {s_1, ..., s_M}
s_m in R^{d_slot}
```

`M` は context length に依存しない。追加 memory は `O(M * d_slot)` である。

### 3.2 Read Query / Slot Key / Slot Value

現在 hidden state を `x_t`、RWKV core 出力を `h_base_t` とする。

```text
q_r = W_q_r LN(x_t)
k_m = W_k_r s_m
v_m = W_v s_m
```

query/key/value は必要に応じて `float32` へ上げ、query/key は unit norm にする。

```text
sim_m = <unit(q_r), unit(k_m)> in [-1, 1]
```

### 3.3 Absolute Relevance

学習可能閾値:

```text
tau = 2 sigmoid(theta) - 1
```

Trim-and-Square:

```text
rel_m = relu((sim_m - tau) / (1 - tau + eps))^2
```

`sim=tau` で 0、`sim=1` で 1 になる。slot 間 softmax は使わない。

### 3.4 Non Sum-To-One Aggregation

```text
z = sum_m rel_m unit(v_m)
u = TanhNorm(z)
```

全 slot が無関係なら `z` と `u` は 0 近傍になりうる。これは softmax attention との差分である。

### 3.5 Residual Fusion

```text
gate = sigmoid(W_g LN(x_t) + b_g)
read_out = W_o u
h_t = h_base_t + lambda_screen * gate * read_out
```

`lambda_screen` は小さく初期化し、backbone を初期学習で破壊しないようにする。

---

## 4. Write Screening

read と write は分離する。

```text
q_w = W_q_w LN([x_t; h_base_t])
k_w_m = W_k_w s_m
rel_w_m = TrimSquare(<unit(q_w), unit(k_w_m)>, tau_w)
```

候補更新:

```text
delta_s_m = tanh(W_delta [LN(x_t); h_base_t; e_m])
```

write update:

```text
s_m <- s_m + mu_bank(m) * rel_w_m * (delta_s_m - s_m)
```

`read_screening_only` phase では write relevance を使わず、slow updater のみを使う。`read_write` phase では `cfg.use_write_screening=True` のときだけ write branch が有効になる。

---

## 5. Multi-Timescale Banks

slot は short / mid / long bank に分ける。

```text
0: short
1: mid
2: long
```

更新率上限:

```text
mu_short_max = 0.05
mu_mid_max   = 0.02
mu_long_max  = 0.005
```

仮説は、long bank を低速更新にすることで一時情報による長期記憶汚染を減らせる、というものである。

---

## 6. Implementation Notes

現在の reference implementation は次を満たす。

- JAX/Flax Linen
- Optax train step
- installable package: `rwkv7m`
- RWKV-LM-V7 compatible `.bin/.idx` data reader and batch sampler
- `read_screening_only` / `read_write` phase
- invalid phase rejection
- config validation
- chunked recurrent state carry in the reference RWKV path
- write branch parameters initialized whenever `use_write_screening=True`
- write screening warm-up via slot identity and a tiny update floor to avoid zero-slot dead starts

現在の実装は研究用 reference path であり、production fused kernels や upstream RWKV-7 checkpoint compatibility は未実装である。

---

## 7. Evaluation Plan

### 7.1 Baselines

最低限、以下を比較する。

1. RWKV-7 baseline
2. parameter-matched RWKV-7
3. FLOPs-matched RWKV-7
4. RWKV-7 + read-screening-only state-level screening
5. RWKV-7 + read/write state-level screening
6. RWKV-7 + multi-timescale bank
7. softmax-over-slots memory
8. random slot read/write control
9. Mamba / RetNet / DeltaNet 系近接規模モデル

### 7.2 Ablations

- unit norm on/off
- value unit norm on/off
- Trim-and-Square vs sigmoid gate
- learnable tau vs fixed tau
- TanhNorm on/off
- read-only vs read/write
- shared read/write score vs separated score
- bank-specific update rate on/off
- long bank update rate ablation
- slot count
- screened layer count

### 7.3 Retrieval Tasks

- passkey retrieval
- Needle-in-a-Haystack variants
- RULER-style multi-needle retrieval
- Multi-Query Associative Recall
- key-value retrieval with distractors
- long-context consistency probes

### 7.4 Long-Form Tasks

- character setting retention
- named entity recurrence accuracy
- chapter summary recall
- contradiction rate
- long dialogue memory retention
- same-name entity confusion
- temporary information contaminating long-term setting

### 7.5 Causal Analysis

Relevance visualization alone is not enough. Required interventions:

- slot ablation
- slot patching
- read relevance shuffle
- write suppression
- long bank freeze
- short bank freeze
- logits/generation delta after intervention

---

## 8. Acceptance Criteria

The design should be considered useful only if it shows:

1. retrieval improvement over matched baselines,
2. long-context recall improvement beyond PPL,
3. near-zero read-out in all-irrelevant conditions,
4. less irrelevant memory read than softmax-over-slots,
5. lower long-bank unnecessary update under write screening,
6. causal contribution from at least some slots,
7. stable training with TanhNorm,
8. manageable dead-slot / slot-collapse behavior.

---

## 9. Risks

1. Slot specialization is not guaranteed.
2. High tau can starve memory use.
3. PPL and recall may diverge.
4. Naive slot computation does not automatically improve wall-clock latency.
5. State-level screening may help observability without giving human-interpretable slots.
6. Results from token-level Multiscreen do not directly transfer to RWKV state slots.

---

## 10. References

- Ken M. Nakanishi, *Screening Is Enough*, arXiv:2604.01178, 2026.
- Bo Peng et al., *RWKV-7 “Goose” with Expressive Dynamic State Evolution*, arXiv:2503.14456, 2025.
- Ashish Vaswani et al., *Attention Is All You Need*, arXiv:1706.03762, 2017.
- Albert Gu and Tri Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, arXiv:2312.00752, 2023.
- Yutao Sun et al., *Retentive Network: A Successor to Transformer for Large Language Models*, arXiv:2307.08621, 2023.
- Songlin Yang et al., *Parallelizing Linear Transformers with the Delta Rule over Sequence Length*, arXiv:2406.06484, 2024.
- Simran Arora et al., *Zoology: Measuring and Improving Recall in Efficient Language Models*, arXiv:2312.04927, 2023.
- Nelson F. Liu et al., *Lost in the Middle: How Language Models Use Long Contexts*, arXiv:2307.03172, 2023.
- Cheng-Ping Hsieh et al., *RULER: What’s the Real Context Size of Your Long-Context Language Models?*, arXiv:2404.06654, 2024.
