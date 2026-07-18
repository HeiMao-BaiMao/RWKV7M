# RWKV 系 Recurrent LLM における State-Level Screening
## Confidence-Preserving Competitive Write and Sparse Novel Allocation

**版**: design-locked-draft-v4

**対象**: RWKV-7 系、または固定サイズ recurrent state を持つ efficient sequence model

**実装対応**: 本リポジトリの JAX/Flax NNX、portable reference、GPU/TPU Pallas paths。v4機構はopt-in実装済み。TPU v5e-4実機検証済み、GPU v2実機検証は未実施。

**主張の強さ**: 本稿は研究仮説であり、性能改善や memory hygiene は実験で検証されるべきである。

---

## 1. Abstract

本稿は、RWKV 系 recurrent language model の compressed state に、Multiscreen 的な absolute relevance screening を移植する設計を提案する。

標準 softmax attention は候補集合上で相対重みを作るため、全候補が無関係でも総和 1 の重みが必ず割り当てられる。これに対して state-level screening は、固定個数の state slot を独立に評価し、readでは閾値を超えたslotだけを絶対relevanceで集約する。writeでは絶対eligibilityを保持したままeligible slotを競合させ、既存slotと一致しない情報だけを、hard-forward/soft-backward admissionと疎なbank-aware victim routingを通して割り当てる。

中心仮説は次である。

> RWKV coreを主経路として維持したまま固定容量slot memoryを追加し、absolute read relevance、confidence-preserving competitive write、admission-controlled sparse replacementを組み合わせれば、token-to-token attentionを復活させずに長文recallとmemory hygieneを改善できる可能性がある。

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

### 3.5 Value-Space Gate and Residual Fusion

```text
g_v = sigmoid(W_g LN(x_t) + b_g) in R^{d_v}
read_out = W_o (u * g_v)
h_t = h_base_t + lambda_screen * read_out
```

gateを`d_model`空間ではなく`d_v`空間で適用し、`d_model -> d_model` gateを`d_model -> d_v`へ縮小する。既存checkpointとablationのため、旧model-space gateをlegacy modeとして残す。gate activationは`sigmoid`を既定とし、符号付きbranchを調べる場合だけ`tanh(silu(.))`を独立ablationにする。

`lambda_screen`は小さく初期化し、backboneを初期学習で破壊しないようにする。multi-read時の確定案はtile数だけで正規化する。

```text
effective_lambda = softplus(lambda_raw) / sqrt(n_read_tiles)
```

screened layer数による追加除算は既定に含めず、別の深さ方向ablationとする。

---

## 4. Write Screening

read と write は分離する。read は slot ごとの absolute relevance を最後まで維持する。一方 write は、既存情報との一致度を absolute eligibility として判定した後、書き込み対象となった slot だけを競合させる。

### 4.1 Factorized Slot Candidate

旧 candidate projection は `[LN(x_t); h_base_t; e_m]` から slot ごとの候補を直接生成するため、parameter 数が大きい。次期設計では rank `r` の latent 空間で三要素を合成する。

```text
x_latent = W_x LN(x_t)
h_latent = W_h h_base_t
slot_latent_m = W_e e_m
delta_s_m = tanh(W_out silu(x_latent + h_latent + slot_latent_m))
```

候補 rank は `32 / 64 / 128` を比較する。parameter 数と主要 FLOPs が旧 projection より減らない形状では factorization を採用しない。旧 projection は checkpoint 互換と ablation のため残す。

### 4.2 Confidence-Preserving Competitive Routing

write query と slot write key から absolute eligibility を計算する。

```text
q_w = W_q_w LN([x_t; h_base_t])
k_w_m = W_k_w s_m
e_m = TrimSquare(<unit(q_w), unit(k_w_m)>, tau_w)
```

eligible slot 間の相対 route は power normalization で求めるが、総 write 強度は absolute confidence `c=max_m(e_m)` で抑える。

```text
denom = where(sum_j e_j^gamma > 0, sum_j e_j^gamma, 1)
p_m = e_m^gamma / denom
r_matched_m = c * p_m
```

これにより、弱い eligibility を正規化だけで総量 1 の強い write へ増幅することを避ける。全 slot が threshold 以下なら `e_m=0`、`c=0` となり、既存 slot への matched write は発生しない。read relevance にはこの正規化を適用しない。

### 4.3 Admission-Controlled Sparse Novel Allocation

`c` が novelty threshold 未満の token は、既存 slot と一致しない新規候補とみなす。ただし novelty だけでは保存価値を意味しないため、学習可能な admission を通す。noveltyとadmissionはいずれもforwardではhard、backwardではsoft surrogateを使う。

```text
novel_hard = c < novelty_threshold
novel_soft = sigmoid((novelty_threshold - c) / novelty_temperature)
novel_st = novel_soft + stop_gradient(novel_hard - novel_soft)

admission_soft = sigmoid(admission_logit(x_t, h_base_t))
admission_hard = admission_soft >= admission_threshold
admission_st = admission_soft
             + stop_gradient(admission_hard - admission_soft)
```

初期実装は bank と slot を階層的に選ぶ。3-way bank projection は token ごとの short / mid / long route を作り、各 bank 内では age と usage から victim score を作る。age は bank 内で正規化する。

```text
bank_soft = softmax(bank_logit / bank_temperature)
bank_hard = one_hot(argmax(bank_logit))
bank_st = bank_soft + stop_gradient(bank_hard - bank_soft)

slot_logit_m =
    age_weight * normalized_age_m
    - usage_weight * usage_ema_m

slot_soft_b = masked_softmax(slot_logit / temperature, bank=b)
slot_hard_b = one_hot(argmax(slot_logit within bank b))
slot_st_b = slot_soft_b + stop_gradient(slot_hard_b - slot_soft_b)

victim_st_m = sum_b bank_st_b * slot_st_b,m
r_novel_m = novel_st * admission_st * victim_st_m
r_matched_applied_m = (1 - novel_st) * r_matched_m
r_write_m = r_novel_m + r_matched_applied_m
```

forward は不採用tokenを厳密にrejectし、採用時は一つの bank の top-1 victim へ疎に書く。backward は novelty、admission、bank、slot のsoft surrogateを通して勾配を流す。これにより不採用writeはcontentとageを変更しない。novelty surrogateはTrim-and-Squareの微分が非ゼロな領域でconfidence勾配を回復するが、Trim-and-Square自身の完全reject領域は意図どおりゼロ勾配のままである。top-k と quota は独立 ablation とする。

### 4.4 Final Update and Accounting

```text
r_m = where(is_novel, r_novel_m, r_matched_m)
s_m <- s_m + mu_bank(m) * r_m * (delta_s_m - s_m)
```

slot content、age reset、write metrics は raw eligibility ではなく、実際に適用された `r_m` から更新する。age は write されなければ進み、適用 write があるときだけ reset する。allocation に使う usage EMA は absolute read activity から更新し、write eligibility の高さを「利用された」と数えない。multi-readではtile数による増幅を避けるため、slotごとのtile最大read relevanceをusage signalにする。したがって write を reject された token は slot content を変えず、age を reset せず、write count にも入らないが、実際に slot を読んだ場合は read usage へ反映される。

互換性と段階的評価のため、write mode を次のように分離する。

```text
disabled
legacy_unconditional
legacy_threshold
competitive_novel
```

`legacy_unconditional` は現行 `read_screening_only` の slow updater、`legacy_threshold` は現行 `write_rel_floor` を含む read/write 更新を再現する。既存 config / checkpoint を読み込んだ場合は legacy mode へ明示的に写像し、暗黙に次期 routing へ変更しない。

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

novel allocation は Section 4.3 の階層 route で bank-aware にする。bank ごとの write 数、eviction age、usage 分布を必須 metric とし、bank collapse が観測された場合に限って quota または balance regularizer を比較する。

### 5.1 Projected-State Invariant and Deferred Group Update

現在の accelerator recurrence は、slot 本体に加えて read key、value、write key の projected state を保持し、同一の scalar update strength で更新する。これにより、各 projected state が slot projection と整合する。

slot channel だけに group-wise `mu` を導入すると、projected state の更新と一致しなくなり、sequence chunk 境界によって結果が変わる可能性がある。したがって group-wise update は、projected state を再計算するか、projection と可換な group contract を定義するまで保留する。

### 5.2 Training Tape Checkpointing

checkpointを無効にしたaccelerator backwardはtokenごとに6個のFP32 carryを保存する。screened layer、batch、tokenあたりのtape容量は次になる。

```text
4 * M * (d_slot + 2*d_k + d_v + 2) bytes
```

実装済みの`checkpoint_interval = 8 / 16 / 32` pathは、slot、read key、value、write keyの4個のcontent carryを区間境界にだけ保存し、age、usage、applied update strengthの3個のscalar carryをtokenごとに保存する。`C = ceil(T / I) + 1`を境界数とすると追加tape容量は次になる。

```text
4 * B * screened_layers * M
  * (C * (d_slot + 2*d_k + d_v) + 3*T) bytes
```

backwardはcandidateとscalar tapeから区間内contentを逆算し、境界checkpointで累積誤差を打ち切る。逆算は`1 - strength`で除算するため、checkpoint有効時はhalf-life由来rateと`write_rel_floor`を含む最大実効strengthを`0.95`以下にconfig validationで制約し、kernel内clampによる暗黙の勾配変更は行わない。interval 16では、tracked 0.185B presetのbatch 1、512 tokenが約0.74 MiB、7B presetのbatch 1、4,096 token、4 screened layerが約115.44 MiBとなる。これはtapeだけの理論値であり、実デバイスのpeak memory削減量や再計算コストを示すものではない。GPUとTPUは別々のcheckpoint forward/reverse kernel本文を持つ。

### 5.3 Multi-Read Tile

read subspace を増やす場合も総 dimension を固定する。

```text
d_k_tile = Dk_total / n_read_tiles
d_v_tile = Dv_total / n_read_tiles
```

query/key/value は tile ごとの Dense を並べず、一回の batched projection から reshape する。各 tile は独立した threshold、unit normalization、Trim-and-Square、reject-all aggregation を持つ。tile 出力を concat した後、Section 3.5 の value-space gate と output projection を通す。write path は最初の実装では一系統のまま維持する。

---

## 6. Implementation Notes

### 6.1 Implemented Baseline

現在の実装は次を満たす。

- JAX/Flax NNX training/runtime path
- portable projected reference recurrence
- backend-specific GPU/TPU Pallas screening forward/backward
- Linen numerical and conversion reference
- Optax train step
- installable package: `rwkv7m`
- RWKV-LM-V7 compatible `.bin/.idx` data reader and batch sampler
- `read_screening_only` / `read_write` phase
- invalid phase rejection
- config validation
- chunked recurrent state carry in the reference RWKV path
- write branch parameters initialized whenever `use_write_screening=True`
- write screening warm-up via slot identity and a tiny update floor to avoid zero-slot dead starts

accelerator path は dense projection を XLA へ出し、time recurrence を Pallas へ分離している。GPU Triton path は L40S、TPU path は v5e で projected recurrence の forward、gradient、tracked performance gate を通過している。ただし Hopper / Blackwell の Mosaic GPU、TPU pod 規模、multi-slice、production training の実証は未完了である。upstream RWKV-7 checkpoint compatibility も未証明である。

### 6.2 Implemented Opt-In Screening v2

本稿v4で追加した次の要素は、既存configのlegacy mappingを維持したopt-in機能として実装されている。

1. write mode enum と route / admission / allocation metrics,
2. value-space gate,
3. factorized slot candidate,
4. confidence-preserving competitive routing,
5. hard-forward/soft-backward novelty・admission と straight-through sparse bank-aware allocation,
6. Screening training-tape checkpointing,
7. fixed-total-dimension multi-read tile と `1/sqrt(n_read_tiles)` scaling.

各段階はconfigで個別に切り替え可能であり、tracked exampleは`configs/rwkv7m-0.185b-screening-v2.json.example`である。portable referenceと、GPU/TPUそれぞれのPallas kernel本文は、CPU interpret modeでforwardおよびall-input gradient parityを確認している。benchmarkはoutput・gradient・lossの閾値を既定でfail-closedにする。TPU v5e-4では2026-07-16時点のv2について、実機lowering、6出力と17入力gradient parity、`T=128, B=1, M=16, d_slot=128, d_k=d_v=64`のrecurrence測定、4-device model-axis optimizer stepを確認した。Pallas medianはforward `0.572 ms`、forward+backward `4.856 ms`で、同一referenceの`3.874 ms`、`12.380 ms`に対してそれぞれ6.77倍、2.55倍であった。ただしこの実機記録はhard-forward/soft-backward novelty・admission修正前であり、修正後のreal-TPU lowering/parityは再検証を要する。これはpeak memory、完全0.185B train step、model quality、GPU v2の証拠でもない。group-wise slot updateはSection 5.1のinvariantを満たす再設計まで対象外とする。

---

## 7. Evaluation Plan

### 7.1 Baselines

最低限、以下を比較する。

1. RWKV-7 baseline
2. parameter-matched RWKV-7
3. FLOPs-matched RWKV-7
4. 現行 `legacy_unconditional` State-Level Screening
5. 現行 `legacy_threshold` State-Level Screening
6. v4 各改善の単独追加
7. v4 全改善版
8. parameter-matched FFN branch
9. compute-matched FFN branch
10. softmax-over-slots memory
11. random slot read/write control
12. Mamba / RetNet / DeltaNet 系近接規模モデル

parameter 数だけでなく、学習 token 数と wall-clock の双方で比較する。accelerator kernel 単体の優位と model quality の改善を混同しない。

### 7.2 Ablations

- unit norm on/off
- value unit norm on/off
- Trim-and-Square vs sigmoid gate
- learnable tau vs fixed tau
- TanhNorm on/off
- legacy unconditional / legacy threshold / competitive novel
- shared read/write score vs separated score
- model-space gate vs value-space gate
- sigmoid gate vs `tanh(silu(.))`
- factorized candidate rank `32 / 64 / 128` vs legacy candidate
- confidence multiplier on/off
- admission on/off、hard-forward threshold、surrogate temperature
- victim top-1 vs top-k, temperature, straight-through estimator
- fixed / biased / quota-based bank allocation
- bank-specific update rate on/off
- long bank update rate ablation
- tape checkpoint interval `None / 8 / 16 / 32`
- read tile count with fixed total key/value dimensions
- `1/sqrt(n_read_tiles)` residual scaling on/off
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
- admission suppression
- novel allocation suppression
- bank-preserving slot shuffle
- long bank freeze
- short bank freeze
- full memory branch zeroing
- memory reset at controlled token positions
- logits/generation delta after intervention

### 7.6 System and Slot Metrics

quality 指標に加えて、次を記録する。

- train tokens/s、step latency、peak accelerator memory
- loss vs training tokens、loss vs wall-clock
- route mass、route entropy、top-1 concentration
- admission mean / saturation、novel-token rate
- bank ごとの matched write、allocation、eviction 数
- slot utilization、dead-slot rate、duplicate / cosine redundancy
- eviction 時の age と usage
- memory branch を無効化した counterfactual delta

主結果は複数 seed で報告し、単一 run の route visualization を一般化しない。

---

## 8. Acceptance Criteria

実装受け入れには、少なくとも次の invariant が必要である。

1. `legacy_unconditional` が現行結果を許容誤差内で再現する,
2. sequence chunk size を変えても同一 write mode の意味論が変わらない,
3. 弱い eligibility が route normalization により強い write へ増幅されない,
4. admission / routing で不採用となった token が slot content を更新せず、age reset や write count を発生させない,
5. forward、gradient、checkpoint-resume parity が reference と accelerator path で成立する。

研究上有用と判断するには、さらに次を満たす必要がある。

1. matched baseline に対する retrieval improvement,
2. PPL だけでは説明できない long-context recall improvement,
3. all-irrelevant 条件で near-zero read-out,
4. softmax-over-slots より少ない irrelevant memory read,
5. long bank の不要な更新と重複 slot の減少,
6. 少なくとも一部 slot の causal contribution,
7. TanhNorm と sparse routing を含む stable training,
8. manageable な dead-slot、route-collapse、bank-collapse behavior,
9. parameter / compute 増加を含めても許容できる loss-vs-wall-clock と peak memory.

---

## 9. Risks

1. Slot specialization is not guaranteed.
2. High tau can starve memory use.
3. PPL and recall may diverge.
4. Naive slot computation does not automatically improve wall-clock latency.
5. State-level screening may help observability without giving human-interpretable slots.
6. Results from token-level Multiscreen do not directly transfer to RWKV state slots.
7. Straight-through routing introduces biased gradients and may become temperature-sensitive.
8. Admission can saturate to always-write or never-write without monitoring or regularization.
9. Sparse victim selection can collapse onto one slot or one bank.
10. Tape checkpointing lowers memory at the cost of backward recomputation and may regress short-context throughput.
11. Multi-read can duplicate retrieval subspaces or amplify the residual branch despite fixed total dimensions.
12. Group-wise update can violate projected-state consistency; it remains deferred rather than assumed safe.

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
