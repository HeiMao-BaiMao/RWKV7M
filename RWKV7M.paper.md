# RWKV系 Recurrent LLMにおける State-Level Screening

## Capacity-Calibrated Absolute Read, Confidence-Preserving Memory Editing, and Sparse Novel Allocation

**版**: design-locked-draft-v5

**対象**: RWKV-7系、または固定容量recurrent stateを持つefficient sequence model

**基礎実装**: JAX/Flax NNX、portable reference recurrence、v4 GPU/TPU Pallas recurrence

**意味論バージョン**:

```text
screening-v4-legacy
screening-v4-competitive
screening-v5-core
screening-v5-retention
```

**実装状況**:

* v4 legacyおよびcompetitive screeningはopt-in実装済み
* v4にはportable referenceとGPU/TPU Pallas pathが存在する
* 補正後v4 recurrenceはTPU v5e-4で実機parityおよび性能測定済み
* corrected v4 competitive GPU pathの実機検証は未実施
* v5 coreのPhase 1 portable referenceおよびNNX統合は実装済みである
* portable v5 coreはMI300X単基で0.185Bの50-step finite runとpost-training gradient gateを通過したが、0.3Bはrun間でNaNが再現せず、数値再現性は未確立である
* tracked v5 recovery configは`tied` edit、redundancy-aware victim、soft-to-hard read、temporary self-index loss、上下write curriculumを用いる
* v5 Pallas、v5 checkpoint redesign、v5 retentionは未実装である
* warm-up限定admission floor、soft read、self-index loss、上限write budgetは実装済みだが、有効性は未実証である
* 短時間MI300X runではmemory residualのcollapseと高いslot redundancyが観測され、v5のmodel quality改善は未実証である

**主張の強さ**:

本稿は固定容量recurrent memoryの設計仮説および実装契約である。長文recall、memory hygiene、学習効率、wall-clock性能の改善は、matched baselineと複数seedによる実験でのみ主張される。

---

# 1. Abstract

本稿は、RWKV系recurrent language modelに固定容量のslot memoryを追加し、絶対relevanceに基づくread screening、confidence-preserving matched write、admission-controlled novel allocation、および明示的なmemory lifecycle管理を行う設計を提案する。

標準softmax attentionは候補集合内の相対重みを生成するため、すべての候補が無関係でも総和1のread massを割り当てる。一方、State-Level Screeningは各slotを独立に評価し、閾値以下のslotを完全にrejectできる。

ただし、absolute screeningを固定容量memoryへ単純に適用するだけでは不十分である。従来設計には少なくとも以下の根本問題が存在する。

1. novelty thresholdがslot数とkey次元に依存する
2. 空slotと低品質slotの意味論が分離されていない
3. matched writeとnovel allocationのwrite amplitudeが不連続になり得る
4. hard address selectionと書込み量が結合している
5. 複数slotへの曖昧な一致が強いmatched writeへ変換され得る
6. read relevanceの単純和がslot数に応じて増幅または飽和する
7. admissionおよびvictim selectionが明示的なcapacity objectiveから導出されていない
8. straight-through estimatorの実装parityと学習上の妥当性が区別されていない
9. checkpoint reverse recurrenceの代数的可逆性と数値安定性が混同されている
10. backend、意味論、benchmark provenanceがpaper-gradeに固定されていない

v5では、State-Level Screeningを固定容量のbounded online dictionaryとして再定式化し、次の機構を導入する。

* occupied slotのみを対象とするcapacity-calibrated screening
* explicit occupancy state
* bounded、non-amplifying read aggregation
* ambiguity-aware matched-write confidence
* hard addressとcontinuous write amplitudeの分離
* empty-first allocation
* age、usage、redundancy、将来利用価値を考慮するvictim policy
* interval-aware checkpoint stability budget
* versioned semantic golden vectors
* optional adaptive retentionおよびerase/write分離

中心仮説は次である。

> RWKV coreを主経路として維持しながら、固定容量memoryを絶対relevance、容量校正、疎なaddressing、連続的なmemory editingによって管理すれば、token-to-token attentionを復活させずに、irrelevant read、memory collision、stale retrieval、破壊的なslot replacementを減らせる可能性がある。

---

# 2. Scope and Non-Claims

本稿が対象とするのは、context全体を保存する外部databaseではなく、各screened layerに付加される小規模なpersistent differentiable memoryである。

```text
number of slots M:
    context lengthに依存しない

slot dimension d_slot:
    model configurationで固定

inference memory:
    O(M * d_slot)

token recurrence:
    O(M) または tiled O(M)
```

本設計は以下を自動的には保証しない。

* slotの人間可読な意味分解
* context lengthに比例する情報容量
* attentionと同等のexact retrieval
* hard routingのunbiased gradient
* wall-clock speedup
* long-context quality改善
* upstream RWKV checkpoint互換性
* production-grade multi-host scaling
* KDAまたはDeltaNetと同等の学習挙動

slotは観測可能なmemory unitではあるが、symbolic memoryまたはhuman-interpretable entity storeとはみなさない。

---

# 3. Memory as a Bounded Online Dictionary

## 3.1 State

screened layer (l) は、最大 (M) 個のslotからなるmemoryを持つ。

```text
S_t = {s_t,1, ..., s_t,M}
s_t,m in R^{d_slot}
```

各slotには次のmetadataを持たせる。

```text
o_t,m:
    occupancy bit

age_t,m:
    最後の適用writeからの経過clock

usage_t,m:
    absolute read activityのEMA

bank_m:
    short / mid / long policy class
```

retention profileでは追加で、

```text
vitality_t,m in [0, 1]
```

を持つ。

persistent projected stateは、

```text
k_read_t,m
k_write_t,m
v_t,m
```

である。

## 3.2 Online Decision Variables

各tokenでmemory policyは次を決める。

```text
read relevance:
    r_read_t,m

matched-write route:
    r_match_t,m

novel decision:
    n_t in {0, 1}

admission decision:
    a_t in {0, 1}

victim address:
    q_t,m in {0, 1}

erase mass:
    e_t,m in [0, 1]

write mass:
    w_t,m in [0, 1]
```

hard decisionは原則としてaddressに限定する。

```text
hard:
    novelか
    保存するか
    どのslotへ書くか

continuous:
    どれだけ消すか
    どれだけ書くか
```

## 3.3 Constrained Objective Interpretation

本設計は、次の制約付きonline memory problemを近似するamortized policyとして解釈する。

```text
minimize

    expected language-model loss
  + lambda_read      * irrelevant read cost
  + lambda_write     * write frequency
  + lambda_collision * memory collision
  + lambda_duplicate * slot redundancy
  + lambda_eviction  * future eviction regret
  + lambda_stability * recurrent-state instability

subject to

    number of occupied slots <= M
    read may reject all slots
    rejected writes do not alter state
    hard allocation changes at most top-k slots
    recurrent state is chunk-invariant
```

本稿は、全routing ruleがこの目的関数の厳密解であるとは主張しない。

重要なのは、各heuristicを独立した設計選択ではなく、

```text
retrieval
collision avoidance
write cost
capacity allocation
state stability
```

の近似として位置づけることである。

---

# 4. Slot Occupancy and Initialization

## 4.1 Explicit Occupancy

各slotにhard occupancy stateを持たせる。

```text
o_t,m in {0, 1}
```

初期状態は、

```text
o_0,m = 0
s_0,m = 0
k_read_0,m = 0
k_write_0,m = 0
v_0,m = 0
age_0,m = 0
usage_0,m = 0
```

とする。

空slotはreadおよびmatched writeの候補にならない。

```text
read_rel_t,m *= o_{t-1,m}
write_eligibility_t,m *= o_{t-1,m}
```

これにより、zero slotとの偶然一致、unit normalizationの不定性、warm-up用tiny update floorへの依存を減らす。

## 4.2 Empty-First Allocation

novel writeでは、occupied slotをevictする前にempty slotを使用する。

```text
if any empty slot exists in selected bank:
    choose empty slot
else:
    choose occupied victim
```

empty allocationはevictionとして数えない。

```text
allocation_count += 1
eviction_count += occupied_before_write
```

empty slotへの初回write後、

```text
o_t,m = 1
age_t,m = 0
usage_t,m = 0
```

とする。

## 4.3 Deterministic Tie-Breaking

hard argmaxが同値の場合、backend間で結果が変わらないよう、tie-break contractを固定する。

既定は、

```text
lowest slot index wins
```

とする。

random tie-breakingは独立ablationとし、semantic parity testでは使用しない。

---

# 5. Absolute Read Screening

## 5.1 Query, Key, and Value

現在hidden stateを (x_t)、RWKV core出力を (h^{base}_t) とする。

```text
q_read_t = W_q_read LN(x_t)

k_read_t,m = persistent read-key state
v_t,m      = persistent value state
```

queryとkeyはfloat32でunit normalizationする。

```text
q_hat_t = q_read_t / max(||q_read_t||_2, eps_norm)
k_hat_t,m = k_read_t,m / max(||k_read_t,m||_2, eps_norm)

sim_t,m = <q_hat_t, k_hat_t,m>
```

empty slotでは、

```text
sim_t,m = -infinity
```

として扱う。

## 5.2 Safe Threshold Parameterization

従来の、

```text
tau = 2 sigmoid(theta) - 1
```

は、(\tau \to 1) のときTrim-and-Squareの分母および勾配を不安定にする。

v5では、

```text
tau =
    tau_min
    + (tau_max - tau_min) * sigmoid(theta)

-1 < tau_min < tau_max < 1
```

とする。

`tau_max`は必ず、

```text
tau_max <= 1 - delta_tau
```

を満たす。

`eps`だけに依存して(\tau \to 1)を許可しない。

## 5.3 Capacity-Calibrated Threshold

slotごとのfalse positive rateが一定でも、slot数が増えると「少なくとも一つの無関係slotを読む確率」は増加する。

そこで、read thresholdは、

```text
number of occupied slots
key dimension
number of read tiles
target family-wise false-read rate
```

に対して校正する。

null similarityのCDFを $F_{d_{\mathrm{test}}}$ とすると、analytic initializationは、

$$
\tau^{\mathrm{base}}_{\mathrm{read}}
=
F_{d_{\mathrm{test}}}^{-1}
\left(
(1-\delta_{\mathrm{read}})^{1/N_{\mathrm{tests}}}
\right)
$$

とする。

このfamily-wise式はnull similarity testの独立性を仮定する。tile間またはslot間に
相関がある場合は初期化近似としてのみ使用し、empirical false-read curveを
最終calibrationの根拠とする。

```text
N_tests =
    max(1, occupied_slot_count * n_read_tiles)
```

一回のsimilarity testに使うkey次元を $d_{\mathrm{test}}$ とする。single-readでは
$d_{\mathrm{test}}=d_k$、multi-readでは
$d_{\mathrm{test}}=d_{k,\mathrm{tile}}$ である。高次元unit vectorのGaussian null
近似では、family-wise CDFを

$$
q = (1-\delta_{\mathrm{read}})^{1/N_{\mathrm{tests}}}
$$

として、

$$
\tau^{\mathrm{base}}_{\mathrm{read}}
\approx
\frac{\Phi^{-1}(q)}{\sqrt{d_{\mathrm{test}}}}
$$

を用いる。より緩い上界として、

$$
\tau^{\mathrm{base}}_{\mathrm{read}}
\approx
\sqrt{
\frac{
2\log(N_{\mathrm{tests}}/\delta_{\mathrm{read}})
}{
d_{\mathrm{test}}
}
}
$$

も書けるが、小さいtile次元と多いtest数では過度に保守的になるため、実装の
既定値には用いない。実際、$N_{\mathrm{tests}}=64$、$d_{\mathrm{test}}=16$、
$\delta=0.05$では上界が0.946、Gaussian CDF近似が0.789となる。前者はtracked
MI300X runのread starvationを直接誘発し得る値だった。

ただし、学習後のkey分布は等方的ではないため、analytic thresholdを保証として扱わない。

実装では、

```text
tau_read =
    clip(
        tau_base(M_occupied, d_test, n_tiles)
        + learned_tau_offset,
        tau_min,
        tau_max
    )
```

とする。

headline configurationでは、固定thresholdだけでなく、shuffled query-key pairまたはknown-nonmatch pairによるempirical false-read curveを報告する。

## 5.4 Trim-and-Square

absolute relevanceは、

```text
trim_t,m =
    relu(
        (sim_t,m - tau_read)
        / (1 - tau_read)
    )

rel_raw_t,m =
    trim_t,m^2
```

とする。

```text
sim <= tau:
    relevance = 0

sim = 1:
    relevance = 1
```

read relevanceはslot間softmaxで正規化しない。

## 5.5 Read Dead-Zone

Trim-and-Squareは閾値以下でquery、key、thresholdへの勾配をゼロにする。

これはhard rejectionの意味論としては正しいが、初期学習で全slotがrejectされるとmemory branchが学習できない。

v5では、以下のいずれかを明示的に選択する。

```text
hard_read:
    forwardもbackwardもTrim-and-Square

shadow_surrogate:
    forwardはTrim-and-Square
    backwardはsmooth surrogate

threshold_warmup:
    初期tauを低くし、学習中に校正値へ移行
```

初期設計の既定は、

```text
hard_read + threshold_warmup
```

だった。しかしMI300X短時間runでhard readが消失したため、tracked recovery
profileではtraining中だけ、

```text
smooth forward / smooth backward
    -> anneal
hard forward / hard backward
```

へ移行する。smooth relevanceは正規化座標
$x=(sim-\tau)/(1-\tau)$に対するsquared softplusとし、slot間で総和1へ正規化
しない。deterministic evaluationとinferenceは常にhard readを使う。この
curriculumはtraining semanticsを一時的に変更するため、hard-only ablationと
区別して報告する。

shadow surrogateは独立ablationとし、hard readと同じモデルとして報告しない。

## 5.6 Occupancy and Optional Vitality

v5 coreでは、

```text
rel_t,m =
    o_{t-1,m} * rel_raw_t,m
```

とする。

retention profileでは、

```text
rel_t,m =
    o_{t-1,m}
    * vitality_pre_t,m^rho_read
    * rel_raw_t,m
```

とする。

vitalityはrelevanceを増幅せず、抑制方向にのみ作用する。

## 5.7 Bounded Non-Amplifying Aggregation

単純な、

```text
sum_m rel_m unit(v_m)
```

は、関連slot数が増えたときvector normが増大し、TanhNormやresidual branchを飽和させ得る。

一方、通常の平均やsoftmax normalizationは、弱い総relevanceを強いreadへ増幅し得る。

v5では、弱いread vectorを増幅せず、合成後のvector normが1を超える場合だけ抑える。

```text
value_hat_t,m =
    v_t,m / max(||v_t,m||_2, eps_norm)

z_sum_t =
    sum_m rel_t,m * value_hat_t,m

read_energy_t =
    ||z_sum_t||_2^2

read_denom_t =
    sqrt(
        1 + relu(read_energy_t - 1)
    )

z_t =
    z_sum_t / read_denom_t
```

このとき、

```text
read_energy <= 1:
    normalizationなし

read_energy > 1:
    ||z_t||_2 = 1
```

となる。

これはslot probability normalizationではなく、合成vectorに対するnorm clippingである。
したがって、value vectorの向きが揃っている場合を含め、slot数に依存せず
`||z_t||_2 <= 1` が成立する。

全slotがrejectされた場合、

```text
z_t = 0
```

を維持する。

## 5.8 TanhNorm

必要に応じて、

$$
\operatorname{TanhNorm}(z)
=
\tanh(\lVert z\rVert_2)
\frac{z}{\lVert z\rVert_2+\epsilon}
$$

を適用する。

```text
u_t = TanhNorm(z_t)
```

TanhNormの有無はablationとする。

## 5.9 Value-Space Gate and Residual Fusion

```text
g_value_t =
    sigmoid(
        W_gate LN(x_t) + b_gate
    )

read_out_t =
    W_out(
        u_t * g_value_t
    )
```

screening residualは、

```text
h_t =
    h_base_t
    + lambda_effective * read_out_t
```

とする。

multi-read時は、

```text
lambda_effective =
    softplus(lambda_raw)
    / sqrt(n_read_tiles)
```

とする。

screened layer数による一律除算は既定にしない。

代わりに、各layerについて次を記録する。

```text
screening residual RMS
base residual RMS
screening/base RMS ratio
gradient RMS
```

全screened layerの残差エネルギーが増大する場合のみ、depth scalingまたはresidual budget regularizationを比較する。

---

# 6. Write Candidate

## 6.1 Factorized Candidate

candidateはslotごとに生成する。

```text
x_latent =
    W_x LN(x_t)

h_latent =
    W_h h_base_t

slot_latent_m =
    W_e e_m

delta_s_t,m =
    tanh(
        W_candidate_out
        silu(
            x_latent
            + h_latent
            + slot_latent_m
        )
    )
```

rankは、

```text
32 / 64 / 128
```

を比較する。

## 6.2 Projection Consistency

persistent projected candidateは、同じcandidate slotから計算する。

```text
delta_k_read_t,m =
    P_k_read delta_s_t,m

delta_k_write_t,m =
    P_k_write delta_s_t,m

delta_v_t,m =
    P_v delta_s_t,m
```

slot candidateとprojected candidateを独立networkで生成してはならない。

これにより、

```text
persistent projected state
=
projection of recurrent slot state
```

という意味論を維持する。

## 6.3 Parameter and FLOP Contract

factorizationを採用する条件は、parameter数だけでなく、主要MACsも減少することである。

報告する値は次とする。

```text
parameter count
theoretical MACs
activation bytes
parameter bytes
projection latency
end-to-end step latency
```

FLOPs削減だけでwall-clock改善を主張しない。

rankごとに、

```text
legacy candidate
factorized rank 32
factorized rank 64
factorized rank 128
```

を同一shapeで比較する。

---

# 7. Capacity-Calibrated Write Matching

## 7.1 Separate Write Query

read queryとwrite queryは共有しない。

```text
q_write_t =
    W_q_write
    LN(
        concat(
            x_t,
            h_base_t
        )
    )
```

write keyは、

```text
k_write_t,m
```

を使用する。

```text
sim_write_t,m =
    <unit(q_write_t), unit(k_write_t,m)>
```

empty slotはmatched write候補から除外する。

## 7.2 Absolute Eligibility

```text
eligibility_t,m =
    o_{t-1,m}
    * TrimSquare(
        sim_write_t,m,
        tau_write
    )
```

`tau_write`もoccupied slot数およびkey次元に応じて校正する。

read thresholdとwrite thresholdは共有しない。

## 7.3 Competitive Route

eligible slot間のrouteは、

```text
route_power_t,m =
    eligibility_t,m^gamma

route_denom_t =
    where(
        sum_j route_power_t,j > 0,
        sum_j route_power_t,j,
        1
    )

p_t,m =
    route_power_t,m
    / route_denom_t
```

とする。

ただし、`p`はaddress distributionであり、総write massではない。

## 7.4 Ambiguity-Aware Confidence

従来の、

```text
confidence =
    max_m eligibility_m
```

だけでは、複数の類似slotが存在する曖昧な状態でも強いmatched writeが発生し得る。

v5ではroute concentrationを考慮する。

```text
c_abs_t =
    max_m eligibility_t,m

route_concentration_t =
    sum_m p_t,m^2
```

`route_concentration`は、

```text
one clear slot:
    1に近い

K個へ均等:
    1/Kに近い
```

となる。

effective matched confidenceは、

```text
if c_abs_t == 0:
    c_match_t = 0

elif eta_ambiguity == 0:
    c_match_t = c_abs_t

else:
    c_match_t =
        c_abs_t
        * route_concentration_t^eta_ambiguity
```

とする。

この分岐により、eligible slotがない場合の`0^0`を意味論から排除する。

これにより、

```text
weak match
diffuse match
duplicate-slot ambiguity
```

が強いwriteへ変換されることを抑制する。

`eta_ambiguity=0`でv4 behaviorへ戻る。

## 7.5 Capacity-Calibrated Novelty

noveltyは固定thresholdではなく、occupied slot数およびkey次元を考慮して判定する。

```text
novel_hard_t =
    c_match_t
    < tau_novel(
        M_occupied,
        d_k,
        target_false_match_rate
    )
```

soft surrogateは、

```text
novel_soft_t =
    sigmoid(
        (
            tau_novel
            - c_match_t
        )
        / novelty_temperature
    )
```

とする。

straight-through形式は、

```text
novel_st_t =
    novel_soft_t
    + stop_gradient(
        novel_hard_t
        - novel_soft_t
    )
```

とする。

固定thresholdはlegacy modeとして残す。

headline v5 configurationでは、少なくとも、

```text
M = 16 / 64 / 256
d_k = 32 / 64 / 128
```

におけるfalse-match curveを報告する。

---

# 8. Admission

## 8.1 Novelty Is Not Utility

既存slotと一致しないことは、保存価値を意味しない。

novel tokenはadmission policyを通す。

```text
admission_soft_t =
    sigmoid(
        admission_logit(
            x_t,
            h_base_t,
            c_match_t,
            memory_load,
            bank_load
        )
    )
```

```text
admission_hard_t =
    admission_soft_t
    >= admission_threshold
```

```text
admission_st_t =
    admission_soft_t
    + stop_gradient(
        admission_hard_t
        - admission_soft_t
    )
```

## 8.2 Write-Budget Regularization

admissionが常に1へ飽和することを防ぐため、write budgetを導入する。

```text
write_rate =
    mean_t(
        admission_soft_t
        * novel_soft_t
    )
```

上限budgetのみを課す場合、

```text
L_write_budget =
    lambda_budget
    * relu(
        write_rate
        - target_max_write_rate
    )^2
```

とする。

固定target write rateを最適値として主張しない。

以下をablationする。

```text
no budget
upper-bound budget
learned dual budget
bank-specific budget
```

never-write collapseは、

```text
novel rate
admission mean
accepted novel rate
memory branch causal delta
```

で検出する。

## 8.3 Warm-up-Limited Admission Floor

上限write budgetはalways-writeを抑制するが、never-writeには罰則を与えない。
Phase 1のopt-in curriculumとして、空容量が残る初期期間だけsoft admissionへ下限を置く。

```text
remaining_empty_fraction_t =
    1 - mean_m(o_t,m)

target_min_rate(step) =
    initial_min_rate
    * max(
        1 - step / admission_floor_steps,
        0
      )
    * stop_gradient(
        remaining_empty_fraction_t
      )
```

```text
L_admission_floor =
    lambda_floor
    * relu(
        target_min_rate(step)
        - mean_t(
            novel_soft_t
            * admission_soft_t
          )
      )^2
```

この補助損失は、warm-up終了後またはmemory満杯時に厳密に0とする。恒久的なwrite quota、
固定最適write rate、memory利用の証拠として扱わない。headline比較では、floorなし、floorあり、
およびmemory branch counterfactualを分けて報告する。

同じwarm-up区間では、`lambda_screen`にもannealする非負下限を設定できる。これは初期に
residual scaleだけを0へ落とす退化解を抑えるためのcurriculumであり、warm-up終了後はlearned
scaleだけを使用する。gate biasは飽和初期化せず、read RMS、base RMS、両者の比、learned scale、
適用中のfloorを記録する。

## 8.4 Temporary Self-Index Curriculum

empty-first allocationだけでは、書き込まれたcandidate keyが、そのwriteを
発生させたqueryから再検索可能であることを保証しない。query/key geometryが
未学習のままでは、全tokenがnovelとなりslot置換だけが継続し得る。

新規writeをhard-forwardで受理したtokenについて、選択されたcandidateの
write keyとread keyへ一時的なmargin lossを課す。

```text
target_write = tau_novel_similarity + margin
target_read_tile = tau_read_tile + margin

L_self_index =
    relu(target_write - sim(q_write, candidate_write_key))^2
    + mean_tile(
        relu(target_read_tile - sim(q_read_tile, candidate_read_key_tile))^2
      )
```

admission、bank、victim addressはstop-gradientしたhard decisionとして扱い、
query/key/candidate projectionへだけ幾何学習信号を流す。これによりloss回避の
ためにadmissionを下げる経路を作らない。係数はwarm-up中にゼロへannealし、
恒久的な同一query再構成目的にはしない。

tracked recovery profileは同時にupper write budgetを有効化する。lower floorは
never-writeだけ、upper budgetはall-novel/all-writeだけを抑えるため、両者を
単一の固定write率として解釈しない。

---

# 9. Hard Address and Continuous Memory Editing

## 9.1 Address and Amplitude Separation

v5では、hard routingを離散decisionに限定し、accepted writeのcontent editing量は
continuousに保つ。

```text
hard:
    matchedかnovelか
    admissionするか
    selected bank
    selected allocation slot

continuous:
    erase mass
    write mass
```

noveltyまたはadmission境界を越えただけで、accepted writeのerase/write massを
自動的に1へ固定する設計を既定にしない。

## 9.2 Matched Route

matched address massは、

```text
matched_address_t,m =
    (1 - novel_st_t)
    * c_match_t
    * p_t,m
```

とする。

## 9.3 Erase and Write Gates

matched writeの各slotについて、

```text
erase_gate_t,m =
    sigmoid(
        erase_logit_t,m
    )

write_gate_t,m =
    sigmoid(
        write_logit_t,m
    )
```

を計算する。

安定性を優先する既定modeは、

```text
capacity_conserving
```

とする。

```text
erase_mass_t,m =
    base_mass_t,m
    * erase_gate_t,m

write_mass_t,m =
    erase_mass_t,m
    * write_gate_t,m
```

したがって、

```text
0 <= write_mass <= erase_mass <= 1
```

が成立する。

比較modeは次とする。

```text
tied:
    erase_mass = write_mass = base_mass

capacity_conserving:
    write_mass <= erase_mass

free_edit:
    eraseとwriteを独立予測
```

`free_edit`はnorm drift監視なしにheadline configurationへ使用しない。

## 9.4 Matched Update

matched writeでは、

```text
base_mass_t,m =
    mu_bank(m)
    * matched_address_t,m
```

とする。

slot updateは、

```text
s_t,m =
    (
        1 - erase_mass_t,m
    )
    * s_{t-1,m}
    +
    write_mass_t,m
    * delta_s_t,m
```

とする。

projected statesも同じscalarで更新する。

```text
k_read_t,m =
    (
        1 - erase_mass_t,m
    )
    * k_read_{t-1,m}
    +
    write_mass_t,m
    * delta_k_read_t,m
```

```text
k_write_t,m =
    (
        1 - erase_mass_t,m
    )
    * k_write_{t-1,m}
    +
    write_mass_t,m
    * delta_k_write_t,m
```

```text
v_t,m =
    (
        1 - erase_mass_t,m
    )
    * v_{t-1,m}
    +
    write_mass_t,m
    * delta_v_t,m
```

## 9.5 Exact Reduction to the v4 Content Update

次の設定でv4のcontent update recurrenceへ代数的に還元される。

```text
eta_ambiguity = 0
edit_mode = tied
capacity_calibration = fixed
occupancy = all occupied
```

このとき、

```text
erase_mass =
    write_mass =
    mu_bank * matched_route
```

となる。

これはcontent updateの還元条件であり、query/key target、threshold、novelty、admission、
bank routeを含む完全なv4互換性には、それらも`screening-v4-competitive`契約へ固定する必要がある。
`capacity_conserving` modeのsigmoid gate biasを大きい正値にするだけでは近似にすぎず、
exact compatibilityとは呼ばない。

---

# 10. Sparse Novel Allocation

## 10.1 Hierarchical Bank Route

bank routeは、

```text
bank_soft_t =
    softmax(
        bank_logit_t
        / bank_temperature
    )
```

```text
bank_hard_t =
    one_hot(
        argmax(bank_logit_t)
    )
```

```text
bank_st_t =
    bank_soft_t
    + stop_gradient(
        bank_hard_t
        - bank_soft_t
    )
```

とする。

forwardでは1 bankを選択する。

## 10.2 Empty Slot Selection

selected bankにempty slotがある場合、occupied victim scoringを使用しない。

bankごとのempty slot maskは、

```text
empty_mask_b,m =
    1[bank_m == b]
    * (1 - o_{t-1,m})
```

とする。`has_empty_b = any_m(empty_mask_b,m)`の場合、

```text
empty_hard_b =
    one_hot(
        lowest empty slot index in bank b
    )
```

を使用する。empty address自体にはsoft victim scoreを使用しないが、どのbankを選ぶかの
surrogate gradientは`bank_st`を通して維持する。

## 10.3 Victim Score

empty slotがない場合のみvictim scoreを計算する。

```text
victim_score_t,m =
      age_weight
      * normalized_age_t,m

    - usage_weight
      * normalized_usage_t,m

    - vitality_weight
      * vitality_pre_t,m

    - predicted_use_weight
      * predicted_future_use_t,m

    + redundancy_weight
      * redundancy_t,m
```

高いscoreほどeviction候補とする。

v5 coreでvitalityを使用しない場合、

```text
vitality_t,m = 1
```

として項を無効化する。

## 10.4 Redundancy

duplicate slotはcapacityを消費し、曖昧なmatched routeを生む。

slot redundancyは例えば、

```text
redundancy_t,m =
    max_{j != m}
    cosine(
        unit(k_write_t,m),
        unit(k_write_t,j)
    )
```

とする。

(O(M^2))計算を避ける場合、

```text
periodic update
sampled pairs
bank-local pairs
low-rank Gram approximation
```

を用いる。

redundancyはread relevanceではなくvictim protectionにのみ使用することを既定とする。

## 10.5 Predicted Future Use

ageとusageは過去利用のみを表し、将来利用価値を直接表現しない。

optional eviction criticは、

```text
predicted_future_use_t,m =
    sigmoid(
        utility_critic(
            slot_summary_t,m,
            age_t,m,
            usage_t,m,
            bank_m
        )
    )
```

を予測する。

training targetは、slotがoverwriteされるまでの範囲で、

```text
future_read_target_t,m =
    max_{u in [t+1, t+H]}
        stop_gradient(
            rel_u,m
        )
```

とする。

```text
L_eviction_critic =
    binary_cross_entropy(
        predicted_future_use,
        future_read_target
    )
```

このcriticはoptionalであり、v5 coreの成立条件ではない。

ただし、age/usage heuristicのみを最終方式として理論的最適と主張しない。

## 10.6 Empty-First Allocation Route

empty slotがないbankについてのみ、occupied slot間で、

```text
slot_soft_b =
    masked_softmax(
        victim_score
        / victim_temperature,
        bank=b
    )
```

```text
slot_hard_b =
    one_hot(
        argmax(
            victim_score within bank b
        )
    )
```

```text
victim_st_b =
    slot_soft_b
    + stop_gradient(
        slot_hard_b
        - slot_soft_b
    )
```

とする。

bank内allocation addressは、

```text
allocation_st_b =
    where(
        has_empty_b,
        stop_gradient(empty_hard_b),
        victim_st_b
    )
```

とする。最終allocation addressは、

```text
allocation_st_t,m =
    sum_b
        bank_st_b
        * allocation_st_b,m
```

となる。

metadata更新に使うhard-forward addressは、同じ分岐の`bank_hard`、`empty_hard`、
`slot_hard`から`allocation_hard_t,m`として構成する。

## 10.7 Novel Write Amplitude

novel addressはhardだが、amplitudeはcontinuousとする。

```text
novel_base_t,m =
    novel_st_t
    * admission_st_t
    * allocation_st_t,m
```

hard metadata eventは、

```text
accepted_novel_hard_t,m =
    novel_hard_t
    * admission_hard_t
    * allocation_hard_t,m
```

とする。

novel erase/write gateを、

```text
novel_erase_gate_t =
    sigmoid(novel_erase_logit_t)

novel_write_gate_t =
    sigmoid(novel_write_logit_t)
```

とする。occupied victimへのeraseは、

```text
novel_erase_mass_t,m =
    o_{t-1,m}
    * novel_base_t,m
    * novel_erase_gate_t
```

既定の`capacity_conserving` modeでは、writeを、

```text
novel_write_mass_t,m =
    novel_base_t,m
    * novel_write_gate_t
    * (
        (1 - o_{t-1,m})
        + o_{t-1,m}
          * novel_erase_gate_t
    )
```

とする。

したがってoccupied victimでは、

```text
novel_write_mass <= novel_erase_mass
```

となる。empty slotではeraseすべき旧contentがないため、

```text
novel_erase_mass = 0
novel_write_mass =
    novel_base * novel_write_gate
```

とする。`tied` modeでは`erase_mass=o_prev*novel_base`、
`write_mass=novel_base`とする。`free_edit` modeだけがoccupied victimでも
erase/write gateを独立に適用できる。

novel replacementを完全置換に近づける場合でも、addressとamplitudeは別に保つ。

## 10.8 Final Write

```text
erase_mass_final_t,m =
    matched_erase_mass_t,m
    + novel_erase_mass_t,m
```

```text
write_mass_final_t,m =
    matched_write_mass_t,m
    + novel_write_mass_t,m
```

matchedとnovelはhard-forwardで排他的なので、同一tokenで両方が同じslotへ適用されない。

---

# 11. Accounting Semantics

slot contentとwrite metricsはraw eligibilityではなく、実際に適用されたupdateから更新する。
ageはaccepted allocationまたはapplied updateでresetし、occupancyはaccepted hard allocationで更新する。

## 11.1 Applied Write

```text
applied_mass_t,m =
    max(
        erase_mass_final_t,m,
        write_mass_final_t,m
    )

accounted_write_hard_t,m =
    applied_mass_t,m
    >= write_accounting_floor
```

`write_accounting_floor`は明示的なsemantics fieldとし、backendごとのmachine epsilonへ
暗黙に依存させない。continuous massの統計はfloor適用前、write countとage resetは
`accounted_write_hard`から計算する。

## 11.2 Age

```text
if accepted_novel_hard_t,m == 1:
    age_t,m = 0

elif o_{t-1,m} == 0:
    age_t,m = 0

elif accounted_write_hard_t,m == 1:
    age_t,m = 0

else:
    age_t,m =
        age_{t-1,m}
        + delta_clock_t
```

rejectされたtokenはageをresetしない。

## 11.3 Usage

usageはabsolute read activityから更新する。

```text
usage_signal_t,m =
    max_over_tiles(
        rel_t,m,tile
    )
```

```text
usage_t,m =
    usage_decay
    * usage_{t-1,m}
    +
    (
        1 - usage_decay
    )
    * usage_signal_t,m
```

write eligibilityをusageとして数えない。

## 11.4 Occupancy

```text
o_t,m =
    o_{t-1,m}
    OR
    accepted_novel_hard_t,m
```

occupancyはcontinuous write amplitudeではなく、hard-forward allocation eventを表す。
これにより、sub-threshold content updateをunoccupied slotへ残す矛盾を避ける。
matched writeはempty slotへ発生しないため、occupancyを新規に立てない。

## 11.5 Metrics

最低限、次を別々に記録する。

```text
matched address mass
matched erase mass
matched write mass

novel decision rate
admission rate
accepted novel rate

empty allocation count
occupied eviction count

bank selection count
slot selection count

write reject count
write saturation count
write accounting floor
```

---

# 12. Multi-Timescale Banks

slotは、

```text
short
mid
long
```

のpolicy classへ分ける。

bankは意味カテゴリではなく、memory lifecycle priorである。

```text
short:
    high plasticity
    short retention prior
    low eviction protection

mid:
    balanced plasticity
    medium retention prior

long:
    low plasticity
    long retention prior
    high eviction protection
```

初期update-rate ceilingの例は、

```text
mu_short_max = 0.05
mu_mid_max   = 0.02
mu_long_max  = 0.005
```

とする。

これらは理論値ではなく初期探索点である。

`mu_*_max`はmatched editのceilingである。novel allocationにbank別ceilingを
導入する場合は別fieldとして明示し、matched `mu`へ暗黙に連結しない。

bank collapseが観測される前から強いquota regularizerを導入しない。

観測する指標は、

```text
bank occupancy
bank allocation rate
bank eviction rate
bank read mass
bank write mass
bank mean age
bank mean usage
bank redundancy
```

である。

collapse時のみ、

```text
minimum occupancy quota
soft balance loss
bank prior adjustment
```

を比較する。

---

# 13. Optional Adaptive Retention

本節は`screening-v5-retention` profileにのみ適用する。

v5 coreの成立にKDA型retentionは必須ではない。

## 13.1 Slot Vitality

各occupied slotに、

```text
vitality_t,m in [0, 1]
```

を持たせる。

vitalityはslot contentのnormではなく、memory confidenceとして使用する。

unit-normalized key/valueへscalar decayを掛けても、

```text
unit(alpha k) = unit(k)
```

となるため、単純なcontent scalingではread behaviorを十分に変えられない。

## 13.2 Retention Rate

```text
half_life_t,m > 0

alpha_t,m =
    2^(
        -delta_clock_t,m
        / half_life_t,m
    )
```

half-lifeは、

```text
log half_life_t,m =
      bank_half_life_prior[bank_m]
    + token_retention_bias_t
    + optional usage correction
    + optional relevance correction
    - optional conflict correction
```

とする。

初期実装では、

```text
bank prior
+ token-level scalar correction
```

までに限定する。

## 13.3 Vitality Update

```text
vitality_pre_t,m =
    alpha_t,m
    * vitality_{t-1,m}
```

token $t$のread relevance、write eligibility、victim scoringは、このpre-update
`vitality_pre_t,m`を使用する。write後の`vitality_t,m`を同じtokenのroutingへ戻してはならない。

actual writeが適用された場合、

```text
renew_t,m =
    clip(
        applied_mass_t,m,
        0,
        1
    )
```

```text
vitality_t,m =
    vitality_pre_t,m
    +
    renew_t,m
    * (
        1 - vitality_pre_t,m
    )
```

empty allocationでは、

```text
vitality_t,m = 1
```

とする。

reject writeはvitalityをrenewしない。

## 13.4 Read and Write Gating

readでは、

```text
rel_t,m *= vitality_pre_t,m^rho_read
```

write matchingでは任意に、

```text
eligibility_t,m *=
    vitality_pre_t,m^rho_write
```

を適用する。

```text
rho_write = 0
```

を安全な既定候補とする。

古いslotをmatched candidateから急速に除外すると、同一entityが過剰にnovel判定される可能性があるためである。

## 13.5 Retention Clock

比較するclockは次とする。

```text
token clock:
    delta_clock = 1

idle clock:
    delta_clock =
        1 - read_activity

conflict clock:
    delta_clock =
        conflict_signal

hybrid clock:
    delta_clock =
          eta_token
        + eta_idle * idle_signal
        + eta_conflict * conflict_signal
```

長期設定と一時情報で最適clockが異なる可能性がある。

## 13.6 Grouped Retention

slot channelごとのdecayは、一般にprojectionと可換ではない。

```text
P(Ds) != D P(s)
```

したがって、任意channel-wise retentionを既存projected-state recurrenceへ直接導入してはならない。

grouped retentionを導入する場合、slotを、

```text
z_m =
    concat(
        z_m,1,
        ...,
        z_m,G
    )
```

へ分割し、persistent projectionをblock-separableにする。

```text
P =
    block_diag(
        P_1,
        ...,
        P_G
    )
```

group decayを、

```text
D =
    block_diag(
        alpha_1 I,
        ...,
        alpha_G I
    )
```

とすると、

```text
P D z =
    D_projected P z
```

が成立する。

grouped retentionは、

```text
G = 1 / 4 / 8 / 16
```

を比較する。

block restrictionによる表現力低下を補うため、memoryへの入力前およびmemory出力後ではdense mixingを許可する。

---

# 14. Straight-Through Estimator Governance

## 14.1 Three Different Questions

次の三つを区別する。

```text
numerical parity:
    referenceとacceleratorが同じsurrogateを計算するか

estimator validity:
    surrogateがhard-forward modelに有効な学習信号を与えるか

optimization stability:
    temperature、threshold、seedに対して安定か
```

gradient parityはestimator validityを証明しない。

## 14.2 Required Metrics

```text
route flip rate
novel decision flip rate
admission flip rate
bank flip rate
victim flip rate

gradient cosine across temperatures
gradient norm across temperatures

admission saturation
novelty saturation
bank entropy
victim concentration
```

## 14.3 Required Comparisons

```text
hard-forward / soft-backward
soft-forward / soft-backward
hard-forward with stopped routing gradient
temperature sweeps
fixed route control
random route control
```

STEを使用したという理由だけでend-to-end改善をrouting学習の成功とみなさない。

---

# 15. Checkpointing and Numerical Stability

## 15.1 General Recurrence

erase/write分離後のrecurrenceは、

$$
s_t
=
A_t s_{t-1}
+
B_t \Delta_t
$$

と書ける。

```text
A_t =
    1 - erase_mass_t

B_t =
    write_mass_t
```

逆算は、

$$
s_{t-1}
=
\frac{
s_t - B_t\Delta_t
}{
A_t
}
$$

である。

## 15.2 Algebraic and Numerical Conditions

```text
A_t > 0
```

は代数的可逆性の条件にすぎない。

数値安定性はcheckpoint interval全体の誤差増幅で決まる。

interval $\mathcal I$について、

$$
K_{\mathcal I}
=
\sum_{t\in\mathcal I}
-\log\left(
\max(A_t,\epsilon)
\right)
$$

をreconstruction budgetと定義する。

`A_t`がbatch/slot/groupごとに異なる場合は各reversible state trajectoryについて
$K_{\mathcal I}$を計算し、interval平均ではなくtrajectory間の最大値を
worst-case budgetとして扱う。

単純な、

```text
max strength <= 0.95
```

だけを数値保証として使用しない。

validationでは、

```text
worst-case interval budget
observed interval budget histogram
gradient absolute error
gradient relative L2 error
age-gradient error
usage-gradient error
```

を使用する。

許容budgetはdtype、shape、backend、intervalごとに実測校正する。

universal constantとして主張しない。

## 15.3 Irreversible Novel Replacement

強いnovel eraseで、

```text
A_t < reverse_floor
```

となる場合、通常の逆算を行わない。

選択肢は次とする。

```text
checkpoint boundary insertion
local forward recomputation
sparse reset journal
```

reset journalは、

```text
token index
batch index
slot index
pre-update slot
pre-update read key
pre-update write key
pre-update value
pre-update vitality
```

を保存する。

追加memoryは、

```text
O(
    number of irreversible events
    * state per slot
)
```

である。

novel rateが高い場合、journalが全state tapeへ近づくため、

```text
journal events per token
journal bytes per token
recompute time
```

を必ず報告する。

## 15.4 Tape Contents

erase/write分離後は、少なくとも、

```text
erase mass
write mass
routing decisions
candidate state
scalar metadata
```

が必要である。

旧applied-strength一つだけではrecurrenceを復元できない。

---

# 16. Multi-Read Tiles

総key/value dimensionは固定する。

```text
d_k_tile =
    D_k_total / n_read_tiles

d_v_tile =
    D_v_total / n_read_tiles
```

projectionは一度だけ行い、reshapeしてtileへ分ける。

各tileは独立した、

```text
query normalization
key normalization
capacity-calibrated threshold
Trim-and-Square
reject-all aggregation
```

を持つ。

capacity calibrationに使うkey dimensionはtotal dimensionではなく
`d_k_tile`である。

threshold calibrationでは、

```text
occupied slots * number of tiles
```

をtotal test countとして扱う。

usage signalはtile和ではなく、

```text
max relevance across tiles
```

を使用する。

tile数による残差増幅は、

```text
1 / sqrt(n_read_tiles)
```

で抑える。

write pathは初期実装では一系統とする。

---

# 17. Semantic Versioning and Compatibility

write modeとsemantics versionは別fieldとして管理する。

```text
write_mode:
    disabled
    legacy_unconditional
    legacy_threshold
    competitive_novel

semantics_version:
    screening-v4-legacy
    screening-v4-competitive
    screening-v5-core
    screening-v5-retention
```

有効な組合せは明示的に検証する。

| semantics version | valid write mode | status |
| --- | --- | --- |
| `screening-v4-legacy` | `disabled`, `legacy_unconditional`, `legacy_threshold` | implemented compatibility contract |
| `screening-v4-competitive` | `competitive_novel` | implemented predecessor contract |
| `screening-v5-core` | `competitive_novel` | portable Phase 1 candidate; MI300X 0.185B 2,000-step / 0.3B 400-step finite gate通過、memory quality/Pallas accelerator gate未通過 |
| `screening-v5-retention` | `competitive_novel` | design-only |

同じ`write_mode=competitive_novel`でも、semantics versionが異なればoccupancy、
threshold、routing、editing、metadataの契約は異なる。

既存checkpointを読み込んだ場合、暗黙にv5へ変更しない。

```text
missing semantics version:
    write_mode == competitive_novelならscreening-v4-competitive
    それ以外はscreening-v4-legacy

v4 checkpoint:
    declared v4 semanticsを維持

v5 core checkpoint:
    occupancyおよびnew metadata必須

v5 retention checkpoint:
    vitality state必須
```

v5 benchmark toolingでは、headline commandが必ずmodeを明示する。

```text
--write-mode
--semantics-version
--backend-policy
```

未指定でheadline benchmarkを実行しない。

## 17.1 Recovery profileのMI300X反証結果

2026-07-22のMI300X再測定では、Gaussian-null threshold、soft-to-hard read、
self-index loss、上下write budget、redundancy-aware victimを含むtracked profileを
評価した。0.185Bは2,000 stepをfinite完走したが、step 500以降のslot利用は
1 / 16、step 2,000のmemory residual/base RMSは9.42e-6だった。0.3Bも400 stepを
finite完走したが、2 screened layerのaggregate slot利用は3.125%、residual比は
5.37e-7だった。memory-off counterfactualのloss差はrun間で-5e-6から+1.6e-5で、
一貫した改善を示さなかった。

したがって、recovery profileは旧all-write/high-redundancy collapseを解消したが、
memory利用を成立させたのではなく、under-allocationと層間collapseへ退化解を
移したと判定する。次版はglobal平均後のbudgetではなく、screened layer・bank別の
occupancy/allocation curriculumを用い、empty capacityを各層で埋める必要がある。
このgateを通るまでretention、checkpoint redesign、v5 Pallas最適化を品質改善の
根拠として進めない。

全step時系列の再解析では、0.185B/0.3Bとも最初の非ゼロoptimizer更新を反映する
step 3でloss spikeを示し、memory collapseの主要部分は最初の数十stepで生じて
いた。さらに、補助損失がsoft write率とscreened-layer平均に作用していたため、
hard forwardではほぼwriteしないままsoft制約だけを満たす解と、片方の層が他方の
floor不足を隠す解が可能だった。

修正版では、write floorとupper budgetを層別に計算し、forward値をhard実現率、
backwardをsoft novelty/admission経路とするstraight-through surrogateへ変更する。
empty-memory bootstrap中はupper budgetを停止する。またtrunkの初期transient中は
Screening recurrence、residual、補助損失、optimizer更新（weight decayを含む）を
停止する。その後hard occupancyを維持したrecurrenceを開始し、residual、補助損失、
optimizer更新を独立LR multiplierとともに線形起動する。

MiniPile magic samplerかつ`carry_state=false`は数値健全性とthroughputのgateには
使えるが、long-memory品質の主gateにはしない。主品質gateはdistractor付きdelayed
key/value retrievalとし、学習時は文書全体を同一optimizer step内でrecurrent
chunkingする。optimizer stepを跨ぐ単純なstate carryはBPTTを切断し、後段answer
lossから前段write pathを学習できないためである。streaming評価では文書境界を
認識した行別state resetを用い、memory-off loss/accuracy deltaを必須とする。

---

# 18. Backend Validation Policy

backendについて、

```text
available
validated
headline-eligible
experimental
```

を分離する。

例:

```text
portable reference:
    available
    validated
    headline-eligible for correctness

pallas_tpu:
    available
    validated for specified shapes/devices

pallas_gpu_triton:
    availability depends on environment
    v5 requires new real-device gate

pallas_gpu_mosaic:
    experimental until device-native v5 parity
```

`auto` backendは、利用可能であるという理由だけで未検証kernelをheadline実験へ使用しない。

未検証backendは、

```text
explicit experimental flag
```

を要求する。

---

# 19. Semantic Golden Vectors

reference、GPU、TPUが一致しただけでは、正しい意味論を実装したことにならない。

各semantics versionについて、versioned golden vectorsを保存する。

```text
screening-v4-legacy
screening-v4-competitive
screening-v5-core
screening-v5-retention
```

各golden setは次を含む。

```text
input tensors
initial slot state
expected read relevance
expected matched route
expected novelty decision
expected admission decision
expected victim address
expected erase/write mass
expected final state
expected metadata
expected gradients
```

境界caseは最低限、次を含む。

```text
all empty
one occupied slot
all occupied

all rejected
exactly at threshold
slightly below threshold
slightly above threshold

weak aggregate below norm cap
aggregate exactly at norm cap
multiple aligned values above norm cap
canceling values

one eligible slot
multiple equal eligible slots
diffuse eligibility
duplicate keys

admission exactly at threshold
novelty exactly at threshold
applied mass below accounting floor
applied mass exactly at accounting floor
applied mass above accounting floor

empty-slot allocation
occupied eviction

bank tie
slot tie

checkpoint boundary write
irreversible novel replacement
```

---

# 20. Benchmark Provenance

v5実装以降のすべてのbenchmark artifactは次を保存する。

```text
schema version

semantics version

git commit SHA
dirty tree state
tree hash

config file
config hash
lockfile hash

Python version
JAX version
jaxlib version
Flax version
Optax version

CUDA / libtpu / plugin version
driver version
device name
device topology

dtype policy
compile flags
backend policy

input seed
input tensor hash

warm-up count
iteration count
raw timing samples

compile included or excluded
device transfer included or excluded
synchronization policy

output errors
per-input gradient errors
loss difference

peak memory
collective counts
```

medianだけでなく、

```text
raw samples
mean
median
standard deviation
p10
p90
```

を保存する。

kernel speedupとmodel qualityを同一表の単一指標へ統合しない。

---

# 21. Implementation Status of the Predecessor

補正後v4 recurrenceでは、2026年7月18日にTPU v5e-4上で以下が確認されている。

```text
device-native lowering
6 recurrence outputs
17 differentiable input gradients
T = 128 recurrence parity
checkpointed T = 512 parity
4-device 0.185B optimizer-step execution
```

tracked recurrence shapeは、

```text
T = 128
B = 1
M = 16
d_slot = 128
d_k = 64
d_v = 64
```

である。

報告されたmedianは、

```text
Pallas forward:
    0.578 ms

reference forward:
    3.949 ms

Pallas forward + backward:
    4.888 ms

reference forward + backward:
    13.171 ms
```

である。

これはv4 predecessor kernelに関する測定であり、次の証拠ではない。

```text
v5 correctness
v5 quality
v5 GPU performance
full-model steady-state throughput
measured peak-memory reduction
multi-host scaling efficiency
production stability
```

v5では新しいstate、routing、erase/write semanticsを追加するため、既存parity結果を流用しない。

---

# 22. Evaluation Plan

## 22.1 Baselines

最低限、以下を比較する。

1. RWKV-7 baseline
2. parameter-matched RWKV-7
3. FLOPs-matched RWKV-7
4. compute-budget-matched RWKV-7
5. legacy unconditional screening
6. legacy threshold screening
7. v4 competitive novel
8. v5 core
9. v5 core without capacity calibration
10. v5 core without occupancy
11. v5 core without ambiguity penalty
12. v5 core with tied erase/write
13. v5 core with capacity-conserving erase/write
14. v5 retention
15. softmax-over-slots memory
16. random read/write control
17. parameter-matched FFN branch
18. compute-budget-matched FFN branch
19. Mamba系matched-size model
20. RetNet系matched-size model
21. DeltaNetまたはGated DeltaNet系matched-size model

可能な場合はKDA系も比較するが、異なるstate geometryとkernel成熟度を考慮し、単純な一対一比較として扱わない。

## 22.2 Training Match

最低限、次を揃える。

```text
training tokens
optimizer
learning-rate schedule
batch tokens
context distribution
tokenizer
dataset order
parameter budget
compute budget
wall-clock budget
```

parameter-matchedとcompute-budget-matchedを同じbaselineとして扱わない。
ここでFLOPs-matchedはtokenあたりの理論演算量、compute-budget-matchedは
学習run全体の理論演算量、wall-clock-matchedは実時間budgetを指し、別々に報告する。

## 22.3 Retrieval Tasks

```text
passkey retrieval
Needle-in-a-Haystack
RULER multi-needle retrieval
RULER aggregation
RULER multi-hop tracing
MQAR
delayed key-value recall
key-value recall with distractors
same-key multiple-value recall
```

## 22.4 Load-Factor Evaluation

memory load factorを、

$$
\text{load factor}
=
\frac{
\text{independent active associations}
}{
\text{occupied slots}
}
$$

と定義する。

単一scoreだけでなく、

```text
recall vs load factor
false match vs load factor
novel rate vs load factor
eviction rate vs load factor
stale read vs load factor
```

を報告する。

slot数を増やしてscoreが上がっただけではarchitecture改善とみなさない。

## 22.5 Revision and Forgetting Tasks

```text
fact revision:
    A is X
    later A is Y
    query A

temporary override:
    normal state A
    temporary state B
    later restore A

entity relocation:
    location changes repeatedly

same-name conflict:
    distinct entities share surface name

obsolete instruction replacement

distractor contamination

short-lived state:
    should expire

persistent setting:
    should not expire
```

主要指標は、

```text
new-fact accuracy
old-fact intrusion
revision latency
stale read rate
false expiration
over-retention
retain-revise Pareto frontier
```

とする。

## 22.6 Position Sensitivity

relevant informationを、

```text
beginning
middle
end
uniform random position
```

へ置き、Lost-in-the-Middle型の位置依存性を測る。

## 22.7 Slot and System Metrics

```text
slot occupancy
dead-slot rate
duplicate-slot rate
cross-bank redundancy
cross-layer redundancy

read mass
read energy
reject-all rate

matched confidence
route concentration
route entropy

novel rate
admission rate
write rate

erase mass
write mass

eviction age
eviction usage
eviction regret proxy

tokens/s
step latency
compile time
peak memory
communication time
scaling efficiency
```

## 22.8 Causal Analysis

```text
slot ablation
slot patching
read relevance shuffle
matched-write suppression
admission suppression
novel allocation suppression
erase-gate suppression
write-gate suppression
retention freeze
vitality reset
bank-preserving shuffle
memory reset
memory branch zeroing
```

可視化だけをcausal evidenceとして扱わない。

---

# 23. Statistical Protocol

synthetic memory tasksは最低5 seed、matched language-model trainingは最低3 seedとする。

報告するものは、

```text
mean
standard deviation
confidence interval
effect size
per-seed values
```

である。

task-level比較にはpaired bootstrapを使用できる。

seed-level aggregateには、

```text
paired permutation test
paired t-statistic
```

のいずれかを効果量と併記する。

単一runのroute visualizationを一般化しない。

---

# 24. Acceptance Criteria

## 24.1 Semantic Acceptance

1. legacy modeが旧結果を許容誤差内で再現する
2. empty slotがreadおよびmatched writeへ参加しない
3. empty slotがoccupied victimより先に選ばれる
4. weak eligibilityがnormalizationだけで強いwriteへ増幅されない
5. diffuse matchがclear matchと同じconfidenceを持たない
6. hard addressとwrite amplitudeが分離されている
7. rejectされたtokenがcontent、age、write count、vitalityを変更しない
8. all-irrelevant条件でread outputが0近傍になる
9. read aggregationがrelevant slot数だけで無制限に増幅しない
10. sequence chunk sizeで意味論が変わらない
11. slotとprojected statesが同じerase/write係数で更新される
12. thresholdが1へ接近して数値不安定になるconfigを拒否する

## 24.2 Numerical Acceptance

1. referenceとacceleratorでforward parityが成立する
2. 全differentiable inputのgradient parityが成立する
3. threshold境界caseのgolden vectorが一致する
4. checkpoint intervalごとのreconstruction budgetを報告する
5. irreversible eventを不安定な逆算で処理しない
6. absolute errorとrelative errorを入力別に評価する
7. ageおよびusage gradientを独立に評価する
8. NaN、Inf、silent clampを許可しない

## 24.3 Scientific Acceptance

1. matched baselineに対するretrieval改善
2. PPLだけでは説明できないrecall改善
3. false readの減少
4. stale readの減少
5. load-factor崩壊点の改善
6. old-fact intrusionの減少
7. false expirationの許容範囲内維持
8. duplicate slotの減少
9. 一部slotまたはmemory operationのcausal contribution
10. 複数seedで方向が再現
11. loss-vs-wall-clockが許容範囲
12. 追加peak memoryが許容範囲

## 24.4 Systems Acceptance

1. headline backendがdevice-nativeで検証されている
2. raw timing sampleが公開される
3. compileとexecutionが分離される
4. full-step throughputが報告される
5. peak memoryが実測される
6. multi-device correctnessとscaling efficiencyが分離される
7. unvalidated backendが暗黙にheadline pathへ選ばれない

---

# 25. Falsification Criteria

次の場合、本設計の中心仮説は支持されない。

```text
PPLは改善するがmemory taskが改善しない

retrievalは改善するが
parameter-matched controlで同等になる

slot数増加だけで改善が説明される

write頻度増加だけで改善が説明される

all-irrelevant readがsoftmax controlと変わらない

novelty calibrationがM変更に対して安定しない

revision accuracy改善と同時に
persistent fact recallが大幅悪化する

adaptive retentionがfixed half-lifeを上回らない

STE temperatureまたはseedで結果が反転する

kernel speedupがfull-stepへ反映されない

memory branchをzeroingしてもlogitが変化しない
```

negative resultも報告対象とする。

---

# 26. Risks

1. absolute thresholdが学習初期にmemory starvationを起こす
2. threshold warm-upが最終的にも低thresholdを固定する
3. capacity calibrationのnull modelが学習後のkey分布と一致しない
4. occupancy導入で初期memory utilizationが遅れる
5. admissionがalways-writeまたはnever-writeへcollapseする
6. duplicate slotが曖昧routeを自己強化する
7. ambiguity penaltyが複数有用slotへのwriteを過剰抑制する
8. empty-first allocationがbank imbalanceを悪化させる
9. age/usage victim policyが将来重要なslotを削除する
10. eviction criticが短いtraining segmentへ過適合する
11. capacity-conserving updateが表現力を制限する
12. free-edit updateがslot normを発散させる
13. hard-forward routingのbiased gradientが不安定になる
14. multi-read tileがduplicate retrieval subspaceを形成する
15. fixed bank partitionがtask分布に合わない
16. vitalityが全slotで0または1へ飽和する
17. token-clock retentionが無関係な長文で過剰忘却する
18. event-clock retentionが一時情報を保持しすぎる
19. grouped projectionがcross-group表現力を低下させる
20. reset journalが高novel-rate taskで肥大化する
21. Pallas kernelの複雑化がwall-clock benefitを消す
22. reference、GPU、TPUの意味論が時間とともにdriftする
23. 複数screened layerが同一情報を重複保存する
24. slot observabilityがinterpretabilityと誤認される
25. occupied slot数の離散変化でcapacity-calibrated thresholdが不連続に変わる

---

# 27. Implementation Priority

## Phase 0: Semantics and Reproducibility

```text
semantics_version
explicit benchmark mode
provenance schema
golden vectors
backend validation status
```

この段階はmodel behaviorを変えない。

## Phase 1: v5 Core Memory Semantics

```text
occupancy state
empty-first allocation
safe threshold parameterization
capacity-calibrated read/write threshold
bounded read aggregation
ambiguity-aware matched confidence
```

## Phase 1 Decision Gate

Phase 1はportable referenceを先に実装し、accelerator kernel、checkpoint redesign、
adaptive retentionへ直ちに展開しない。v4 predecessor、no-screening baseline、
parameter-matched controlと比較し、synthetic memory taskは最低5 seed、
matched small language-model runは最低3 seedで評価する。

次を満たす場合のみPhase 2以降へ進む。

```text
semantic acceptanceとchunk invarianceが成立する

false read、load-factor collapse、revisionまたはstale readの
少なくとも一つでv4を再現可能に上回る

改善がparameter数、compute増加、write頻度だけでは説明されない

seed、threshold、slot countを変えても改善方向が反転しない

事前に定めたloss、throughput、peak-memory budget内に収まる
```

このgateを満たさない場合はv5 coreをproduction pathへ昇格せず、
negative resultとablationを保存して設計を凍結または棄却する。

## Phase 2: Continuous Memory Editing

```text
hard address / continuous amplitude
erase/write separation
capacity-conserving mode
new accounting semantics
```

## Phase 3: Capacity Management

```text
redundancy metric
revised victim score
optional future-use critic
write-budget regularization
```

## Phase 4: Checkpoint Redesign

```text
interval-aware stability budget
erase/write tape
irreversible-event handling
reset journal or recompute path
```

## Phase 5: Adaptive Retention

```text
vitality
half-life policy
retention clock
retention causal interventions
```

## Phase 6: Grouped Retention

```text
grouped latent state
block-separable projections
group-specific decay
new Pallas kernels
```

Phase 5以降は、v5 coreがmatched experimentsで有効と確認された場合にのみproduction candidateとする。

---

# 28. Claim Boundary

実装完了時に主張できるのは次である。

> State-Level Screeningを、明示的occupancy、容量校正されたabsolute relevance、曖昧性を考慮したcompetitive routing、hard addressとcontinuous memory editingの分離、および数値安定性を明示した固定容量online memoryとして再設計した。

実験なしに次を主張してはならない。

```text
long-context recallを改善した
memory hygieneを改善した
KDAより優れている
softmax attentionを置換できる
human-readable memoryを獲得した
hard eviction問題を解決した
production trainingに利用可能である
```

v5 retentionを実装した場合でも、主張は次に限定する。

> KDAおよびGated Delta系研究に着想を得たadaptive retentionとerase/write分離を、projected-state consistencyを保つslot memoryへ統合した。

KDAの直接再実装とは呼ばない。

---

# 29. References

1. Ken M. Nakanishi, *Screening Is Enough*, [arXiv:2604.01178](https://arxiv.org/abs/2604.01178), 2026.
2. Bo Peng et al., *RWKV-7 “Goose” with Expressive Dynamic State Evolution*, [arXiv:2503.14456](https://arxiv.org/abs/2503.14456), 2025.
3. Bo Peng et al., *RWKV: Reinventing RNNs for the Transformer Era*, [arXiv:2305.13048](https://arxiv.org/abs/2305.13048), 2023.
4. Ashish Vaswani et al., *Attention Is All You Need*, [arXiv:1706.03762](https://arxiv.org/abs/1706.03762), 2017.
5. Albert Gu and Tri Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, [arXiv:2312.00752](https://arxiv.org/abs/2312.00752), 2023.
6. Yutao Sun et al., *Retentive Network: A Successor to Transformer for Large Language Models*, [arXiv:2307.08621](https://arxiv.org/abs/2307.08621), 2023.
7. Songlin Yang et al., *Parallelizing Linear Transformers with the Delta Rule over Sequence Length*, [arXiv:2406.06484](https://arxiv.org/abs/2406.06484), 2024.
8. Songlin Yang et al., *Gated Delta Networks: Improving Mamba2 with Delta Rule*, [arXiv:2412.06464](https://arxiv.org/abs/2412.06464), 2024.
9. Kimi Team et al., *Kimi Linear: An Expressive, Efficient Attention Architecture*, [arXiv:2510.26692](https://arxiv.org/abs/2510.26692), 2025.
10. Ali Hatamizadeh, Yejin Choi, Jan Kautz, *Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention*, [arXiv:2605.22791](https://arxiv.org/abs/2605.22791), 2026.
11. Ali Behrouz et al., *It’s All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization*, [arXiv:2504.13173](https://arxiv.org/abs/2504.13173), 2025.
12. Ali Behrouz et al., *Titans: Learning to Memorize at Test Time*, [arXiv:2501.00663](https://arxiv.org/abs/2501.00663), 2025.
13. Simran Arora et al., *Zoology: Measuring and Improving Recall in Efficient Language Models*, [arXiv:2312.04927](https://arxiv.org/abs/2312.04927), 2023.
14. Nelson F. Liu et al., *Lost in the Middle: How Language Models Use Long Contexts*, [arXiv:2307.03172](https://arxiv.org/abs/2307.03172), 2023.
15. Cheng-Ping Hsieh et al., *RULER: What’s the Real Context Size of Your Long-Context Language Models?*, [arXiv:2404.06654](https://arxiv.org/abs/2404.06654), 2024.
16. Alex Graves et al., *Neural Turing Machines*, [arXiv:1410.5401](https://arxiv.org/abs/1410.5401), 2014.
17. Alex Graves et al., *Hybrid Computing Using a Neural Network with Dynamic External Memory*, [arXiv:1605.06065](https://arxiv.org/abs/1605.06065), 2016.
18. Francesco Locatello et al., *Object-Centric Learning with Slot Attention*, [arXiv:2006.15055](https://arxiv.org/abs/2006.15055), 2020.
19. Yoshua Bengio et al., *Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation*, [arXiv:1308.3432](https://arxiv.org/abs/1308.3432), 2013.
20. L. A. Belady, *A Study of Replacement Algorithms for a Virtual-Storage Computer*, IBM Systems Journal 5(2), 1966.
