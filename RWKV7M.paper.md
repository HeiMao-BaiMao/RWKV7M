# RWKV 系 Recurrent LLM における State-Level Screening
## Slot-Based Absolute Relevance Read/Write の研究設計草稿

**版**: academic-draft-v2  
**対象**: RWKV-7 系、またはそれに近い recurrent / state-based / matrix-state language model  
**位置づけ**: 本文書は、Multiscreen の screening 機構を RWKV 系の圧縮 state 表現へ移植するための研究設計草稿である。`state-level screening` は既存の標準モジュール名ではなく、本稿で定義する設計仮説である。  
**主張の強さ**: 性能改善、解釈可能性、memory hygiene に関する記述は、既存研究からの直接的帰結ではなく、実験で検証すべき仮説として扱う。

---

## 1. 要約

本稿の中心命題は、Multiscreen の token-level screening を RWKV にそのまま移植することではない。むしろ、RWKV 系 recurrent LLM の圧縮 state を固定個数の slot として構成し、現在入力に対して十分 relevant な slot のみを absolute relevance に基づいて read / write する設計を提案する。

標準的な softmax attention では、候補 key の集合に対して重みが相対的に再配分される。真に relevant な key が存在しない場合でも、重みの総和は 1 になるため、何らかの候補が必ず読まれる。Multiscreen はこの点を問題視し、bounded similarity と明示的閾値によって各候補を独立に採否判定する screening を提案した。[^multiscreen]

RWKV-7 は、固定サイズ state を用いる recurrent / state-based sequence model であり、token あたり定数メモリ・定数推論時間を掲げる。generalized delta rule、vector-valued gating、vector-valued in-context learning rate、relaxed value replacement rule により、state evolution を強化している。[^rwkv7]

この背景から、本稿の設計仮説は次のようにまとめられる。

> RWKV 系 recurrent LLM において、圧縮 state を slot 化し、現在入力に対する absolute relevance に基づいて read / write を制御すれば、token-to-token attention を復活させずに、長文 recall、無関係 state の読み出し抑制、長期 state の汚染低減を改善できる可能性がある。

ただし、この仮説は未検証である。Multiscreen の実験結果は token-level screening に関するものであり、RWKV 系 state slot への移植効果を保証しない。本設計の有効性は、PPL だけでなく、associative recall、long-context retrieval、long-form consistency、slot-level causal intervention によって検証される必要がある。

---

## 2. 研究背景

### 2.1 Softmax attention の相対配分性

Transformer は recurrence や convolution を使わず、attention 機構によって sequence transduction を行うモデルとして導入された。[^attention] その後、self-attention は大規模言語モデルの中心的構成要素となった。

標準的な softmax attention は、query に対して各 key の相対的な重みを割り当てる。これは候補集合内で「どれが最も近いか」を決めるには有効だが、「読むべき候補が存在しない」という状態を表現しにくい。候補がすべて無関係でも、softmax は総和 1 の分布を返す。

Multiscreen はこの相対配分性を問題とし、各 key を明示的閾値で独立に採否判定する screening を提案した。[^multiscreen] 本稿はこの思想を、token-to-token interaction ではなく、RWKV 系の compressed recurrent state に対する access problem として再解釈する。

### 2.2 RWKV-7 と state-based memory

RWKV-7 “Goose” は、固定サイズ state による recurrent sequence modeling を行う。[^rwkv7] その記憶は明示的な token 履歴ではなく、圧縮された state evolution として存在する。

したがって、RWKV 系で問題になるのは「どの過去 token を見るか」ではなく、「現在の state のうち何を読むか、どこに新情報を書くか」である。本稿の state-level screening は、この state access を slot 単位で制御する試みである。

### 2.3 Efficient sequence model における recall 問題

Mamba は selective state space model により、入力依存の state propagation / forgetting を導入した。[^mamba] RetNet は recurrence と attention の関係を整理し、parallel / recurrent / chunkwise recurrent の三つの計算形式を提示した。[^retnet] DeltaNet 系研究は、linear transformer や state-space model が softmax attention と比べて in-context retrieval で弱くなる問題を背景に、delta rule 型更新による associative recall 改善を試みている。[^deltanet]

Zoology は、efficient language model の性能差の大きな部分が in-context recall によって説明されることを示し、Multi-Query Associative Recall により recall 能力を評価した。[^zoology] この知見は、state-based / recurrent 系モデルを評価する際、validation loss や perplexity だけでは不十分であることを示唆する。

### 2.4 長文 context 評価の注意点

Lost in the Middle は、長文 context を扱えるモデルでも、関連情報の位置によって性能が大きく変化し、とくに中間位置の情報利用が不安定になることを示した。[^lost_middle] RULER は、Needle-in-a-Haystack 型評価が長文理解の一部しか測れないことを指摘し、multi-hop tracing や aggregation を含む包括的な長文評価を提案した。[^ruler]

したがって、本設計の評価では、PPL、synthetic retrieval、associative recall、multi-hop retrieval、long dialogue consistency、創作設定維持率を分けて測る必要がある。

---

## 3. 設計上の位置づけ

State-level screening は、Multiscreen、RWKV-7、Mamba、RetNet、DeltaNet のいずれかに含まれる既存機構ではない。本稿の新規性は、Multiscreen の token-level screening を、RWKV 系 compressed recurrent state slot に対する read / write screening として再定義する点にある。

本設計は次の既存研究を参照する。

1. Multiscreen から、bounded similarity、明示的閾値、非 sum-to-one aggregation、TanhNorm の思想を借りる。[^multiscreen]
2. RWKV-7 から、固定サイズ recurrent state を主記憶として扱う設計を借りる。[^rwkv7]
3. Mamba / RetNet / DeltaNet から、attention 以外の長系列 modeling における state evolution と selective update の問題意識を借りる。[^mamba][^retnet][^deltanet]
4. Zoology / Lost in the Middle / RULER から、長文 recall と context utilization を PPL から分離して測る評価観を借りる。[^zoology][^lost_middle][^ruler]

重要なのは、これらの既存成果が本設計の有効性を直接保証するわけではないという点である。本設計は、既存研究に基づく仮説であり、独立に検証される必要がある。

---

## 4. 基本設計

### 4.1 State slot

各 screened layer \(\ell\) に、固定個数 \(M\) の slot bank を持たせる。

\[
S_t^\ell = \{s_{t,1}^\ell, s_{t,2}^\ell, \dots, s_{t,M}^\ell\}
\]

\[
s_{t,m}^\ell \in \mathbb{R}^{d_s}
\]

\(M\) は context length に依存しない固定値である。追加 memory は \(O(Md_s)\) であり、token 履歴長 \(T\) には直接依存しない。

### 4.2 Query / key / value

現在 token の layer 入力 hidden state を \(x_t^\ell \in \mathbb{R}^{d}\) とする。

\[
q_t^\ell = W_q^\ell \operatorname{LN}(x_t^\ell)
\]

各 slot から key と value を作る。

\[
k_{t,m}^\ell = W_k^\ell s_{t-1,m}^\ell
\]

\[
v_{t,m}^\ell = W_v^\ell s_{t-1,m}^\ell
\]

### 4.3 Unit normalization

\[
\operatorname{unit}(z)=\frac{z}{\max(\lVert z\rVert_2,\varepsilon)}
\]

\[
\bar q_t^\ell=\operatorname{unit}(q_t^\ell),\qquad
\bar k_{t,m}^\ell=\operatorname{unit}(k_{t,m}^\ell),\qquad
\bar v_{t,m}^\ell=\operatorname{unit}(v_{t,m}^\ell)
\]

query と key を unit norm にすると、similarity は cosine similarity になり、値域は \([-1,1]\) に制約される。この制約により、閾値 \(\tau\) は「どの程度の類似度以上を relevant とみなすか」という解釈を持つ。

value unit norm は初期実装では有効にする。ただし、value norm に情報量や信頼度が含まれる可能性があるため、最終評価では ablation 対象とする。

### 4.4 Similarity

\[
c_{t,m}^\ell
=
\langle
\bar q_t^\ell,\bar k_{t,m}^\ell
\rangle
\]

\[
c_{t,m}^\ell \in [-1,1]
\]

ここで \(c_{t,m}^\ell\) は attention logit ではない。softmax には渡さない。

### 4.5 Trim-and-Square relevance

学習可能閾値を次のように定義する。

\[
\tau^\ell = 2\sigma(\theta_\tau^\ell)-1
\]

推奨する relevance は次である。

\[
r_{t,m}^{content,\ell}
=
\left[
\max\left(
0,
\frac{c_{t,m}^\ell-\tau^\ell}{1-\tau^\ell+\varepsilon}
\right)
\right]^2
\]

この式は、\(c=\tau\) で relevance 0、\(c=1\) で relevance 1 になる。Multiscreen 型の Trim-and-Square を、実装上扱いやすい閾値直接指定形式に変形したものである。[^multiscreen]

### 4.6 非 sum-to-one aggregation

read relevance を総和 1 に正規化しない。

\[
z_t^\ell
=
\sum_{m=1}^{M}
r_{t,m}^{read,\ell}
\bar v_{t,m}^\ell
\]

すべての relevance が 0 に近い場合、\(z_t^\ell\) も 0 に近くなる。これにより、「今回は読むべき slot が存在しない」という状態を表現できる。

これは softmax attention との主要な違いである。softmax では候補集合に irrelevant な要素しかない場合でも、相対的に最も大きい候補へ重みが流れる。

### 4.7 TanhNorm

非 sum-to-one aggregation では、複数 slot が同時に活性化した場合に出力ノルムが増加する可能性がある。そこで TanhNorm を用いる。

\[
\operatorname{TanhNorm}_\kappa(z)
=
\kappa
\tanh
\left(
\frac{\lVert z\rVert_2}{\kappa}
\right)
\frac{z}{\lVert z\rVert_2+\varepsilon}
\]

\[
u_t^\ell
=
\operatorname{TanhNorm}_\kappa(z_t^\ell)
\]

\(\kappa\) は出力ノルム上限に近いスケールであり、初期値は \(\kappa=1\) とする。

TanhNorm は、小さい \(z\) に対してはほぼ恒等写像として働き、大きい \(z\) に対してはノルムを \(\kappa\) 近傍へ抑える。ただし、多数 slot の弱い活性化と少数 slot の強い活性化の差をノルム上で圧縮する可能性があるため、pre-TanhNorm norm、post-TanhNorm norm、active slot 数は別々に記録する。

### 4.8 RWKV core との residual fusion

RWKV core の出力を次とする。

\[
h_t^{base,\ell}, R_t^\ell
=
\operatorname{RWKVCore}^\ell(x_t^\ell,R_{t-1}^\ell)
\]

screening 出力は線形射影と gate を通じて residual に加える。

\[
g_t^{screen,\ell}
=
\sigma(W_g^\ell\operatorname{LN}(x_t^\ell)+b_g^\ell)
\]

\[
\tilde u_t^\ell
=
W_o^\ell u_t^\ell
\]

\[
h_t^\ell
=
h_t^{base,\ell}
+
\lambda_\ell g_t^{screen,\ell}\odot \tilde u_t^\ell
\]

\(\lambda_\ell\) は小さく初期化する。推奨初期値は 0.01 から 0.05 程度である。これは学習初期に screening branch が backbone を破壊しないようにするためである。

---

## 5. Read / write 分離

### 5.1 Read relevance

read relevance は、現在の hidden state がどの slot を参照すべきかを表す。

\[
r_{t,m}^{read,\ell}
=
\operatorname{ScreenRead}(x_t^\ell,s_{t-1,m}^\ell)
\]

read 出力は次である。

\[
u_t^\ell
=
\operatorname{TanhNorm}_\kappa
\left(
\sum_m r_{t,m}^{read,\ell}\bar v_{t,m}^\ell
\right)
\]

### 5.2 Write relevance

write relevance は、現在の情報をどの slot に書き込むべきかを表す。read relevance と write relevance は同一視しない。

\[
q_t^{write,\ell}
=
W_{q,w}^\ell
\operatorname{LN}([x_t^\ell;h_t^{base,\ell}])
\]

\[
c_{t,m}^{write,\ell}
=
\left\langle
\operatorname{unit}(q_t^{write,\ell}),
\operatorname{unit}(W_{k,w}^\ell s_{t-1,m}^\ell)
\right\rangle
\]

\[
r_{t,m}^{write,\ell}
=
\left[
\max\left(
0,
\frac{c_{t,m}^{write,\ell}-\tau_w^\ell}{1-\tau_w^\ell+\varepsilon}
\right)
\right]^2
\]

slot update は漸進的更新にする。

\[
s_{t,m}^\ell
=
s_{t-1,m}^\ell
+
\mu_m^\ell r_{t,m}^{write,\ell}
(
\Delta s_{t,m}^\ell
-
s_{t-1,m}^\ell
)
\]

hard overwrite は用いない。RWKV-7 が state evolution を重視する設計である以上、screening branch も state 遷移を不連続に破壊しない方が望ましい。[^rwkv7]

### 5.3 Read/write 分離の理由

現在参照すべき情報と、現在更新すべき情報は一致しない。たとえば、長期人物設定は生成時に読む必要があるが、直近の一時的な発話内容で上書きすべきではない。この区別を失うと、長文会話や創作タスクで memory contamination が起きやすくなる。

したがって、read score と write score は原則として分離する。

---

## 6. Slot bank と時間スケール

### 6.1 Short / mid / long bank

創作・長文会話を想定する場合、slot を少なくとも三種類に分ける。

- **short bank**: 直近文脈、局所 coherence、直近会話
- **mid bank**: 場面、章、短期要約、直近の主題
- **long bank**: 世界設定、人物設定、持続的事実

### 6.2 Bank-specific update rate

bank ごとに更新率を制約する。

\[
\mu_{long}<\mu_{mid}<\mu_{short}
\]

推奨初期値は次である。

\[
\mu_{short,max}=0.05,\qquad
\mu_{mid,max}=0.02,\qquad
\mu_{long,max}=0.005
\]

long bank は簡単に書き換えない。これは、小説・長文会話において、長期設定が一時的文脈で破壊されることを防ぐためである。

### 6.3 Slot age

slot に age を持たせる場合、最後に有意に更新されてからの経過 step を \(a_{t,m}^\ell\) とする。

age mask の一例は次である。

\[
g_{t,m}^{age,\ell}
=
\sigma
\left(
\frac{w_{b(m)}^\ell-a_{t,m}^\ell}{\sigma_{b(m)}^\ell+\varepsilon}
\right)
\]

最終 read relevance は次である。

\[
r_{t,m}^{read,\ell}
=
r_{t,m}^{content,\ell}
g_{t,m}^{age,\ell}
g_m^{bank,\ell}
\]

ただし、この式は age が大きい slot の read relevance を下げるため、short / mid bank には適していても、long bank では「古いが重要な記憶」を読みにくくする危険がある。初期実装では age mask と bank bias は省略し、content relevance のみで評価する。導入する場合も、long bank では age mask を無効化するか、bank ごとに別の age policy を使う。

---

## 7. MiPE / RoPE の扱い

Multiscreen は MiPE を導入しているが、これは token 位置に対する位置的構造を扱うための機構である。RoPE は absolute position を回転行列で符号化し、self-attention に relative position dependency を導入する位置表現として提案された。[^roformer]

State-level screening における slot index は token index ではない。slot \(m\) は「系列中の m 番目の token」ではなく、圧縮 memory bank 内の格納場所である。したがって、token 位置向けの MiPE / RoPE を slot index へ直接適用すると、位置幾何の意味が崩れる可能性がある。

推奨は以下である。

1. 初期版では MiPE / RoPE を導入しない。
2. 必要な場合は、slot age embedding または bank embedding に置き換える。
3. token index 用の rotary encoding を slot index に直貼りしない。

---

## 8. Phase 設計

### 8.1 Phase 1: Read-screening-only phase

従来の “read-only phase” という名称は、slot 自体は slow updater で更新されるため誤解を招く。本稿では **read-screening-only phase** と呼ぶ。

目的は、state slot read が retrieval に寄与するかを分離して確認することである。

- fixed slot 数
- unit normalization
- learnable tau
- Trim-and-Square relevance
- 非 sum-to-one aggregation
- TanhNorm
- small residual fusion
- thresholded write screening なし
- slot update は低速・安定な updater のみ

この段階では memory hygiene ではなく retrieval 改善を見る。

### 8.2 Phase 2: Read/write separated phase

目的は、memory hygiene を検証することである。

- read score / write score 分離
- write relevance による update modulation
- bank-specific update rate
- slot age
- optional age mask

この段階では、long bank の汚染低減、contradiction rate、長期設定維持率を見る。

### 8.3 Phase 3: Multi-timescale memory phase

目的は、創作・長文会話における実用的 memory control を検証することである。

- short / mid / long bank の明示管理
- usage EMA
- dead-slot regularization
- slot diversity loss
- causal intervention hooks
- long-form benchmark

---

## 9. 期待される効果と仮説

### 9.1 Retrieval 改善

Multiscreen は、screening により長文 perplexity、retrieval performance、長 context 推論 latency で有利な傾向を示している。[^multiscreen] ただし、その結果は token-level screening に関するものであり、state-level screening に直接外挿することはできない。

**仮説 H1**: State-level screening は、RWKV 系モデルにおける associative recall、passkey retrieval、multi-query recall を改善する。

### 9.2 無関係 state の読み出し抑制

softmax over slots を使うと、すべての slot が無関係な場合でも相対的に最も近い slot へ重みが流れる。非 sum-to-one screening では、すべての relevance が 0 になれる。

**仮説 H2**: State-level screening は、無関係な internal memory の読み出しを減らし、長文生成時の設定混線を低減する。

### 9.3 Memory hygiene

write screening と bank-specific update rate を導入すると、short-term information が long-term slot を汚染する頻度を下げられる可能性がある。

**仮説 H3**: Read/write 分離と long bank の低速更新は、長期設定の破壊率を下げる。

### 9.4 Interpretability ではなく observability

slot relevance は可視化できる。しかし、それは直ちに解釈可能性を意味しない。ある slot の relevance が高いことは、その slot が出力に寄与した可能性を示すが、その slot が人間可読な「人物設定」や「章要約」に対応するとは限らない。

したがって、本設計では interpretability ではなく、まず **memory access observability** と呼ぶ。解釈可能性を主張するには、slot ablation、slot patching、counterfactual write suppression などの因果的検証が必要である。

---

## 10. 限界とリスク

### 10.1 未検証提案である

本設計は、Multiscreen と RWKV-7 の直接的な組み合わせとして既存研究で検証されたものではない。Multiscreen の実験結果は token-level screening の結果であり、RWKV state slot に対する screening の効果を保証しない。[^multiscreen][^rwkv7]

### 10.2 State slot が意味単位に分化する保証はない

RWKV 系の state は圧縮表現である。slot を導入しても、それぞれが人物・場面・事実のような意味単位に自然分化する保証はない。slot collapse、dead slot、冗長 slot が起こる可能性がある。

### 10.3 閾値が強すぎると記憶が痩せる

Trim-and-Square は閾値未満で relevance を 0 にするため、初期学習で閾値が高すぎると slot が使われにくくなる。必要に応じて以下を検討する。

- tau を低く初期化する
- leaky relevance warm-up を使う
- tau schedule を導入する
- dead-slot regularization を後半から弱く入れる

### 10.4 PPL 改善と recall 改善は一致しない可能性がある

Zoology は、efficient model の性能差に in-context recall 能力が大きく関係することを示している。[^zoology] また、Lost in the Middle や RULER は、長文 context を入力できることと、その情報を頑健に使えることが異なることを示している。[^lost_middle][^ruler]

したがって、PPL が改善しない場合でも retrieval / consistency が改善する可能性があり、逆に PPL が改善しても memory access が改善していない可能性がある。

### 10.5 Latency 優位は kernel 実装依存である

score が 0 になっても、naive 実装では全 slot を計算する。active slot skip や cached key/value update を含めない限り、理論的な sparsity は wall-clock latency に反映されない可能性がある。latency を主張する場合は、prefill latency、decode latency、memory footprint、active compute ratio を分けて報告する必要がある。

---

## 11. 評価計画

### 11.1 Baseline

最低限、以下を比較する。

1. RWKV-7 baseline
2. parameter-matched RWKV-7
3. FLOPs-matched RWKV-7
4. RWKV-7 + read-screening-only state-level screening
5. RWKV-7 + read/write state-level screening
6. RWKV-7 + read/write + short/mid/long bank
7. RWKV-7 + softmax-over-slots memory
8. RWKV-7 + top-k slot selection
9. RWKV-7 + random slot read/write control
10. Mamba / RetNet / DeltaNet 系の近接規模モデル

Mamba、RetNet、DeltaNet は、attention 以外の efficient sequence modeling の代表的比較対象として重要である。[^mamba][^retnet][^deltanet]

### 11.2 Ablation

以下の ablation を必須とする。

- unit norm あり / なし
- value unit norm あり / なし
- Trim-and-Square vs sigmoid gate
- learnable tau vs fixed tau
- TanhNorm あり / なし
- read-screening-only vs read/write
- read/write shared score vs separated score
- age mask あり / なし
- bank-specific update rate あり / なし
- long bank update rate を short と同じにした条件
- slot 数 \(M\) の比較
- screened layer 数の比較

### 11.3 Retrieval-oriented benchmark

以下を含める。

- passkey retrieval
- Needle-in-a-Haystack variants
- RULER-style multi-needle retrieval
- Multi-Query Associative Recall
- key-value retrieval with distractors
- long-context consistency probe

NIAH 単独では不十分である。RULER が指摘するように、単純な needle retrieval は長文理解の一部しか測れない可能性がある。[^ruler]

### 11.4 Long-form / creative writing benchmark

創作・長文会話向けには、以下を追加する。

- キャラクター設定維持率
- 固有名詞再出現精度
- 章要約からの再参照成功率
- contradiction rate
- long dialogue memory retention
- 同名異人物の混同率
- 一時情報が長期設定を破壊する率
- 伏線回収成功率

### 11.5 Causal analysis

relevance 可視化だけでは不十分である。以下の介入を行う。

- slot ablation
- slot patching
- read relevance shuffle
- write suppression
- long bank freeze
- short bank freeze
- 特定 slot の zeroing による logits / generation 変化の測定

これにより、slot relevance が単なる相関ではなく、出力に因果的寄与を持つかを評価する。

---

## 12. 実装仕様の要約

### 12.1 最小 read branch

\[
q_t = W_q \operatorname{LN}(x_t)
\]

\[
k_m = W_k s_m,\qquad v_m = W_v s_m
\]

\[
c_m = \langle \operatorname{unit}(q_t),\operatorname{unit}(k_m)\rangle
\]

\[
r_m =
\left[
\max
\left(
0,
\frac{c_m-\tau}{1-\tau+\varepsilon}
\right)
\right]^2
\]

\[
z_t=\sum_m r_m \operatorname{unit}(v_m)
\]

\[
u_t=\operatorname{TanhNorm}_\kappa(z_t)
\]

\[
h_t=h_t^{base}+\lambda \sigma(W_g\operatorname{LN}(x_t))\odot W_o u_t
\]

### 12.2 最小 write update

\[
s_{t,m}
=
s_{t-1,m}
+
\mu_m r_{t,m}^{write}
(\Delta s_{t,m}-s_{t-1,m})
\]

read-screening-only phase では \(r_{t,m}^{write}\) を使わず、安定な slow update のみを用いる。

### 12.3 JAX / Flax 実装上の注意

実装を JAX / Flax で行う場合、slot 更新は in-place 代入ではなく、`jax.lax.scan`、`jnp.where`、broadcasting、PyTree の immutable update で実装する。PyTorch 風の `slots[:, m] = ...` は、論文説明用の擬似コードでは許容されるが、JAX 実装仕様としては使わない。

---

## 13. 採否基準

本設計を採用するには、少なくとも次を示す必要がある。

1. parameter-matched / FLOPs-matched baseline に対して retrieval 指標が改善する。
2. PPL だけでなく associative recall / long-context retrieval が改善する。
3. all-irrelevant 条件で read-out がゼロ近傍になる。
4. softmax-over-slots baseline より無関係 slot 読み出しが少ない。
5. write screening により long bank の不要更新が減る。
6. slot ablation により、少なくとも一部 task で因果的寄与が確認できる。
7. TanhNorm を外すと安定性が落ちる、または TanhNorm の必要性が実験的に説明される。
8. dead slot / slot collapse が管理可能である。

これらを満たさない場合、本設計は「理論的に妥当そうな追加 memory branch」に留まり、state-level screening としての有効性は示されない。

---

## 14. 結論

State-level screening は、Multiscreen の absolute relevance screening を RWKV 系 recurrent state に移植するための設計仮説である。softmax attention を復活させるのではなく、固定個数の state slot に対して独立した read / write 判定を行うことで、RWKV 系の定数メモリ・定数推論時間という性質を比較的保ちながら、長文 recall と memory hygiene を改善する可能性がある。

ただし、核心的な不確実性は残る。RWKV の圧縮 state が slot 単位で意味的に分化する保証はなく、Multiscreen の token-level screening の成果が state-level screening に直接移る保証もない。したがって、本設計は PPL だけではなく、retrieval、long-form consistency、slot-level causal intervention を含む評価によって検証される必要がある。

---

## 脚注

[^multiscreen]: Ken M. Nakanishi, *Screening Is Enough*, arXiv:2604.01178, 2026. https://arxiv.org/abs/2604.01178

[^rwkv7]: Bo Peng et al., *RWKV-7 “Goose” with Expressive Dynamic State Evolution*, arXiv:2503.14456, 2025. https://arxiv.org/abs/2503.14456

[^attention]: Ashish Vaswani et al., *Attention Is All You Need*, arXiv:1706.03762, 2017. https://arxiv.org/abs/1706.03762

[^mamba]: Albert Gu and Tri Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, arXiv:2312.00752, 2023; COLM 2024. https://arxiv.org/abs/2312.00752

[^retnet]: Yutao Sun et al., *Retentive Network: A Successor to Transformer for Large Language Models*, arXiv:2307.08621, 2023. https://arxiv.org/abs/2307.08621

[^deltanet]: Songlin Yang et al., *Parallelizing Linear Transformers with the Delta Rule over Sequence Length*, arXiv:2406.06484, 2024. https://arxiv.org/abs/2406.06484

[^zoology]: Simran Arora et al., *Zoology: Measuring and Improving Recall in Efficient Language Models*, arXiv:2312.04927, 2023; ICLR 2024. https://arxiv.org/abs/2312.04927

[^lost_middle]: Nelson F. Liu et al., *Lost in the Middle: How Language Models Use Long Contexts*, arXiv:2307.03172, 2023; Transactions of the Association for Computational Linguistics, 2024. https://arxiv.org/abs/2307.03172

[^ruler]: Cheng-Ping Hsieh et al., *RULER: What’s the Real Context Size of Your Long-Context Language Models?*, arXiv:2404.06654, 2024. https://arxiv.org/abs/2404.06654

[^roformer]: Jianlin Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, arXiv:2104.09864, 2021. https://arxiv.org/abs/2104.09864

---

## 参考文献

Arora, Simran, Sabri Eyuboglu, Aman Timalsina, Isys Johnson, Michael Poli, James Zou, Atri Rudra, and Christopher Ré. “Zoology: Measuring and Improving Recall in Efficient Language Models.” arXiv:2312.04927, 2023. https://arxiv.org/abs/2312.04927

Gu, Albert, and Tri Dao. “Mamba: Linear-Time Sequence Modeling with Selective State Spaces.” arXiv:2312.00752, 2023. https://arxiv.org/abs/2312.00752

Hsieh, Cheng-Ping, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, and Boris Ginsburg. “RULER: What’s the Real Context Size of Your Long-Context Language Models?” arXiv:2404.06654, 2024. https://arxiv.org/abs/2404.06654

Liu, Nelson F., Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. “Lost in the Middle: How Language Models Use Long Contexts.” arXiv:2307.03172, 2023. https://arxiv.org/abs/2307.03172

Nakanishi, Ken M. “Screening Is Enough.” arXiv:2604.01178, 2026. https://arxiv.org/abs/2604.01178

Peng, Bo, Ruichong Zhang, Daniel Goldstein, Eric Alcaide, Haowen Hou, Janna Lu, William Merrill, Guangyu Song, Kaifeng Tan, Saiteja Utpala, Nathan Wilce, Johan S. Wind, Tianyi Wu, Daniel Wuttke, and Christian Zhou-Zheng. “RWKV-7 ‘Goose’ with Expressive Dynamic State Evolution.” arXiv:2503.14456, 2025. https://arxiv.org/abs/2503.14456

Su, Jianlin, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. “RoFormer: Enhanced Transformer with Rotary Position Embedding.” arXiv:2104.09864, 2021. https://arxiv.org/abs/2104.09864

Sun, Yutao, Li Dong, Shaohan Huang, Shuming Ma, Yuqing Xia, Jilong Xue, Jianyong Wang, and Furu Wei. “Retentive Network: A Successor to Transformer for Large Language Models.” arXiv:2307.08621, 2023. https://arxiv.org/abs/2307.08621

Vaswani, Ashish, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. “Attention Is All You Need.” arXiv:1706.03762, 2017. https://arxiv.org/abs/1706.03762

Yang, Songlin, Bailin Wang, Yu Zhang, Yikang Shen, and Yoon Kim. “Parallelizing Linear Transformers with the Delta Rule over Sequence Length.” arXiv:2406.06484, 2024. https://arxiv.org/abs/2406.06484
