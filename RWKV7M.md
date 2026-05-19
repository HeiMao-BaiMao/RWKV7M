# JAX / Flax 実装用仕様書: RWKV 系 State-Level Screening

## 0. 文書の目的

この文書は、コーディングエージェントに JAX / Flax で実装させるための実装仕様である。目的は、RWKV 系 recurrent / state-based LLM に対して、固定個数の slot memory を追加し、Multiscreen 的な absolute relevance screening を state read/write に適用することにある。

この文書は論文草稿ではない。実装者が迷わずコードへ落とすため、以下を固定する。

- 数式
- テンソル形状
- Flax Module 構造
- recurrent state の PyTree 構造
- `lax.scan` による学習・推論処理
- read-only phase と read/write phase の違い
- loss と logging stats
- 最小実装、推奨実装、非推奨実装
- 単体テストと受け入れ条件

実装対象は JAX / Flax / Optax とする。既存 RWKV-7 実装へ直接パッチする場合でも、この文書では RWKV core を抽象 interface として扱う。

---

## 1. 設計の要約

### 1.1 何を作るか

各 screened layer に、固定個数 `M` の slot bank を持たせる。

\[
S_t^\ell = \{s_{t,1}^\ell, \dots, s_{t,M}^\ell\},\qquad s_{t,m}^\ell \in \mathbb{R}^{d_s}
\]

現在 token の hidden state `x_t` から query を作り、各 slot から key/value を作る。query-key cosine similarity を `[-1, 1]` に制限し、学習可能閾値を超えた slot だけを read する。

softmax は使わない。relevance を総和 1 に正規化しない。これにより、全 slot が無関係なときに read-out がゼロ近傍になる。

### 1.2 何故この組み合わせか

この機構は、以下の問題を同時に解くために組み合わせる。

1. **unit normalization**  
   query-key similarity を cosine similarity にし、値域を `[-1, 1]` に固定する。閾値 `tau` が「どの類似度以上を relevant とみなすか」という解釈を持つ。

2. **Trim-and-Square relevance**  
   slot ごとに独立した absolute relevance を得る。softmax のように、無関係な slot にも相対的に重みが配られる問題を避ける。

3. **非 sum-to-one 集約**  
   relevant な slot がない場合を表現する。softmax では常にどこかを読むため、この状態を表しにくい。

4. **TanhNorm**  
   非正規化集約では複数 slot が同時に活性化するとノルムが増える。TanhNorm で read-out のノルムを上限 `C` に近く抑える。

5. **residual fusion**  
   RWKV core の出力を破壊しない。screening branch は小さい residual correction として加える。

6. **read/write 分離**  
   「今読むべき slot」と「今書くべき slot」は同一とは限らない。長文会話や創作では、設定を読む必要はあっても、それを一時情報で上書きすべきでない場合がある。

7. **short / mid / long bank**  
   slot ごとに更新時間スケールを変える。short は頻繁に更新し、long は保守的に更新する。

---

## 2. 実装範囲

### 2.1 MVP で必ず実装するもの

MVP は以下に限定する。

- Flax Linen 実装
- fixed slot 数 `M`
- read-only screening branch
- slow slot updater
- unit normalization
- learnable tau threshold
- Trim-and-Square relevance
- TanhNorm
- residual fusion
- `lax.scan` による token scan
- loss は next-token cross entropy のみ
- logging stats:
  - active slot count
  - mean read relevance
  - max read relevance
  - pre-TanhNorm norm
  - post-TanhNorm norm

### 2.2 Phase 2 で実装するもの

MVP が動作したあとに実装する。

- write screening branch
- read query と write query の分離
- bank-specific update rate
- slot age
- age mask
- read/write relevance logging
- update norm logging

### 2.3 Phase 3 で実装するもの

研究評価用。

- short / mid / long bank の厳密管理
- optional slot usage EMA
- dead-slot regularization
- diversity loss
- update penalty
- causal intervention hooks
- slot ablation / read shuffle / write suppression

---

## 3. 記号とテンソル形状

### 3.1 基本記号

| 記号 | 意味 |
|---|---|
| `B` | batch size |
| `T` | sequence length |
| `L` | layer 数 |
| `d` | model width |
| `d_s` | slot state width |
| `d_k` | screening key/query width |
| `d_v` | screening value width |
| `M` | slot 数 |
| `V` | vocabulary size |

### 3.2 実装テンソル

入力系列:

```text
input_ids:  int32[B, T]
target_ids: int32[B, T]
```

embedding 後:

```text
x: float[B, T, d]
```

time step `t` の layer 入力:

```text
x_t_l: float[B, d]
```

slot bank:

```text
slots_l: float[B, M, d_s]
ages_l:  int32[B, M]
```

read branch:

```text
q_r:       float[B, d_k]
k:         float[B, M, d_k]
v:         float[B, M, d_v]
sim_r:     float[B, M]
rel_r:     float[B, M]
z:         float[B, d_v]
u:         float[B, d_v]
read_out:  float[B, d]
```

write branch:

```text
q_w:       float[B, d_k]
k_w:       float[B, M, d_k]
sim_w:     float[B, M]
rel_w:     float[B, M]
delta_s:   float[B, M, d_s]
new_slots: float[B, M, d_s]
```

---

## 4. Flax state 設計

### 4.1 Model variables と recurrent state を分ける

Flax では学習パラメータと recurrent state を明確に分ける。

- `params`: 学習対象。Dense weight, tau, gate, bank update parameters など。
- `ScreenState`: forward 時に持ち回る recurrent state。slot values, ages, optional usage EMA など。

`ScreenState` は `flax.struct.dataclass` で定義する。

```python
from flax import struct
import jax.numpy as jnp

@struct.dataclass
class LayerScreenState:
    slots: jnp.ndarray      # [B, M, d_s]
    ages: jnp.ndarray       # [B, M], int32 or float32
    usage_ema: jnp.ndarray  # [B, M], optional; MVPでは zeros のままでもよい

@struct.dataclass
class ModelScreenState:
    layers: tuple           # tuple[LayerScreenState, ...] length L_screened
```

### 4.2 JAX での注意

in-place update は使わない。PyTorch 的な `slots[:, m] = ...` は不可。必ず vectorized update または `jnp.where` / `jax.vmap` / broadcasting を使う。

悪い例:

```python
slots[:, m] = new_value
```

良い例:

```python
slots = slots + update_rate[..., None] * (delta_s - slots)
```

---

## 5. Core interface

### 5.1 RWKV core の抽象 interface

既存 RWKV core を直接実装する必要はない。screening branch は次の interface を仮定する。

```python
h_base, new_rwkv_state = rwkv_layer.apply_core(
    x_t_l,          # [B, d]
    rwkv_state_l,   # PyTree
    deterministic: bool,
)
```

返り値:

```text
h_base: float[B, d]
new_rwkv_state_l: PyTree
```

### 5.2 MVP 用の代替 core

コーディングエージェントが standalone 実装を作る場合、最初は RWKV core を簡易 recurrent block で代替してよい。

```text
h_base = MLP(LN(x_t)) + x_t
state  = dummy_state
```

ただし、最終統合ではこの placeholder を RWKV core に差し替える。

---

## 6. 数式仕様: 共通関数

### 6.1 Unit normalization

\[
\operatorname{unit}(x) = \frac{x}{\max(\lVert x\rVert_2, \varepsilon)}
\]

実装:

```python
def unit_norm(x, axis=-1, eps=1e-6):
    norm = jnp.linalg.norm(x, axis=axis, keepdims=True)
    return x / jnp.maximum(norm, eps)
```

理由: query/key を cosine 空間へ写し、threshold を意味ある値にするため。

### 6.2 Learnable tau

raw parameter `theta` を `tau in (-1, 1)` に写す。

\[
\tau = 2\sigma(\theta)-1
\]

実装:

```python
def bounded_tau(theta):
    return 2.0 * jax.nn.sigmoid(theta) - 1.0
```

推奨初期値:

```text
tau_init = 0.0
```

`tau=0.5` から始めると初期に dead slot が起きやすい。初期は緩く、必要なら schedule で上げる。

### 6.3 Trim-and-Square relevance

推奨式:

\[
r_\tau(c)
=
\left[
\max\left(
\frac{c-\tau}{1-\tau+\varepsilon}, 0
\right)
\right]^2
\]

実装:

```python
def trim_square(sim, tau, eps=1e-6):
    x = (sim - tau) / (1.0 - tau + eps)
    return jnp.square(jax.nn.relu(x))
```

理由: `sim=tau` で 0、`sim=1` で 1。閾値の意味が明確で、Multiscreen 型の acceptance-width formulation と同値に書ける。

### 6.4 Optional leaky warm-up relevance

初期学習で dead slot が出る場合のみ使う。

\[
r(c) = (1-\alpha) r_\tau(c) + \alpha \sigma(\gamma(c-\tau))
\]

`alpha` は warm-up 中に 1 から 0 へ下げる。

実装:

```python
def relevance_with_warmup(sim, tau, alpha, gamma=8.0):
    hard = trim_square(sim, tau)
    soft = jax.nn.sigmoid(gamma * (sim - tau))
    return (1.0 - alpha) * hard + alpha * soft
```

MVP では `alpha=0` でよい。

### 6.5 TanhNorm

\[
\operatorname{TanhNorm}_C(z)
=
C\tanh\left(\frac{\lVert z\rVert_2}{C}\right)
\frac{z}{\lVert z\rVert_2+\varepsilon}
\]

実装:

```python
def tanh_norm(z, cap=1.0, eps=1e-6):
    norm = jnp.linalg.norm(z, axis=-1, keepdims=True)
    safe_norm = jnp.maximum(norm, eps)
    scale = cap * jnp.tanh(norm / cap) / safe_norm
    return scale * z
```

理由: relevance を sum-to-one にしないため、集約ノルムが active slot 数に依存する。TanhNorm で branch 出力の暴走を抑える。

---

## 7. Read branch 仕様

### 7.1 入力

```text
x_t:   float[B, d]
slots: float[B, M, d_s]
ages:  float[B, M]
```

### 7.2 Query / key / value

\[
q_t^r = W_q^r \operatorname{LN}(x_t)
\]

\[
k_m = W_k s_m,
\qquad
v_m = W_v s_m
\]

\[
\bar q_t^r = \operatorname{unit}(q_t^r),
\quad
\bar k_m = \operatorname{unit}(k_m),
\quad
\bar v_m = \operatorname{unit}(v_m)
\]

Flax 実装では Dense は以下。

```text
q_proj_r: Dense(d_k, use_bias=False)
k_proj_r: Dense(d_k, use_bias=False)
v_proj:   Dense(d_v, use_bias=False)
```

MVP では read key と write key は共有しない前提でもよいが、Phase 2 では `k_proj_r` と `k_proj_w` を分けることを推奨する。

### 7.3 Similarity

\[
c_{t,m}^r = \langle \bar q_t^r, \bar k_m \rangle
\]

実装:

```python
sim_r = jnp.einsum('bd,bmd->bm', q_r, k_r)
```

### 7.4 Read relevance

\[
r_{t,m}^{content}=r_{\tau_r}(c_{t,m}^r)
\]

age mask を入れる場合:

\[
g_{t,m}^{age}=\sigma\left(\frac{w_{b(m)}-a_{t,m}}{\sigma_{b(m)}+\varepsilon}\right)
\]

bank bias を入れる場合:

\[
g_m^{bank}=\sigma(\beta_{b(m)})
\]

最終 read relevance:

\[
r_{t,m}^{read}=r_{t,m}^{content}\,g_{t,m}^{age}\,g_m^{bank}
\]

MVP では:

\[
r_{t,m}^{read}=r_{t,m}^{content}
\]

### 7.5 Aggregation

\[
z_t=\sum_{m=1}^{M} r_{t,m}^{read}\bar v_m
\]

\[
u_t=\operatorname{TanhNorm}_C(z_t)
\]

実装:

```python
z = jnp.einsum('bm,bmd->bd', rel_r, v)
u = tanh_norm(z, cap=config.tanh_norm_cap)
```

### 7.6 Residual fusion

\[
g_t=\sigma(W_g\operatorname{LN}(x_t)+b_g)
\]

\[
\tilde u_t=W_o u_t
\]

\[
h_t=h_t^{base}+\lambda g_t\odot \tilde u_t
\]

実装:

```python
g = jax.nn.sigmoid(gate_proj(layer_norm(x_t)))  # [B, d]
read_out = out_proj(u)                           # [B, d]
h = h_base + lambda_screen * g * read_out
```

推奨初期値:

```text
lambda_screen_init = 0.01
```

理由: 学習初期に screening branch が RWKV core を壊さないようにする。

---

## 8. Slot update 仕様

### 8.1 Read-only phase の slot update

read-only phase とは、read relevance は使うが、write relevance は使わない段階である。slot は非 thresholded slow updater で更新する。

候補更新:

\[
\Delta s_{t,m}
=
\tanh\left(W_\Delta[\operatorname{LN}(x_t); h_t; e_m]\right)
\]

slow update gate:

\[
\omega_{t,m} = \sigma\left(W_\omega[\operatorname{LN}(x_t); s_{t-1,m}; e_m]\right)
\]

update:

\[
s_{t,m}=s_{t-1,m}+\mu_{b(m)}\omega_{t,m}(\Delta s_{t,m}-s_{t-1,m})
\]

MVP ではさらに簡略化してよい。

\[
s_{t,m}=s_{t-1,m}+\mu_{b(m)}(\Delta s_{t,m}-s_{t-1,m})
\]

理由: read branch の有効性を先に確認するため。write screening を同時に入れると、read 改善と write 制御の効果が分離できない。

### 8.2 Read/write phase の slot update

write query:

\[
q_t^w = W_q^w \operatorname{LN}([x_t;h_t^{base}])
\]

write key:

\[
k_m^w=W_k^w s_{t-1,m}
\]

write similarity:

\[
c_{t,m}^w = \langle \operatorname{unit}(q_t^w), \operatorname{unit}(k_m^w) \rangle
\]

write relevance:

\[
r_{t,m}^{write}=r_{\tau_w}(c_{t,m}^w)
\]

write update:

\[
s_{t,m}=s_{t-1,m}+\mu_{b(m)}r_{t,m}^{write}(\Delta s_{t,m}-s_{t-1,m})
\]

理由: read は「今使うべき記憶」、write は「今更新すべき記憶」。この二つを同じ score にすると、一時情報が long memory に混入する危険が高い。

### 8.3 Bank-specific update rate

bank ごとに update rate の上限を分ける。

推奨初期値:

```text
mu_short_max = 0.05
mu_mid_max   = 0.02
mu_long_max  = 0.005
```

learnable parameterization:

\[
\mu_b = \mu_{b,max}\sigma(\theta_{\mu,b})
\]

実装:

```python
mu = mu_max_by_slot * jax.nn.sigmoid(theta_mu_by_slot)  # [M]
```

broadcast:

```python
slots = slots + mu[None, :, None] * rel_w[:, :, None] * (delta_s - slots)
```

read-only phase では `rel_w` の代わりに `omega` または `1` を使う。

### 8.4 Age update

Phase 2 以降で使う。

hard age reset:

\[
a_{t,m}=
\begin{cases}
0 & r_{t,m}^{write} > \eta_{age} \\
a_{t-1,m}+1 & \text{otherwise}
\end{cases}
\]

JAX 実装:

```python
new_ages = jnp.where(rel_w > eta_age, 0.0, ages + 1.0)
```

read-only phase では slow updater が常に動くので、age reset は `omega > eta_age` で判定するか、MVP では age を使わない。

---

## 9. Flax Module 構成

### 9.1 Config dataclass

```python
from dataclasses import dataclass

@dataclass
class ScreeningConfig:
    d_model: int
    d_slot: int
    d_k: int
    d_v: int
    n_slots: int
    screened_layers: tuple[int, ...]
    bank_ids: tuple[int, ...]        # length M; 0 short, 1 mid, 2 long
    tau_init: float = 0.0
    tanh_norm_cap: float = 1.0
    lambda_screen_init: float = 0.01
    eps: float = 1e-6
    use_value_unit_norm: bool = True
    use_age_mask: bool = False
    use_bank_bias: bool = False
    use_write_screening: bool = False
    use_leaky_warmup: bool = False
    leaky_alpha: float = 0.0
    leaky_gamma: float = 8.0
    mu_short_max: float = 0.05
    mu_mid_max: float = 0.02
    mu_long_max: float = 0.005
```

### 9.2 `StateLevelScreening` module

責務:

- read relevance を計算する
- TanhNorm read-out を返す
- write relevance または slow updater で slot を更新する
- stats を返す

推奨 signature:

```python
class StateLevelScreening(nn.Module):
    config: ScreeningConfig

    def __call__(
        self,
        x_t: jnp.ndarray,          # [B, d]
        h_base: jnp.ndarray,       # [B, d]
        state: LayerScreenState,
        *,
        phase: str,
        deterministic: bool,
    ) -> tuple[jnp.ndarray, LayerScreenState, dict]:
        ...
```

返り値:

```text
screened_h: float[B, d]
new_state: LayerScreenState
stats: dict[str, Array]
```

`screened_h` は RWKV core output に residual fusion 済みの hidden state とする。

### 9.3 `ScreenedRWKVLayer`

責務:

1. RWKV core を呼ぶ。
2. 対象 layer なら screening を呼ぶ。
3. 非対象 layer なら core output をそのまま返す。

signature:

```python
class ScreenedRWKVLayer(nn.Module):
    config: ModelConfig
    layer_idx: int

    def __call__(self, x_t, rwkv_state_l, screen_state_l, *, phase, deterministic):
        h_base, new_rwkv_state_l = self.rwkv_core(x_t, rwkv_state_l, deterministic=deterministic)
        if self.layer_idx in config.screened_layers:
            h, new_screen_state_l, stats = self.screening(
                x_t, h_base, screen_state_l, phase=phase, deterministic=deterministic
            )
        else:
            h = h_base
            new_screen_state_l = screen_state_l
            stats = {}
        return h, new_rwkv_state_l, new_screen_state_l, stats
```

### 9.4 `ScreenedRWKVModel`

責務:

- token embedding
- time scan
- layer stack
- logits projection
- loss は外部 training step で計算

signature:

```python
class ScreenedRWKVModel(nn.Module):
    config: ModelConfig

    def __call__(
        self,
        input_ids: jnp.ndarray,          # [B, T]
        rwkv_state: Any,
        screen_state: ModelScreenState,
        *,
        phase: str,
        deterministic: bool,
    ) -> tuple[jnp.ndarray, Any, ModelScreenState, dict]:
        ...
```

返り値:

```text
logits: float[B, T, V]
new_rwkv_state: PyTree
new_screen_state: ModelScreenState
stats: dict[str, Array]
```

---

## 10. `lax.scan` 仕様

### 10.1 時間方向 scan

JAX では Python loop ではなく `jax.lax.scan` を使う。

carry:

```text
carry = (rwkv_state, screen_state)
```

scan input:

```text
x_t_ids: int32[B]
```

scan output:

```text
logits_t: float[B, V]
stats_t: PyTree
```

擬似コード:

```python
def step(carry, token_ids_t):
    rwkv_state, screen_state = carry
    x_t = embed(token_ids_t)  # [B, d]

    layer_stats = []
    for l in range(config.n_layers):
        x_t, rwkv_state_l, screen_state_l, stats_l = layer_l(
            x_t,
            rwkv_state.layers[l],
            screen_state.layers[l],
            phase=phase,
            deterministic=deterministic,
        )
        rwkv_state = rwkv_state.replace_layer(l, rwkv_state_l)
        screen_state = screen_state.replace_layer(l, screen_state_l)
        layer_stats.append(stats_l)

    logits_t = lm_head(final_ln(x_t))
    return (rwkv_state, screen_state), (logits_t, merge_stats(layer_stats))

final_carry, (logits_time_major, stats_time_major) = jax.lax.scan(
    step,
    (rwkv_state, screen_state),
    input_ids.T,
)

logits = jnp.swapaxes(logits_time_major, 0, 1)  # [B, T, V]
```

### 10.2 `replace_layer` の注意

Flax struct は immutable にする。tuple を再構成する関数を用意する。

```python
def tuple_set(xs, i, x):
    return xs[:i] + (x,) + xs[i+1:]
```

JIT 内で Python-level layer loop を使う場合、`n_layers` は static なので許容される。より高度には `scan` over layers も可能だが、MVP では不要。

---

## 11. 初期化仕様

### 11.1 slot 初期化

MVP では slot state はゼロ初期化する。

```python
def init_screen_state(batch_size, config):
    layer_states = []
    for _ in config.screened_layers:
        layer_states.append(LayerScreenState(
            slots=jnp.zeros((batch_size, config.n_slots, config.d_slot), dtype=jnp.float32),
            ages=jnp.zeros((batch_size, config.n_slots), dtype=jnp.float32),
            usage_ema=jnp.zeros((batch_size, config.n_slots), dtype=jnp.float32),
        ))
    return ModelScreenState(layers=tuple(layer_states))
```

ただし、slot ごとの対称性を破るため、candidate update には learned slot embedding `e_m` を必ず入れる。

### 11.2 slot embedding

\[
e_m \in \mathbb{R}^{d_s}
\]

Flax parameter:

```python
slot_embed = self.param(
    'slot_embed',
    nn.initializers.orthogonal(),
    (config.n_slots, config.d_slot),
)
```

orthogonal initializer が shape 制約で難しい場合は normal initializer を使い、直後に normalize してもよい。

### 11.3 tau 初期化

`tau_init` を `theta` に変換する。

\[
\theta = \operatorname{logit}\left(\frac{\tau+1}{2}\right)
\]

実装:

```python
def theta_from_tau(tau):
    p = (tau + 1.0) / 2.0
    p = jnp.clip(p, 1e-4, 1.0 - 1e-4)
    return jnp.log(p) - jnp.log1p(-p)
```

MVP:

```text
tau_r_init = 0.0
tau_w_init = 0.0
```

### 11.4 lambda_screen 初期化

`lambda_screen` は scalar parameter とする。

```python
lambda_raw = self.param(
    'lambda_raw',
    nn.initializers.constant(jnp.log(jnp.exp(0.01)-1)),
    (),
)
lambda_screen = jax.nn.softplus(lambda_raw)
```

固定値でもよいが、学習可能にする場合は softplus で正値制約をかける。

---

## 12. 学習仕様

### 12.1 Loss

基本 loss:

\[
\mathcal{L}_{LM}
=
-\frac{1}{BT}
\sum_{b,t}
\log p(x_{b,t+1}\mid x_{b,\leq t})
\]

実装:

```python
def cross_entropy_loss(logits, targets, mask=None):
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)
    if mask is not None:
        nll = nll * mask
        return jnp.sum(nll) / jnp.maximum(jnp.sum(mask), 1.0)
    return jnp.mean(nll)
```

### 12.2 Auxiliary losses

MVP では使わない。Phase 3 で導入する。

#### Dead slot loss

\[
\bar u_m = \operatorname{mean}_{B,T}(\mathbf{1}[r_{t,m}^{read}>\eta])
\]

\[
\mathcal{L}_{dead}
=
\frac{1}{M}
\sum_m
\max(0,u_{min}-\bar u_m)^2
\]

注意: 強くしすぎると全 slot を無理に読む挙動になり、absolute relevance の意味を壊す。

#### Diversity loss

slot key の collapse を防ぐ。

\[
\mathcal{L}_{div}
=
\frac{1}{M(M-1)}
\sum_{i\ne j}
\cos^2(\bar k_i,\bar k_j)
\]

#### Update penalty

\[
\mathcal{L}_{upd}
=
\frac{1}{BTM}
\sum_{b,t,m}
\lVert s_{t,m}-s_{t-1,m}\rVert_2^2
\]

### 12.3 Train step

```python
@jax.jit
def train_step(train_state, batch, rwkv_state, screen_state, phase):
    def loss_fn(params):
        logits, new_rwkv_state, new_screen_state, stats = train_state.apply_fn(
            {'params': params},
            batch['input_ids'],
            rwkv_state,
            screen_state,
            phase=phase,
            deterministic=False,
        )
        loss = cross_entropy_loss(logits, batch['target_ids'], batch.get('mask'))
        aux = compute_aux_losses(stats, phase)
        total = loss + aux['total']
        metrics = {'loss': loss, **aux, **summarize_stats(stats)}
        return total, (metrics, new_rwkv_state, new_screen_state)

    (total_loss, aux_out), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
    metrics, new_rwkv_state, new_screen_state = aux_out
    train_state = train_state.apply_gradients(grads=grads)
    metrics['total_loss'] = total_loss
    return train_state, new_rwkv_state, new_screen_state, metrics
```

### 12.4 Optimizer groups

Optax で parameter groups を分ける。

推奨:

1. large matrix parameters: weight decay あり
2. bias, LayerNorm scale, tau, lambda, update rate: weight decay なし
3. optional RWKV core params: 既存実装の方針に従う

mask 例:

```python
def decay_mask(params):
    flat = traverse_util.flatten_dict(params)
    mask = {}
    for path, value in flat.items():
        name = '/'.join(path)
        use_decay = (
            value.ndim >= 2
            and 'slot_embed' not in name
            and 'tau' not in name
            and 'lambda' not in name
            and 'norm' not in name.lower()
            and 'bias' not in name.lower()
        )
        mask[path] = use_decay
    return traverse_util.unflatten_dict(mask)
```

---

## 13. 推論仕様

### 13.1 Prefill

長い prompt を `lax.scan` で通し、最終 recurrent state と screen state を得る。

```python
logits, rwkv_state, screen_state, stats = model.apply(
    variables,
    prompt_ids,
    init_rwkv_state,
    init_screen_state,
    phase='inference',
    deterministic=True,
)
```

### 13.2 Decode 1 token

decode では `T=1` の input を渡す。

```python
logits, rwkv_state, screen_state, stats = model.apply(
    variables,
    token_id[:, None],
    rwkv_state,
    screen_state,
    phase='inference',
    deterministic=True,
)
next_logits = logits[:, -1, :]
```

### 13.3 State reset

新しい独立セッションでは screen state をゼロ初期化する。会話継続では state を保持する。

long conversation でユーザーが明示的に「記憶をリセット」と要求する実装を想定する場合、RWKV state と screen state の両方を reset する。

---

## 14. Logging stats

各 forward で以下を記録する。

MVP 必須:

```text
rel_read_mean:       mean(rel_r)
rel_read_max:        max(rel_r)
active_slots_mean:   mean(sum(rel_r > eta_active, axis=-1))
z_norm_mean:         mean(norm(z))
u_norm_mean:         mean(norm(u))
tau_r:               scalar
lambda_screen:       scalar
```

Phase 2:

```text
rel_write_mean
rel_write_max
read_write_overlap
slot_update_norm
age_mean_by_bank
active_slots_by_bank
```

Phase 3:

```text
usage_ema_by_slot
read_histogram_by_slot
write_histogram_by_slot
bank_migration_count
slot_ablation_effect
```

`active slot` 判定閾値:

```text
eta_active = 1e-3
```

---

## 15. Intervention hooks

研究評価のため、forward に intervention config を渡せるようにする。

```python
@struct.dataclass
class InterventionConfig:
    ablate_slots: jnp.ndarray | None       # bool[M]
    shuffle_read: bool = False
    suppress_write: bool = False
    force_read_rel: jnp.ndarray | None = None  # [B, M]
    patch_slots: jnp.ndarray | None = None     # [B, M, d_s]
```

MVP では不要だが、コード構造として後から入れやすいように `StateLevelScreening.__call__` の引数に optional で受けられる設計にする。

### 15.1 Slot ablation

```python
slots = jnp.where(ablate_slots[None, :, None], 0.0, slots)
```

### 15.2 Read shuffle

```python
perm = jax.random.permutation(rng, config.n_slots)
rel_r = rel_r[:, perm]
```

### 15.3 Write suppression

```python
if intervention.suppress_write:
    rel_w = jnp.zeros_like(rel_w)
```

---

## 16. Numerical precision

### 16.1 推奨 dtype

- model activations: `bfloat16`
- similarity / norm / relevance: `float32`
- slots: `bfloat16` または `float32`

推奨は、MVP では slots を `float32` にして安定性を優先する。高速化段階で `bfloat16` slots を検討する。

### 16.2 実装規則

norm と similarity の前に `astype(jnp.float32)` する。

```python
q = q.astype(jnp.float32)
k = k.astype(jnp.float32)
v = v.astype(jnp.float32)
```

TanhNorm 後、必要なら model dtype に戻す。

```python
u = u.astype(model_dtype)
```

理由: cosine similarity と thresholding は小さな数値差で active / inactive が変わる。ここを bf16 にすると不安定になりやすい。

---

## 17. Configuration 推奨値

### 17.1 小規模検証

```yaml
d_model: 512
d_slot: 256
d_k: 64
d_v: 128
n_slots: 16
bank: short=8, mid=4, long=4
screened_layers: [middle layers only]
tau_init: 0.0
tanh_norm_cap: 1.0
lambda_screen_init: 0.01
use_value_unit_norm: true
use_age_mask: false
use_bank_bias: false
use_write_screening: false
```

### 17.2 中規模検証

```yaml
d_model: 1024
d_slot: 512
d_k: 64 or 128
d_v: 256
n_slots: 24
bank: short=8, mid=8, long=8
screened_layers: every 2 or 4 layers
tau_init: 0.0
tanh_norm_cap: 1.0
lambda_screen_init: 0.01
use_age_mask: true after MVP
use_write_screening: true after read-only phase
```

### 17.3 screened layers の推奨

全層に入れない。まず中間層だけに入れる。

例:

```text
L=12: screened_layers = [4, 8]
L=24: screened_layers = [6, 12, 18]
L=32: screened_layers = [8, 16, 24]
```

理由: 低層は局所表現、高層は logits に近い表現が強く、memory branch の影響が過大または不安定になりやすい。中間層から始める方が壊れにくい。

---

## 18. 実装ファイル構成

推奨構成:

```text
project/
  configs/
    screening_small.yaml
    screening_mid.yaml
  src/
    model/
      screening.py          # StateLevelScreening, helper functions
      screened_rwkv.py      # ScreenedRWKVLayer, ScreenedRWKVModel
      state.py              # Flax dataclasses for recurrent states
      rwkv_core.py          # adapter interface or placeholder core
    train/
      train_state.py
      train_step.py
      optimizer.py
      checkpoint.py
    infer/
      generate.py
      sampler.py
    eval/
      mqar.py
      needle.py
      slot_interventions.py
    tests/
      test_screening_math.py
      test_shapes.py
      test_scan.py
      test_state_update.py
```

---

## 19. 単体テスト仕様

### 19.1 数式テスト

#### Unit norm

入力 random tensor に対して、非ゼロ行の norm が 1 に近いこと。

```text
abs(norm(unit(x))-1) < 1e-4
```

#### Trim square

条件:

```text
sim = tau       -> rel = 0
sim < tau       -> rel = 0
sim = 1         -> rel ≈ 1
rel >= 0 always
```

#### TanhNorm

条件:

```text
norm(tanh_norm(z, C)) <= C + eps
small z では tanh_norm(z) ≈ z
```

### 19.2 shape test

ランダム入力で forward し、以下を確認。

```text
logits: [B, T, V]
new slots: [B, M, d_s]
rel_read: [T, B, M] or summarized shape
```

### 19.3 scan consistency

`T=1` を T 回 decode した結果と、`T` token を一括 prefill した結果の logits が近いこと。

```text
max_abs_diff < 1e-4  # dropout off, float32
```

### 19.4 no-softmax test

screening branch 内で slot dimension に対して softmax を使っていないことをコードレビューまたは test hook で確認する。

実装ポリシー:

```text
jax.nn.softmax(..., axis=slot_axis) is forbidden in StateLevelScreening
```

### 19.5 all-irrelevant test

slots と query が直交または tau が非常に高い場合、read-out がゼロ近傍になること。

```text
tau = 0.99
sim < tau
norm(u) ≈ 0
```

これは softmax attention との差を確認する重要テストである。

---

## 20. 受け入れ条件

MVP 実装の受け入れ条件:

1. `jax.jit` された train step が動く。
2. `jax.lax.scan` による prefill が動く。
3. `T=1` decode loop が動く。
4. logits shape が `[B, T, V]`。
5. screen state が forward ごとに更新される。
6. `rel_read` が `[B, M]` として取得できる。
7. `all-irrelevant` 条件で read-out がゼロ近傍になる。
8. TanhNorm 後の norm が cap を超えない。
9. loss が finite。
10. 100 step の toy training で NaN が出ない。

Phase 2 受け入れ条件:

1. write relevance が `[B, M]` で取得できる。
2. `suppress_write=True` で slot update が止まる。
3. long bank の平均 update norm が short bank より小さい。
4. read-only phase と read/write phase を config で切り替えられる。

---

## 21. コーディングエージェントへの実装順序

### Step 1: pure functions

以下を `screening.py` に実装し、単体テストを書く。

- `unit_norm`
- `bounded_tau`
- `theta_from_tau`
- `trim_square`
- `relevance_with_warmup`
- `tanh_norm`

### Step 2: state dataclasses

`state.py` に実装する。

- `LayerScreenState`
- `ModelScreenState`
- `init_screen_state`
- `tuple_set`

### Step 3: StateLevelScreening Module

MVP では read branch + slow updater のみ。

- `q_proj_r`
- `k_proj_r`
- `v_proj`
- `out_proj`
- `gate_proj`
- `delta_proj`
- `slot_embed`
- `tau_r_raw`
- `lambda_raw`
- `mu_by_bank_raw`

### Step 4: model integration

placeholder core でよいので、`ScreenedRWKVLayer` と `ScreenedRWKVModel` を作る。

### Step 5: train step

Optax training loop を作り、toy dataset で 100 step 動かす。

### Step 6: inference

prefill と single-token decode を作る。

### Step 7: Phase 2 write screening

`q_proj_w`, `k_proj_w`, `tau_w_raw` を追加し、`phase='read_write'` で使う。

### Step 8: logging and interventions

stats と intervention hooks を追加する。

---

## 22. 実装上の禁止事項

以下は禁止する。

1. slot dimension に softmax をかけること。
2. relevance を `rel / sum(rel)` で正規化すること。
3. write update で hard overwrite すること。
4. `slots[:, m] = ...` のような in-place update を使うこと。
5. 初期実装で全層に screening を入れること。
6. 初期実装で write screening と bank migration を同時に入れること。
7. tau を高く初期化すること。
8. TanhNorm を省略すること。
9. norm / similarity / thresholding を bf16 のまま処理すること。
10. active slot stats を記録せずに性能だけを見ること。

---

## 23. 実装者向け簡易疑似コード

```python
class StateLevelScreening(nn.Module):
    config: ScreeningConfig

    @nn.compact
    def __call__(self, x_t, h_base, state, *, phase, deterministic):
        cfg = self.config
        slots = state.slots
        ages = state.ages

        x_ln = nn.LayerNorm(dtype=jnp.float32)(x_t).astype(jnp.float32)
        slots_f = slots.astype(jnp.float32)

        # read q/k/v
        q_r = nn.Dense(cfg.d_k, use_bias=False, name='q_proj_r')(x_ln)
        k_r = nn.Dense(cfg.d_k, use_bias=False, name='k_proj_r')(slots_f)
        v = nn.Dense(cfg.d_v, use_bias=False, name='v_proj')(slots_f)

        q_r = unit_norm(q_r)
        k_r = unit_norm(k_r)
        if cfg.use_value_unit_norm:
            v = unit_norm(v)

        sim_r = jnp.einsum('bd,bmd->bm', q_r, k_r)

        tau_r_raw = self.param('tau_r_raw', lambda rng, shape: theta_from_tau(cfg.tau_init), ())
        tau_r = bounded_tau(tau_r_raw)
        rel_r = trim_square(sim_r, tau_r)

        if cfg.use_age_mask:
            rel_r = rel_r * compute_age_mask(ages, cfg)

        z = jnp.einsum('bm,bmd->bd', rel_r, v)
        u = tanh_norm(z, cap=cfg.tanh_norm_cap)

        gate = jax.nn.sigmoid(nn.Dense(cfg.d_model, name='gate_proj')(x_ln))
        read_out = nn.Dense(cfg.d_model, use_bias=False, name='out_proj')(u)

        lambda_raw = self.param('lambda_raw', nn.initializers.constant(-4.6), ())
        lambda_screen = jax.nn.softplus(lambda_raw)

        h = h_base + lambda_screen * gate * read_out.astype(h_base.dtype)

        # candidate update
        slot_embed = self.param('slot_embed', nn.initializers.normal(0.02), (cfg.n_slots, cfg.d_slot))
        slot_embed_b = jnp.broadcast_to(slot_embed[None, :, :], slots_f.shape)
        x_rep = jnp.broadcast_to(x_ln[:, None, :], (x_ln.shape[0], cfg.n_slots, cfg.d_model))
        h_rep = jnp.broadcast_to(h.astype(jnp.float32)[:, None, :], (x_ln.shape[0], cfg.n_slots, cfg.d_model))
        delta_in = jnp.concatenate([x_rep, h_rep, slot_embed_b], axis=-1)
        delta_s = jnp.tanh(nn.Dense(cfg.d_slot, name='delta_proj')(delta_in))

        mu = make_mu_by_slot(self, cfg)  # [M]

        if phase == 'read_only':
            update_strength = mu[None, :, None]
            new_slots = slots_f + update_strength * (delta_s - slots_f)
            new_ages = ages
            rel_w = jnp.zeros_like(rel_r)
        else:
            q_w_in = jnp.concatenate([x_ln, h_base.astype(jnp.float32)], axis=-1)
            q_w = nn.Dense(cfg.d_k, use_bias=False, name='q_proj_w')(q_w_in)
            k_w = nn.Dense(cfg.d_k, use_bias=False, name='k_proj_w')(slots_f)
            q_w = unit_norm(q_w)
            k_w = unit_norm(k_w)
            sim_w = jnp.einsum('bd,bmd->bm', q_w, k_w)
            tau_w_raw = self.param('tau_w_raw', lambda rng, shape: theta_from_tau(cfg.tau_init), ())
            tau_w = bounded_tau(tau_w_raw)
            rel_w = trim_square(sim_w, tau_w)
            update_strength = mu[None, :, None] * rel_w[:, :, None]
            new_slots = slots_f + update_strength * (delta_s - slots_f)
            new_ages = jnp.where(rel_w > 1e-3, 0.0, ages + 1.0)

        new_state = LayerScreenState(
            slots=new_slots.astype(slots.dtype),
            ages=new_ages,
            usage_ema=state.usage_ema,
        )

        stats = {
            'rel_read_mean': jnp.mean(rel_r),
            'rel_read_max': jnp.max(rel_r),
            'active_slots_mean': jnp.mean(jnp.sum(rel_r > 1e-3, axis=-1)),
            'z_norm_mean': jnp.mean(jnp.linalg.norm(z, axis=-1)),
            'u_norm_mean': jnp.mean(jnp.linalg.norm(u, axis=-1)),
            'tau_r': tau_r,
            'lambda_screen': lambda_screen,
            'rel_write_mean': jnp.mean(rel_w),
        }
        return h, new_state, stats
```

この疑似コードは完成コードではない。実装時には `nn.LayerNorm` の共有、param 初期化関数の shape、stats の scan 集約、dtype policy をプロジェクト規約に合わせて整理する。

---

## 24. 最終確認事項

コーディングエージェントは、実装完了時に次を報告する。

- 実装した phase: MVP / Phase 2 / Phase 3
- 使用した core: placeholder / RWKV core adapter
- screened layers
- slot 数と bank 構成
- tau 初期値
- TanhNorm cap
- slot dtype
- train step が JIT 済みか
- prefill と decode の両方が動くか
- all-irrelevant test の結果
- active slot histogram
- 100 step toy training の loss curve

この報告が揃わない実装は、研究実験には進めない。
