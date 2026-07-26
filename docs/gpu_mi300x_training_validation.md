# AMD MI300X 学習挙動・Pallas 実機検証（2026-07-19）

この文書は、現行 RWKV7M 実装を AMD Instinct MI300X 1基で動かした
短時間の実測記録である。主目的は、0.185B / 0.3B モデルが実データ上で
学習可能か、State-Level Screening が実際に利用されているか、ROCm 上の
Triton Pallas 経路が機能するかを、長期学習へ進む前に確認することだった。

## 結論

| 対象 | 実測結果 | 現時点の判定 |
| --- | --- | --- |
| 0.185B Screening v2 | 400 step、6,553,600 tokenをfiniteで完走。lossは20.768から5.789へ低下 | RWKV主経路は学習できる。ただしstep 26までにwrite admissionがほぼゼロになり、Screeningは実質停止した |
| 0.3B legacy read/write | 200 step、6,553,600 tokenをfiniteで完走。lossは18.686から6.091へ低下 | 学習は継続できる。ただしread activityはstep 107までにゼロになり、slotは高い重複を示した |
| 0.3B Screening v2 | 20-step診断はfiniteで完走したが、別の200-step予定runはstep 7からNaN | 本格学習へ進めない。数値不安定性とrun間の再現性を先に解決する必要がある |
| 0.185B Screening v5 core | recovery profileで2,000 step、32,768,000 tokenをfinite完走。lossは20.891から4.954へ低下し、step 2,000 gradientは423/423 leaf finite | 数値gateは通過。ただし1 / 16 slotだけを使い、memory-off loss差は+9e-6。memory品質gateは未通過 |
| 0.3B Screening v5 core | recovery profileで同一seed 30-step 2 runと400-step runをfinite完走。step 400 gradientは480/480 leaf finite | 旧all-writeは抑えたが、aggregate利用率3.125%と層間collapseを示唆。memory-off loss差は-4e-6で品質gate未通過 |
| ROCm Triton Pallas | WKV、Screening、training head、optimizerが実機lowering・実行可能 | AMD GPU対応の基盤は成立。ただしproduction形状のScreening parity gateは未通過 |
| 0.185B v5 synthetic retrieval安定性調査（2026-07-22） | 反復するstep 150--290帯のtraining NaNをfixed batchで隔離。単体call parityは20/20通過、複数Pallas WKV backwardの同一graph共存時のみ破綻 | 第一候補はROCm/Triton lowering・buffer aliasing・custom-call schedulingの相互作用。AMD autoはPallas forward + reference VJPへfail-close。実機full-step受け入れは次回session |

したがって、この測定は「MI300X上で現行JAX/Pallas学習スタックを実行できる」
ことを支持するが、「Screeningが学習品質を改善する」ことは支持しない。
0.185B v2と0.3B legacyの両方で、学習途中にmemory readが実質的に使われなく
なったためである。

## Screening v5 core follow-up

この節は、上記v2/legacy測定後に実装したportable
`screening-v5-core`を対象とする。v5は既存のv4 Pallas Screening kernelを
呼ばず、WKVだけが`pallas_gpu_triton`、Screening recurrenceはportable JAXで
動く。測定revisionは実装修正が`d6d169b`、fail-closed compute測定が
`ad7edce`である。

### 長context backwardの数値修正

初期のv5実装は、空memoryのdiagnostic normを`has_aux`へ返すだけでも、ゼロ点
の未定義VJPが`lax.scan` backwardを汚染した。diagnostic normを
`stop_gradient`で学習経路から分離した後、初期0.185Bの32 x 512 gradientは
423/423 leafがfiniteになった。

しかし7 step学習後のfinite checkpointへ別の固定32 x 512 batchを与えると、
loss 14.187はfiniteのまま、423 leaf中247 leaf、46,597,517値のgradientが
non-finiteになった。context 1、16、128、256ではfiniteで、問題batchの先頭
384 tokenでは再現した。全モデルをFP32にすると423/423 leafがfiniteになった
ため、長区間v5 recurrenceのmixed-precision経路が原因と判断した。

現在のv5 dtype境界は次である。

```text
RWKV主経路                       BF16 compute / BF16 parameter
Screening v5 projection compute  FP32
Screening v5 parameter storage   BF16
Screening recurrence vector/state/cotangent FP32
Screening出力から主経路への境界   BF16
```

この変更後、同じ修正前checkpointと同じ32 x 512 batchで423/423 leafがfiniteに
なった。さらに、0.185Bの50-step checkpointへ別固定batchを与えた診断でも、
loss 8.157、423/423 gradient leafがfiniteだった。曖昧度confidenceでは、正の
eligibilityをroute powerへ通した後にFP32 underflowして分布massがゼロになる
場合も安全に扱うguardを追加した。このguardは独立した未定義VJPを防ぐが、
上記production failureの主因ではなかった。

### v5 MiniPile短時間学習

共通条件はMiniPile magic sampler、`carry_state=false`、BF16 model parameter、
FP32 optimizer/gradient accumulation、full-logits XLA head、Optax、10-step
warmup、seed 42である。compileを含むstep 1をsteady throughputから除外した。

| model | batch x context | steps / tokens | loss first -> last | steady median | checkpoint gradient gate |
| --- | ---: | ---: | ---: | ---: | --- |
| 0.185B v5 | 32 x 512 | 50 / 819,200 | 20.891 -> 8.628 | 28,535 token/s | step 50、423/423 leaf finite |
| 0.3B v5（完走run） | 8 x 1024 | 30 / 245,760 | 18.618 -> 9.685 | 5,433 token/s | step 14/30、480/480 leaf finite |

0.3Bは同じ主要条件の先行runがstep 16からNaNになり、再runは30 stepをfiniteで
完走した。step 14までの軌道もわずかに異なるため、単一の完走runを長時間安定性
の証拠にはしない。ROCm/Pallasの非決定性、学習dynamics、残る数値境界のどれが
分岐を起こしたかは未確定である。

### v5 memory挙動

| model / step | admission | accepted novel | rejected write | slot utilization | read relevance | residual/base RMS | redundancy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.185B / 1 | 0.572 | 0.105 | 0.302 | 0.324 | 5.35e-3 | 4.49e-3 | 0.151 |
| 0.185B / 50 | 0.998 | 0.057 | 0.042 | 0.100 | 1.37e-5 | 2.54e-7 | 0.919 |
| 0.3B / 1 | 0.579 | 0.173 | 0.351 | 0.403 | 4.89e-3 | 3.07e-3 | 0.138 |
| 0.3B / 30 | 1.000 | 1.000 | 0.000 | 0.257 | 1.29e-2 | 1.97e-6 | 0.987 |

0.185Bではwrite branchは完全停止しなかったが、read relevanceと主経路に対する
memory residualが急減した。0.3Bではread relevance自体は残る一方、全tokenを
novelとして受理するwrite saturationと高いslot重複が発生し、主経路に対する
residualはほぼ消えた。どちらも「memory経路を品質改善に利用した」という
Phase 1 quality gateを支持しない。したがってretentionやv5 Pallasを先に実装
しても中心問題は解決しない。synthetic retrieval、memory-off counterfactual、
anti-starvation/read curriculumの検証を先行する。

この実測後、read starvationの直接要因として、4 tile、16 occupied slot、tile
key次元16でanalytic近似が`tau_read=0.946`まで上昇することを確認した。実装は
Gaussian null CDFのfamily-wise quantile（同条件で0.789）へ変更し、training
限定soft-to-hard read、temporary self-index loss、upper write budget、
redundancy-aware victim、memory-off評価を追加した。この時点ではローカルの
意味論・gradientテストだけを対象とするfollow-upであり、新構成のmemory利用と
品質改善は未検証だった。次節に、その後の再測定を分離して記録する。

### 2026-07-22 recovery profile再検証

上記follow-upをcommit `c08d970`で実装し、checkpoint counterfactual評価の
NNX trace境界をcommit `3e1c3ae`で修正した後、別のMI300X VFで再測定した。
ハードウェアはgfx942、HBM約192 GiB、softwareはPython 3.13.14、JAX / jaxlib
0.10.0、ROCm 7.14である。公開repositoryの
`agent/screening-pallas-benchmarks`を新規cloneし、README記載のBlinkDL
MiniPile `.idx` / `.bin`を使用した。WKVは`pallas_gpu_triton`、Screening v5は
`portable_jax_v5`、headは`full_logits_xla`、optimizerはOptaxである。

実機smokeではWKV Pallas、v4 Screening Pallas、training headの4 testが通過
した。checkpoint付きmemory-off evaluatorは、互換`apply`を`jax.jit`内部で
再構成せず、NNX moduleを明示引数とする`nnx.jit`へ変更し、CPU 5 testと
MI300X実評価の両方を通過した。

#### 実学習

| model / run | batch x context | steps / tokens | train loss first -> last | steady median | final gradient gate |
| --- | ---: | ---: | ---: | ---: | --- |
| 0.185B短期 | 32 x 512 | 50 / 819,200 | 20.891 -> 8.770 | 25,781 token/s | 423/423 leaf finite |
| 0.185B curriculum全期間 | 32 x 512 | 2,000 / 32,768,000 | 20.891 -> 4.954 | 25,822 token/s | 423/423 leaf finite |
| 0.3B再現run 1 | 8 x 1,024 | 30 / 245,760 | 18.618 -> 9.584 | 5,030 token/s | 480/480 leaf finite |
| 0.3B再現run 2 | 8 x 1,024 | 30 / 245,760 | 18.618 -> 9.594 | 約5,030 token/s | 480/480 leaf finite |
| 0.3B延長run | 8 x 1,024 | 400 / 3,276,800 | 18.618 -> 6.669 | 5,111 token/s | 480/480 leaf finite |

0.3Bの同一seed 30-step 2 runはどちらもfiniteだった。最終loss差は0.00933、
全step中の最大loss差は0.571であり、以前のstep 16 NaNは再現しなかったが、
bitwiseな決定性を示す結果でもない。0.185B step 2,000と0.3B step 400の
gradient gateは、学習batchとは別の固定MiniPile batchを使い、lossと全gradient
値がfiniteであることをfail-closedで確認した。

#### memory利用

| model / step | novel | accepted novel | rejected write | write budget rate | slot utilization | read relevance | residual/base RMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.185B / 1 | 5.322% | 4.065% | 26.538% | 0.2284 | 21.387% | 7.072e-3 | 5.094e-3 |
| 0.185B / 50 | 0.635% | 0.470% | 0.867% | 0.0223 | 13.867% | 1.932e-2 | 8.057e-6 |
| 0.185B / 500 | 0.195% | 0.195% | 0% | 0.0438 | 6.250% | 2.627e-3 | 4.166e-6 |
| 0.185B / 2,000 | 0.195% | 0.195% | 0% | 0.0279 | 6.250% | 3.746e-3 | 9.422e-6 |
| 0.3B / 1 | 7.434% | 5.231% | 30.029% | 0.2316 | 24.902% | 7.267e-3 | 3.720e-3 |
| 0.3B / 30 | 50.049% | 0.049% | 50.000% | 0.0499 | 3.125% | 2.753e-3 | 4.162e-7 |
| 0.3B / 400 | 50.049% | 0.049% | 50.000% | 0.0459 | 3.125% | 4.909e-3 | 5.372e-7 |

0.185Bではstep 500以降、1 token / 512 tokenと一致するnovel rateと1 / 16の
slot utilizationが固定された。0.3Bでは2 screened layerのaggregateが、
`novel=0.50048828125`、`accepted=0.00048828125`、`rejected=0.5`、
`slot utilization=0.03125`へ固定された。層別metricがないため断定はできないが、
この正確な分数は、一方の層がほぼ全tokenをnovelとして拒否し、他方の層だけが
最初の1 slotを占有した状態と整合する。少なくとも、全層平均のwrite budgetが
目標付近にあることは、各層が健全である証拠にならない。

旧runの全novel/all-writeと0.987 redundancyは解消したが、slot redundancyが0に
なった理由はslotが十分に分離されたからではなく、比較対象となるoccupied slotが
ほぼ1個しかないためである。recovery profileは退化解を、過剰write・高重複から
過少allocation・層間collapseへ移した。memory capacity利用のquality gateは
依然として不合格である。

#### memory-off counterfactual

| checkpoint | evaluation tokens | active loss | off - active loss | prediction RMS delta |
| --- | ---: | ---: | ---: | ---: |
| 0.185B step 50 | 163,840 | 8.598566 | +1.6e-5 | 4.906e-3 |
| 0.185B step 2,000 | 327,680 | 4.587761 | +9e-6 | 8.884e-3 |
| 0.3B step 30 run 1 | 81,920 | 9.757456 | -5e-6 | 3.607e-3 |
| 0.3B step 30 run 2 | 81,920 | 9.733965 | +1.4e-5 | 3.447e-3 |
| 0.3B step 400 | 163,840 | 6.481575 | -4e-6 | 2.741e-3 |

符号はrun間で一貫せず、loss差はすべて1.6e-5以下である。予測logitは厳密な
同一ではないが、現在のmemory branchがこの同一MiniPile評価batch上のcross
entropyを一貫して改善する証拠は得られなかった。held-out品質比較ではない。

#### compute-only complete step

固定device batch、warmup 2、測定5、GC無効、iterationごとの同期を使った。

| model | benchmark LR | forward | backward | optimizer | complete step | throughput | finite gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.185B、32 x 512 | 1e-5 | 135.339 ms | 479.449 ms | 19.445 ms | 612.273 ms | 26,759 token/s | pass |
| 0.185B、32 x 512 | 1e-4 | 135.404 ms | 480.196 ms | 19.756 ms | 692.546 ms | 23,658 token/s | **fail** |
| 0.3B、8 x 1,024 | 1e-4 | 378.170 ms | 1,213.370 ms | 25.536 ms | 1,665.190 ms | 4,920 token/s | pass |

0.185Bの1e-4測定はprepared loss/gradientはfiniteだったが、同じ固定batchを
warmupと測定で繰り返し更新した後のlossとtrain stateがnon-finiteになった。
したがって23,658 token/sは採用値ではない。実MiniPileの2,000-step runはより
高いpeak LRでも異なるbatchを用いてfinite完走したため、これは直ちに通常学習の
NaNを意味しない。一方、以前は同じ1e-4条件が通過していたため、固定batch stress
の数値回帰として未解決事項に残す。

#### この再検証から必要になった変更

次の実装では、global平均後のanti-starvation lossを強めるだけでは不十分である。

1. `novel`、admission、occupancy、write budget、read/residualをscreened layer別・
   bank別に記録し、lossも各層へ適用してから集約する。
2. empty capacityがあるbootstrap期間だけ、各層・各bankの最低occupied slot数または
   allocation rateを要求する。target到達後は0へannealし、永久writeは強制しない。
3. 最初のslotが全queryのmatch先になることを防ぐため、capacity fill中はnoveltyを
   learned matchだけへ依存させず、empty-slot explorationまたは割当予約を導入する。
4. memory residualを単に非ゼロへ固定せず、synthetic retrieval上で
   counterfactual改善を伴う範囲に限ってbranch-scale curriculumを検討する。
5. retention、v5 checkpoint redesign、v5 Pallas化は、複数slot・複数layerの利用と
   positive counterfactualが確認されるまで後回しにする。

この結果はportable v5の数値実行可能性を支持するが、memoryが主機能として利用
されるという中心仮説は支持しない。

### fail-closed compute-only

測定CLIは、prepared loss、prepared gradient、最終loss、更新後train stateの
全てをfinite gateへ含める。benchmark iteration数とoptimizer scheduleの全期間
も分離し、固定batch比較では明示的にpeak LRを上書きできる。

0.185B v5、32 x 512、warmup 2、測定5、optimizer horizon 10,000、peak LR
1e-4では全finite gateを通過した。

| phase | median |
| --- | ---: |
| forward | 126.797 ms |
| backward | 432.916 ms |
| optimizer | 20.245 ms |
| complete step | 696.240 ms |
| complete-step throughput | 23,532 token/s |

同じ条件のbatch 8はprepared loss/gradientがfiniteだったが、固定batchを連続更新
した最終loss/stateがnon-finiteになったため、表示された9,443 token/sを採用
しない。実MiniPile runの28,535 token/sとcompute-onlyの23,532 token/sも、
batch内容、学習率、反復境界が異なる別系列であり、直接比率を性能主張に使わない。

## 2026-07-22 synthetic retrieval安定性調査とAMD WKV fail-close

recovery profile再検証の後、memory品質評価の主vehicleをMiniPile magic
sampler + `carry_state=false`からdocument単位のsynthetic delayed key-value
retrievalへ変更し、同じMI300X系ホストで学習安定性を切り分けた。magic +
`carry_state=false`は毎シーケンスで空memoryから始まるため、chunk境界を越える
保持というScreening固有の価値を評価できない。以後この構成は数値・throughput
gateとしてのみ使う。

### 実験系

- 0.185B preset、batch 8 x 512(4,096 token/step)、LR 1e-4一定
- Screening layer 6、16 slots、`screening-v5-core`
- staged activation: `activation_step=100`、warmup 100 step、
  Screening optimizer LR multiplier 0.1
- retrieval data: train 8,192 / eval 1,024 documents、1 document = 4 chunk x
  128 token、key 256種、distractor 16、answer maskはtoken 4以降
- 評価はdocument_sequential + `carry_state=true`、memory-on/off同時比較

### 調査中に確定した独立バグと修正

1. 低load時のread threshold warm-upがwrite/novelty閾値にも適用され、最初の
   occupied slotがほぼ全tokenへmatchして1-slot collapseを自己強化していた。
   warm-upをread専用に分離した(commit `cc3e42c`)。
2. `--model-config`使用時にoptimizer系CLI overrideが無視されていた
   (commit `828b9e7`)。resume時にexecution overrideが適用されない問題
   (commit `8b761a0`)と、resume後checkpoint metadataが旧設定を記録する問題
   (commit `3717e8f`)も修正した。これらの影響を受けた一部ablation runは無効
   として除外した。
3. Pallas WKV backwardのinverse state reconstructionは、decayがFP32で0へ
   roundする入力で非有限化するアルゴリズム欠陥だった。VJP内forward再実行と
   token別FP32 state tapeへ置換し、GPU/TPU両backendへ回帰テストを追加した
   (commit `2599ad5`)。修正後のPallasはreference比約2.6倍を維持した。
4. victim allocationのmasked min--max正規化は、同値統計のtieで最大約1.17e6の
   人工gradientを作った。spanが`eps`以下なら0へ落とすtie-safe化を実装した
   (commit `2263dcd`)。`norm_eps`も独立設定へ分離した(既定値は不変)。
5. 学習CLIへraw gradient / 更新後parameterのfinite診断とfail-closed停止を
   追加した(commit `d13836c`、`5fb2ad4`)。

### aux ablationの要約

trunk-only checkpoint 100から同一条件でresumeした有効runの結果:

| arm | 結果 | 要点 |
| --- | --- | --- |
| no-aux | step 400までfinite | ただしnovelty=1.0の全書き込みから約4--5%へ漂流し、slot utilization 0.25、residual/base RMS比median 4.94e-6の退化解 |
| self-index-only | step 300までfinite | step 112--119にraw gradient L2最大5.5586e11のspike、直後にほぼ全拒否へcollapse |
| budget-only | step 251でraw gradient NaN | 直前spikeなし、window内budget loss=0のまま失敗 |
| all-aux | step 176 / 290でNaN | step 287--288に4.3e8--2.2e10のspike後、289で正常値へ戻り290で非有限化 |

CEに対するmemory residualが極小(~1e-5)のままでは、routing系ゲートの実効的な
学習信号はaux lossだけになる。安定した中間write率を長期維持した構成はまだ
ない。aux値の急峻さ単独ではNaNを説明できず、auxはrouting/stateの軌道を変える
因子として扱う。

### fixed-batch隔離: step 251 NaNの帰属

budget-only runの最終健全checkpoint 250と固定batchで、失敗を決定論的に
再現・分解した。

- forward loss 8.2013はfinite。CE単独の微分で423 leaf中247 leaf、
  48,508,683値のgradientが非有限。非有限はlayer 0--6、embedding、layer 6
  write-side 16 leafに局在し、layer 7--11とLM headはfinite。
- parameter storageだけFP32へ昇格しても完全に同数・同一pathで非有限。
- global `eps`を1e-5または1e-3にすると有限化するが、`norm_eps`だけの変更では
  不変。tie-safe min--max適用後も不変。full FP32化では有限(grad L2 27.19)。
- `RWKV7M_WKV_BACKEND=reference`でWKVだけportable化すると、既定epsのまま
  全423 leafがfinite(grad L2 27.2002)。full FP32・eps=1e-5・reference WKVの
  3変種が同水準のgradient norm(~27.2)へ合流するため、このcheckpoint状態の
  真の勾配は良条件であり、失敗は数値conditioningではなく実行系にあると
  判断した。
- layer 6のWKV callをcaptureすると、着弾するactivation cotangentがcapture
  時点で既に非有限だった。従って「layer 6 Pallas WKV backwardが発生源」と
  いう一次解釈は撤回した。
- finiteなfull step(全layer Pallas forward + reference pullback)でlayers
  7--11の全20 WKV callをcaptureし、各callをPallas/reference VJPへ単体再投入
  すると20/20でfiniteかつ近似一致した。単一kernel数式バグはこの再現では
  支持されない。
- 集合判別: layers 9--11のみreference pullback化しても、layers 7--8のみでも
  失敗(それぞれ48,508,683 / 48,507,915値)。layers 7--11を同時にreference化
  した場合のみ0非有限で全leaf finite。

結論として、step 251 NaNはv5 Screening recurrence単体の数式バグではなく、
単体では正しい複数のPallas WKV backward custom callが同一full graphに共存
した場合にのみ破綻する。現時点の第一候補はROCm/Tritonのlowering、buffer
aliasing、custom-call schedulingまたはその相互作用であり、単一layer/kernel
数式への局在は主張しない。

### AMD fail-closed dispatch

対策として、WKV backendへ`pallas_gpu_triton_reference_vjp`を追加した。
forwardはTriton Pallasを維持し、reverse-mode pullbackだけportable reference
を使う。AMD deviceの`auto` dispatchはこのhybrid backendへfail-closeした。
NVIDIA L40S/Adaの`pallas_gpu_triton`、Hopper/Blackwellの`pallas_gpu_mosaic`、
TPUの`pallas_tpu`は変更していない。AMDでも明示的な`pallas_gpu_triton`指定は
benchmark・調査用に残る。

このhybrid backendのMI300X実機full-step再検証は、費用を抑えるため次回の
短時間GPU sessionへfail-closedのまま残している。現時点の根拠はfixed-batch
隔離と、CPU interpretを含むローカル回帰(repository全suite 279 passed、
5 skipped)である。

### 再現資産

fixed batch、corrected checkpoint metadata、layer 6 / layers 7--11 capture、
20 call分の解析JSONは`.tmp/mi300x-budget250-wkv/`(50 files、NPZ 25本、
128,046,793 bytes)としてローカル回収済みである。大容量のcheckpoint由来
一時成果物のため`.tmp/`はgitignoreされ、repositoryへはcommitしない。

### retrieval品質の現状

write threshold修正後のstep 205 checkpointに対するheld-out streaming
retrieval評価(128 eval step、ctx 128、memory-on/off同一batch列)は、
loss 9.2866、accuracy 1/256(chance)、memory-off loss delta +1.04e-4、
prediction RMS delta 2.688e-3だった。旧dead branchと異なりmemoryはlogitへ
非ゼロの因果効果を持つが、retrieval精度の改善はまだ示されていない。
memory品質gateは引き続き未通過である。

## 検証対象

### ソフトウェアとハードウェア

- GPU: AMD Instinct MI300X VF、1 device、約192 GiB HBM
- Python: 3.13.14
- JAX / jaxlib: 0.10.0 / 0.10.0
- Flax: 0.12.7
- Optax: 0.2.8
- 実行backend: ROCm GPU
- WKV backend: pallas_gpu_triton
- v2/legacy Screening backend: pallas_gpu_triton
- v5 core Screening backend: portable_jax_v5
- training loss: full-logits XLA
- optimizer: Optax
- 基準revision: 069781cecb5e0b9e8fa95d0a1961b61d1b08180a

検証中にAMD Tritonの制約が見つかったため、実際のrunは上記revisionへ現在の
working-tree修正を加えた状態で行った。修正は、公開上3-bankのlogitをkernel
内部だけ4要素へpadすること、Tritonで未対応だったreduce_orとdynamic sliceを
reductionベースの式へ置き換えること、backwardで余分なbank gradientを公開
3-bank形状へ戻すことである。公開APIと3-bankの意味論は変更していない。

ホストには複数のROCm library setが混在しており、既定環境では
ncclCommWindowDeregisterの未定義symbolによりJAX pluginの初期化に失敗した。
測定では次のlibrary pathを明示した。

~~~bash
export LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib:/opt/rocm/core-7.14/lib64
~~~

v5 follow-upを実行した同系統ホストでは、実在するprefixに合わせて
`/opt/rocm-7.0.2/core-7.14/lib{,64}`を指定した。

これは当該レンタル環境固有の回避策であり、通常のROCm導入手順として一般化
しない。

### データセット

READMEに記載されたBlinkDL MiniPile tokenized datasetを使用した。

- index:
  https://huggingface.co/datasets/BlinkDL/minipile-tokenized/resolve/main/rwkv_vocab_v20230424/minipile.idx
- tokens:
  https://huggingface.co/datasets/BlinkDL/minipile-tokenized/resolve/main/rwkv_vocab_v20230424/minipile.bin
- item数: 1,010,500
- token数: 1,498,226,207
- token dtype: uint16

実学習はmagic sampler、carry_state=falseで行った。評価用held-out dataset、
validation loss、perplexityは今回の短時間runには含めていない。

## Pallas correctness gate

小型の実GPU accelerator testはMI300X上で通過した。

- Screening real-accelerator test: 1 passed、12.44秒
- WKV real Pallas test: passed
- full-XLA training head path: passed
- optimizer path: passed

一方、0.3B v2のproduction recurrence形状に近い次の試験はfail-closed parity
gateを通過しなかった。

~~~text
T=128, B=1, slots=16, d_slot=256
d_k=64, d_v=128, read_tiles=4
checkpoint_interval=16
~~~

主な結果は次の通り。

| 指標 | 結果 |
| --- | ---: |
| output u max abs | 7.629e-6 |
| output slots max abs | 1.490e-8 |
| 最大gradient relative L2 | 0.002071 |
| initial ages gradient max abs | 7.068 |
| scalar loss difference | 6.332e-4 |
| parity gate | fail |

出力誤差とrelative gradient errorは小さいが、initial ages gradientの絶対誤差
上限0.01とloss差上限1e-4を超えた。したがって、次のtimingは原因分析用の
診断値であり、採用済みperformance gateとしては扱わない。

| recurrence window | Pallas median | reference median | 比率 |
| --- | ---: | ---: | ---: |
| forward | 2.176 ms | 16.367 ms | 7.52x |
| forward + backward | 6.710 ms | 76.608 ms | 11.42x |

絶対誤差だけを理由に閾値を緩めるのではなく、age gradientのscale、loss差、
checkpoint reconstructionの寄与を分離して再検証する必要がある。

## Compute-only complete-step

固定batchを事前にdeviceへ配置し、compileとhost transferを除外した
complete-step測定である。各cellはwarmup 2回、測定5回、Python GC無効、
各iteration同期を使用した。forward、backward、optimizerの個別windowとは
別に、barrierを挟まないvalue-and-grad + optimizer全体を測った。

共通条件はBF16、remat_blocks=true、sequence_chunk_size=128、
full-logits XLA head、Optaxである。dataset sampling、host-to-device transfer、
compilation、checkpoint I/O、logging、host metricsは含まない。

| モデル | 構成 | batch x context | parameter | complete step | throughput |
| --- | --- | ---: | ---: | ---: | ---: |
| 0.185B | screeningなし | 8 x 512 | 183,956,736 | 244.230 ms | 16,771 token/s |
| 0.185B | Screening v2 | 8 x 512 | 184,351,693 | 275.060 ms | 14,891 token/s |
| 0.185B | Screening v2 | 16 x 512 | 184,351,693 | 317.623 ms | 25,792 token/s |
| 0.185B | Screening v2 | 32 x 512 | 184,351,693 | 463.974 ms | 35,312 token/s |
| 0.3B | legacy read/write | 8 x 1024 | 297,738,764 | 662.743 ms | 12,361 token/s |
| 0.3B | legacy read/write | 16 x 1024 | 297,738,764 | 793.579 ms | 20,646 token/s |
| 0.3B | legacy read/write | 32 x 1024 | 297,738,764 | 1,129.504 ms | 29,011 token/s |
| 0.3B | Screening v2 | 32 x 1024 | 295,066,394 | 1,176.046 ms | 27,863 token/s |

0.185Bのbatch 8では、v2はscreeningなしより11.2%低いthroughputだった。
batchを8から32へ増やすとv2 throughputは2.37倍になり、MI300Xの並列能力を
小batchでは使い切れていないことが分かる。

0.3B v2の固定batch測定では最終lossがNaNになった。step latency自体は同期
して得た実測値だが、finiteな学習stepの性能として採用してはならない。
同じbatch 32のlegacy比で約4.0%遅いという値も、安定性を直した後に再測定
する必要がある。

## MiniPile上の学習挙動

### 0.185B Screening v2

条件:

- config: configs/rwkv7m-0.185b-screening-v2.json.example
- batch x context: 32 x 512
- 400 step、16,384 token/step、合計6,553,600 token
- learning rate: 1e-3から1e-5、10-step warmup、cosine
- sequence chunk: 128
- seed: 42

| step | loss | throughput | admission mean | rejected write | slot utilization | memory u norm |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 20.768 | 387 token/s | 1.117e-1 | 0.952 | 0.951 | 6.528e-1 |
| 26 | 9.628 | 34,634 token/s | 5.793e-6 | 1.000 | 0.000 | 0.000 |
| 51 | 8.071 | 34,594 token/s | 7.054e-6 | 1.000 | 0.000 | 0.000 |
| 101 | 7.192 | 34,791 token/s | 5.562e-8 | 1.000 | 0.000 | 0.000 |
| 201 | 6.715 | 34,246 token/s | 6.016e-10 | 1.000 | 0.000 | 0.000 |
| 301 | 5.984 | 34,275 token/s | 2.751e-7 | 1.000 | 0.000 | 0.000 |
| 400 | 5.789 | 34,634 token/s | 2.771e-7 | 1.000 | 0.000 | 0.000 |

step 1はcompileを含むためthroughput比較から除外する。lossはfiniteのまま低下
したが、step 26までに全tokenがnovelかつrejectedとなり、applied write、
slot utilization、memory output normがゼロになった。以降のloss改善は
Screening memoryの効果ではなく、ほぼRWKV主経路の学習によるものと解釈する
のが妥当である。

このrunだけからは、v2が品質を改善したとも、screeningなしbaselineより学習
効率がよいとも言えない。むしろ現行初期化・admission・routing条件では、
0.185Bがmemory branchを停止させる退化解を選んだ証拠になっている。

### 0.3B Screening v2

条件:

- config: configs/rwkv7m-0.3b-screening-v2.json.example
- batch x context: 32 x 1024
- 32,768 token/step
- learning rate: 1e-3から1e-5、10-step warmup、cosine
- sequence chunk: 128
- seed: 42

20-step診断runでは、step 3にloss 42.561の大きなspikeがあったものの、
step 20のlossは12.074で、全stepがfiniteだった。steady-state throughputは
約27,250 token/sだった。

同じ初期seedと主要shapeで200 stepを予定した別runは、step 1と2までは
20-step runと同じlossだったが、step 3からわずかに軌道が分かれ、step 7で
最初のNaNを記録した。NaN後はadmission、route mass、memory normもNaNとなり、
手動停止した。CSVには64 step分、run summaryの最終flushには60 step分が残り、
checkpointは保存していない。

20-step完走は長期安定性の証拠にはならない。また、200-step予定runとの分岐
原因は未特定であり、model/routing dynamics、Pallas kernel、
checkpoint reconstruction、optimizer、GPU上の非決定性のいずれかへ
現時点で帰属させることはできない。

### 0.3B legacy read/write

条件:

- config: configs/rwkv7m-0.3b.json.example
- batch x context: 32 x 1024
- 200 step、32,768 token/step、合計6,553,600 token
- seed: 42

| step | loss | throughput | read relevance | memory u norm | slot utilization | slot redundancy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 18.686 | 818 token/s | 7.753e-3 | 8.325e-2 | 1.000 | 0.241 |
| 49 | 7.617 | 28,655 token/s | 2.596e-6 | 3.878e-5 | 0.423 | 0.991 |
| 81 | 6.685 | 28,647 token/s | 0.000 | 0.000 | 0.447 | 0.992 |
| 107 | 6.637 | 28,666 token/s | 0.000 | 0.000 | 0.479 | 0.993 |
| 141 | 6.212 | 28,665 token/s | 0.000 | 0.000 | 0.483 | 0.990 |
| 200 | 6.091 | 28,657 token/s | 0.000 | 0.000 | 0.490 | 0.989 |

legacy recurrenceはfiniteで完走し、slotへのwriteも継続した。しかしread
relevanceとmemory output normはstep 81までに表示精度上ゼロになった。
slot utilizationは約49%残る一方、slot cosine redundancyは約0.989である。
すなわち、slotを更新する計算は続いても、内容が強く重複し、主経路へ読み
戻されていない。

この結果もScreeningの品質寄与を支持しない。legacyがv2より安定だったことは
確認できるが、parameter-matched baseline、held-out validation、複数seedが
ないため、loss値をアーキテクチャ間の優劣として比較してはならない。

## 経過時間と概算費用

価格は検証時に提示された1.29 USD/hourを使用した。

| run | run summary経過 | all-in throughput | 最終steady throughput | 概算費用 |
| --- | ---: | ---: | ---: | ---: |
| 0.185B v2、6.55M token | 4.00分 | 27,275 token/s | 34,634 token/s | 0.086 USD |
| 0.3B legacy、6.55M token | 4.65分 | 23,471 token/s | 28,657 token/s | 0.100 USD |

steady throughputを単純外挿すると、1B tokenは0.185B v2で約8.02時間・
10.35 USD、0.3B legacyで約9.69時間・12.50 USDとなる。20 token/parameterを
機械的に当てはめた場合は、それぞれ約29.6時間・38.2 USD、約57.7時間・
74.4 USDである。

この外挿はcapacity planning用であり、推奨学習token数を示さない。compile、
validation、定期checkpoint、長時間のthermal/host変動、障害復旧、v2の
不安定性を含まない。特に0.3B v2には、finiteな長時間throughputがないため
学習費用を外挿しない。

## 現時点の主要課題

優先順位は次の通り。

1. 0.3B v2の最初の非finite値を、parameter、gradient、optimizer state、
   Screening carryごとに特定する。
2. 同じ固定batchとseedで複数runを行い、step 3以降の軌道分岐が再現するか
   確認する。
3. 0.185Bのadmission collapseを、logit、gradient、threshold、lambda、
   routing massの時系列で診断する。memory利用を強制する変更は、baselineの
   意味を変えるため診断前に導入しない。
4. 0.3B legacyのread collapseとslot redundancyを調べる。writeが続くことを
   memory利用の証拠として扱わない。
5. production recurrence parity gateのloss差とage gradient差を解決する。
   原因が分かるまでfail-closed閾値を緩めない。
6. screeningなし、legacy、v2、parameter-matched FFN controlを同じtoken
   budget、複数seed、held-out validation、memory-off counterfactualで比較する。

## 主張できる範囲

確認できた事実:

- MI300X/ROCm上でJAX 0.10.0のTriton Pallas WKVとScreeningをlowering・実行
  できる。
- 0.185B v2と0.3B legacyはMiniPile上の6.55M-token短時間runをfiniteで完走
  できる。
- MI300Xではbatch拡大によりcomplete-step throughputが大きく改善する。
- 現行設定では0.185B v2のwrite admissionと0.3B legacyのread activityが
  学習初期に消失する。
- 0.3B v2には再現性を含む重大な数値安定性問題が残る。

まだ主張できないこと:

- ScreeningがscreeningなしRWKV baselineよりvalidation loss、学習効率、
  長距離記憶を改善すること。
- AMD Pallas kernelがNVIDIA CUDA版やTPU版より速いこと。
- 0.3B v2が長時間安定して学習できること。
- production recurrence形状が現在のfail-closed parity基準を満たすこと。
- 単一seed・6.55M tokenのtraining lossから最終モデル品質を予測できること。

生成されたJSON/CSVとcheckpointはrepositoryのruntime dependencyにはしない。
再現可能な設定は追跡対象のexample configと本書へ残し、研究上の採用判断は
今後のmatched validation matrixに基づいて行う。

## 2026-07-26 delayed-retrieval追試と実装反映

上記MiniPile診断後、document-aligned delayed key/value retrievalを0.185Bで
実施した。旧tracked profileは`all-write -> empty-memory -> all-write`へ遷移し、
checkpoint 200ではmemory-on/offのloss、accuracy、logitsが完全に一致した。
step 262では最大gradient要素がfiniteな`1.4979e22`まで増加し、naiveなFP32
二乗和だけがoverflowした。したがって旧profileはmemory効果gateを通過していない。

この反証を受け、次のコード変更を行った。

- per-layer hard write率を観測するbounded incremental PI admission controller;
- controllerとadmission-floor/write-budget gradient lossの同時有効化を拒否;
- legacy write budgetのutilization hard switchを連続scaleへ変更;
- activation後2,000 stepのScreening入力stop-gradient;
- four-tile effective initial residual floorを0.1へ増加;
- self-index lossをtracked configで無効化;
- overflow-safe scaled gradient norm;
- max-absolute-gradient guard超過時のparameter/moment/stepを含む完全update skip;
- controller bias/EMA/errorのfull-checkpoint round-tripとportable export境界。

tracked target hard write率は0.05、controller初期`admission_init`は0.16、
max-absolute-gradient guardは1e6である。いずれも初期実験値であり、MI300Xで
再測定していない。構造修正の詳細と未検証事項は
[`state_level_screening_v5_design.md`](state_level_screening_v5_design.md)および
repository rootの`.tmp_report.txt` §28を参照する。

修正後profileについて、現時点で主張できるのは設定・gradient境界・checkpoint
契約のローカル検証だけである。hard write率の収束、multi-slot利用、read energy、
memory residual、held-out causal delta、retrieval accuracyの改善は、同一dataset、
seed、checkpoint評価protocolでのMI300X再実験まで未確認である。
