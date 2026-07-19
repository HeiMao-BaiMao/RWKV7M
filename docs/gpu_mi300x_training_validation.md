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
| ROCm Triton Pallas | WKV、Screening、training head、optimizerが実機lowering・実行可能 | AMD GPU対応の基盤は成立。ただしproduction形状のScreening parity gateは未通過 |

したがって、この測定は「MI300X上で現行JAX/Pallas学習スタックを実行できる」
ことを支持するが、「Screeningが学習品質を改善する」ことは支持しない。
0.185B v2と0.3B legacyの両方で、学習途中にmemory readが実質的に使われなく
なったためである。

## 検証対象

### ソフトウェアとハードウェア

- GPU: AMD Instinct MI300X VF、1 device、約192 GiB HBM
- Python: 3.13.14
- JAX / jaxlib: 0.10.0 / 0.10.0
- Flax: 0.12.7
- Optax: 0.2.8
- 実行backend: ROCm GPU
- WKV backend: pallas_gpu_triton
- Screening backend: pallas_gpu_triton
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
