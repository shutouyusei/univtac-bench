# univtac-bench

UniVTAC(Isaac Lab + TacEx の visuo-tactile simulation benchmark)の fork。
実機 repo `so101-tactile` の policy を simulation で比較するための adapter・変換・実験 script を置く。
upstream は `univtac/UniVTAC`、作業の base は `isaac51` branch(Isaac Sim 5.1、RTX 50 系対応)。
`main` は upstream の Isaac Sim 4.5 版なので使わない。

## 兄弟 repo と役割

| repo | 役割 |
|---|---|
| `so101-tactile` | 実機側。lerobot policy plugin(`plugins/`)、record/train/eval pipeline。**policy の実装はここに置く** |
| `univtac-bench`(この repo) | simulation 側。UniVTAC の環境、`policy/<Name>/deploy_policy.py` の adapter、HDF5 ↔ LeRobotDataset 変換、実験 script |
| `usable-info` | offline 解析(tactile 依存 probe の計器)。sim の checkpoint にも当てる |
| `reflex-act` | 研究 note・論文(git 管理外)。方針は `notes/` を読む |

policy 本体はこの repo に書かない。`pip install -e ../so101-tactile/plugins/<policy>` で editable に入れ、
ここでは UniVTAC の `BasePolicy` interface(`docs/Deploy.md`)に包む adapter だけを持つ。

## Claude の役割

開発と開発ログ(commit message / PR 本文)のみ。結果の解釈・研究判断は user。
仮説は user が一文で書いてから実験を回す。解析 script は実行前に固定する。

## この repo での sim の目標(2026-09-23 決定)

TacForcing(arXiv 2608.25798、π0.5 + streaming action expert + EATA mask、**code 非公開**)を baseline に:

1. TacForcing の枠で「tactile を無視する学習」が本当に起きていないかを検証する。
   本文 Table 2 では固定 tactile が base より悪化(42% → 31%)しており、EATA で回復しているだけの可能性がある。
   tactile 依存の直接測定(接触 frame で tactile を zero / shuffle → action 変化、attention 配分)は本文に無い。
2. ImplicitRDP(arXiv 2512.10946、公式 code `Chen-Wendi/ImplicitRDP`)の tactile 条件付き補助 loss(VRR、force prediction)を
   TacForcing の枠に足すと改善するか。
3. VRR は F/T の物理量(力 → 仮想目標の偏差)を要するので、物理量なしで同じ効果を出せる簡易な代替があるか(発展)。

想定する比較は 2×2: {事前学習 tactile encoder(UniVTAC 配布の shared encoder)/ scratch} × {補助 loss なし / あり}。
各 cell で成功率と tactile 依存を測る。sim は GelSight Mini の画像型 tactile なので、
AnySkin(実機、15 ch 磁気、400 Hz)固有の主張はここでは検証できない。sim で決めるのは構造(mask、streaming、loss)まで。

## 未決定(user が決める)

- base model: 論文どおり π0.5 か、`policy/smolvla` がある SmolVLA か。π0.5 なら公開数値(6 task 平均 65%)と直接比較できる。
- TacForcing 再実装の合格基準: Table 2 の pattern(base 43% → streaming mask なし 51% → EATA 60%、sim)が再現できること。
- 補助 loss の GelSight 画像への翻訳(何を「力」とみなすか)。
- task の選択。実機の USB 挿入 ↔ Insert HDMI、Lego ↔ Insert Hole / Pull-out Key を候補にする。

## 環境構築(agent が最初にやること)

1. `bash scripts/install.sh`(isaac51 版: conda env `UniVTAC`、Python 3.11、Isaac Sim 5.1.0.0、Isaac Lab 2.3.0、
   `third_party/TacEx` を改変 source から build、libuipc の binding を build、cuRobo)。
   この機は Ubuntu 24.04、RTX 5090、driver 580。`docs/Installation.md` と `docs/isaacsim_5_1_migration.md` を先に読む。
2. `python scripts/smoke_isaac51.py` で Isaac が起動することを確認。
3. README の手順で `isaac51/<task>/` の demo(各 task 100 episode、HDF5)を取得。
   配布 checkpoint は Isaac Sim 4.5 版のみで 5.1 の data とは互換でない(README の注記)。
4. `bash eval_policy.sh <task> <task_config> <policy_config> <gpu>` が既存 policy(`policy/ACT` など)で回ることを確認。
5. ここまでの所要時間・詰まった点・修正した file を `docs/setup-log.md` に残す(user が読む)。

上の 1〜4 が通るまで policy adapter には手を付けない。

## 規約

- task ごとに branch、PR で user が review。`isaac51` を base にする(upstream の更新は `git fetch upstream && git rebase upstream/isaac51`)。
- commit message は変更の理由が一文で分かるように。`wip` / `fix` 禁止。
- upstream の file を書き換えるときは、なぜ upstream のままでは動かないかを commit message に書く。
- `experiments/*/README.md` の仮説・結果節は user が書く。Claude は読まない・書かない。

## 参照

- UniVTAC: 論文 arXiv 2602.10093、repo `univtac/UniVTAC`、site https://univtac.github.io/
- TacForcing: arXiv 2608.25798(project site のみ、code なし)
- ImplicitRDP: arXiv 2512.10946、code `Chen-Wendi/ImplicitRDP`(RA-L 2026、HF に dataset と checkpoint)
- AnySkin: arXiv 2409.08276 / Sparsh-Skin: arXiv 2505.11420(磁気 skin の SSL encoder、Xela)
- 方針の全文: `~/Desktop/Robotics/reflex-act/notes/`(2026-09-22 overview と、それ以降の更新)
