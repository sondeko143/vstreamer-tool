# 0090. テストの実行時間は収集 import の削減と既定の並列実行で詰める

- Status: Accepted
- Date: 2026-08-13
- 効力: 既定
- Related: [ADR-0088](0088-mirror-test-layout-on-package-tree.md)（この計測は同じ棚卸しの続き）; [ADR-0085](0085-gate-runtime-weight-on-outcome.md) / [ADR-0087](0087-weight-guarded-by-measurement-not-named-bans.md)（本体 import を動かさない理由）

## Context

スイート全体で 1,225 テスト。壁時計は同じコマンドで 187s / 98s / 37s と 5 倍ぶれるので、まず計測方法を決めてから内訳を取った（pytest 自身が報告する秒数、同条件で 3 回の最小値）。

**ぶれの正体**はファイルキャッシュの状態である。コールドでは `python -c pass` が 1.02s、ウォームでは 0.16s かかる。`python -S -c pass`（site 無効）でも 0.53s で、`-X importtime` は `reprlib` に 90ms、`linecache` に 68ms を計上する — どちらも本来 1ms 未満で、CPU ではなく**ファイルを開くたびのスキャン**の形をしている。この計測はそれを取り除けないので、内訳は最小値で読む。

ウォーム時の内訳:

| 区分 | 時間 | 中身 |
|---|---|---|
| 収集 | 17.2s | 86 テストモジュールのモジュールレベル import |
| 子プロセス系テスト | ~44s | `test_runtime_footprint` 26.6s / `test_main` 11.1s / `cli/test_entrypoint` 4.7s / `worker/test_vc` 1.6s |
| 純 CPU | ~9s | `test_resample` 6.0s / `test_preflight` 3.1s |

**収集コストはファイル数ではなく import 内容で決まる。** `tests/config` は 8 ファイルで 0.10s（12ms/件）、`tests/lib` は 28 ファイルで 14.8s（430ms/件）。ライブラリ単体の import コスト（プロセス起動 0.56s を含む実測）は `scipy.signal` 8.4s、`google.cloud.speech` 5.0s、`sounddevice` 3.3s、`faster_whisper` 3.0s、`onnxruntime` 2.1s、`av` 1.7s。

除外法で当たったのは 1 ファイルだけだった。`tests/lib/test_pitch_extract.py` を外すと `tests/lib` の収集が 14.76s → 8.75s になる。原因は `from scipy import signal` がモジュールスコープにあることで、**使用箇所は `signal.medfilt(f0, 3)` の 1 行**しかない。しかも scipy は `vspeech/` からは 1 件も import されていない実質テスト専用依存である（`pyproject.toml` の rvc extra に同趣旨のコメントが既にある）。

他のテストファイルの重い import はすべて、**テスト対象の本体モジュールが同じものを module スコープで import している**（`sounddevice` は `lib/audio.py` ほか 6 ファイル、`google.*` は `lib/gcp.py` と 3 worker、`onnxruntime` は `lib/{rvc,pitch_extract,onnx_session}.py`）。テスト側だけ遅延させても本体経由で入るので、削減にはならない。

子プロセス系は既に無駄がない。`test_runtime_footprint` の `_measure` は `@cache` 済みで、11 の測定パスは 5 つの parametrize テストで 1 回ずつしか spawn されない。`test_main` の 6 プロセスはエントリポイントを実際に起動することが目的で、過去にテスト 275 件が見逃した logger の cp1252/PIPE バグを摘出している。

## Decision

**2 つだけ決める。**

1. **テストモジュールは、対象の本体モジュールが読み込まないものを module スコープで import しない。** 収集時に払われるコストだからで、単一テストを選んだ実行でも払わされる。今回該当したのは `tests/lib/test_pitch_extract.py` の scipy 1 件で、使用箇所のテスト関数内へ移した。

2. **既定で並列実行する。** `pytest-xdist` を dev 依存に入れ、`addopts` に `-n auto` を足す。デバッグ時は `-n0` でプロセス内直列に戻せる（pdb と `-s` が使える）。

serial と `-n auto` を**交互に** 3 組回した対照計測（キャッシュ状態の偏りを打ち消すため）:

| | serial | `-n auto` (16 コア) |
|---|---|---|
| pair1 | 53.09s | 20.06s |
| pair2 | 37.09s | 20.29s |
| pair3 | 37.21s | 21.53s |
| **最小 / 中央** | **37.1 / 37.2s** | **20.1 / 20.3s** |

速さ（−45%）以上に効くのは**安定性**で、ばらつきが 43% から 7% に縮む。壁時計が「キャッシュミスの総和」ではなく「最長の 1 本」で決まるようになるためで、上のキャッシュ問題に対する実質的な緩和になっている。並列実行時も結果は同一（1208 passed / 16 skipped、exit 0）で、`--cov` との併用も動く。

## Alternatives rejected

- **`vspeech/` 本体の重い import を遅延させる** — 収集の残り 11.4s の大半はこれだが、動かさない。`sounddevice` / `google.*` / `onnxruntime` を関数内へ移すことは、[ADR-0085](0085-gate-runtime-weight-on-outcome.md) / [ADR-0087](0087-weight-guarded-by-measurement-not-named-bans.md) の重さゲートが**測っている対象そのものを変える**行為で、しかも ADR-0087 が名指しで警告している「エントリポイントは軽いまま、動いている worker だけが重い」形に近づける。テストの実行時間のために実行時の不変条件を動かす取引はしない。
- **`--dist loadfile` でファイル単位に固める** — `test_runtime_footprint` の `@cache` が worker をまたいで効くようになり、総 CPU は減る。しかしそのファイル 1 本が 26.6s あるので、担当した worker が critical path になって壁時計は既定の `load` より悪化する。実測しようとしたがこのマシンのノイズでは有意差が取れず（同一条件の `-n auto` が 20.1s と 89.7s の両方を出す）、既定値を離れる根拠が立たなかった。
- **並列を `poe test-fast` のような opt-in タスクにする** — 既定が遅いままなら、いちばん多い「普通に `pytest` を叩く」経路が改善しない。`-n0` という 1 フラグの逃げ道があるので、既定を速いほうに倒すコストは小さい。
- **アンチウイルス除外の案内だけ出して終わりにする** — 効果はおそらく最大だが、リポジトリの外にあり、他のマシンには波及せず、こちらから設定を変えるべきものでもない。並列化はそのマシン設定に依存せず効く。両立するので、除外は別途利用者に案内する。
- **何もしない** — 37s は耐えられなくはないが、実際にはコールドで 100〜190s になり、その状態がいつ来るかは予測できない。予測できない待ちは、テストを回す頻度そのものを下げる。

## Consequences

- 既定の `pytest` が 16 worker で走る。出力の順序は入り混じり、`-s` と pdb は使えない。**デバッグ時は `-n0`**、これは `addopts` を上書きするので追加設定は要らない。
- `test_runtime_footprint` の `_measure` の `@cache` は**プロセス単位**なので、既定の `--dist load` では同じ測定パスが複数の worker で spawn されうる。総 CPU は増え、壁時計は減る。16 コアあるうちは正しい取引だが、コア数の少ないマシンでは逆転しうる。
- worker ごとに収集（11.4s ぶんの import）を払うので、常駐メモリは worker 数に比例する。16 worker × numpy/grpc/google.cloud/onnxruntime で数 GB になる。`-n auto` はコア数に追従するため、メモリの少ないマシンでは `-n 4` などに落とす。
- CI への影響はない。このリポジトリの GitHub Actions は codeql のみで、pytest は走っていない。
- 収集はまだ 11.4s ある。これは本体モジュールの import コストそのもので、上の理由により**この決定の射程外**として残す。次に効く手があるとすれば本体側の依存を減らすことで、それは重さゲートと同じ方向の話になる。
