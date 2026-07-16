# subtitle の OBS バックエンド 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** subtitle worker に OBS バックエンドを足し、obs-websocket 経由で OBS の `Text (GDI+)` ソースへ字幕を push する。tkinter 非依存でヘッドレス起動できるようにする。

**Architecture:** `EventType.subtitle` は不変。`subtitle.worker_type`（`TK` | `OBS`、既定 `TK`）で tts / transcription と同じ形にディスパッチする。純粋な状態機械を `lib/subtitle_state.py` に抽出して両バックエンドで共有し、tk 固有の描画は `worker/subtitle_tk.py` へ移す。OBS バックエンドは `lib/obs_ws.py`（自前の obs-websocket 5.x クライアント）を使い、config を表示スタイルの権威として push する。

**Tech Stack:** Python 3.14 / asyncio / pydantic v2 / websockets 16.x / obs-websocket 5.x protocol / pytest（`asyncio_mode = "auto"`）

**Spec:** [2026-07-16-subtitle-obs-websocket-backend-design.md](../specs/2026-07-16-subtitle-obs-websocket-backend-design.md)

**ADR:** [0040](../../adr/0040-subtitle-obs-backend-via-worker-type.md)（配線）/ [0041](../../adr/0041-subtitle-obs-config-authority.md)（スタイル権威と境界）/ [0042](../../adr/0042-subtitle-obs-failure-tiers.md)（失敗の階層化）/ [0043](../../adr/0043-obs-websocket-client-in-house.md)（依存）。**4 本とも `Proposed`。Task 9 で昇格 / supersede を判定する。**

## Global Constraints

- Python は **3.14 のみ**（`requires-python = ">=3.14,<3.15"`）。
- Import は **1 行 1 つ**（ruff `force-single-line = true`）。`from x import y` 形式。
- **Pydantic v2 のみ**。`parse_obj` / `.dict()` / `.json()` / `root_validator` / `orm_mode` / `Field(env=)` / `json_encoders` は使用禁止。
- 型検査は **ty**（pyright ではない）。`uv run ty check`（プロジェクト全体。ファイル単位だとテストの型エラーを見逃す）。
- テストは `asyncio_mode = "auto"`。`@pytest.mark.asyncio` は不要。
- ブランチは既に `feat/subtitle-obs-backend`。spec と ADR は `740c70a` でコミット済み。
- 全タスク完了後のゲート: `uv run poe check`。既知の許容済み失敗は 2 件（torch の CVE `GHSA-rrmf-rvhw-rf47`、vr2_config の deadcode）。**それ以外の新規失敗は許容しない。**
- **tk バックエンドの挙動は変えない**（spec の非ゴール）。Task 3 / 4 はコードの移動のみで、ロジックに手を入れない。

## ファイル構成

| ファイル | 責務 |
|---|---|
| `vspeech/lib/obs_ws.py` | **[新]** obs-websocket 5.x クライアント。ハンドシェイク・認証・requestId 相関。`websockets` への依存をここに閉じ込める |
| `vspeech/lib/subtitle_state.py` | **[新]** 純粋な状態機械。`Text` / `Texts` / `ingest_text` / `update_display_sec` / `how_many_should_we_pop` / `age_panels`。tkinter 非依存 |
| `vspeech/lib/obs_text_settings.py` | **[新]** config → `text_gdiplus` 設定 dict の純粋なマッピング。OBS バージョン依存の形式をここだけに閉じ込める |
| `vspeech/worker/subtitle.py` | **[改]** ディスパッチャと `create_subtitle_task` のみ。**tkinter を import しない** |
| `vspeech/worker/subtitle_tk.py` | **[新/移]** tk バックエンド。既存コードを逐語移動 |
| `vspeech/worker/subtitle_obs.py` | **[新]** OBS バックエンド。接続・再接続・push ループ |
| `vspeech/config.py` | **[改]** `SubtitleWorkerType` / `SubtitleObsConfig` / `SubtitleConfig.worker_type` / `.obs`、`export_to_toml` の secret 展開 |
| `vspeech/preflight.py` | **[改]** `_check_subtitle` を追加し `_CHECKERS` に登録 |

---

### Task 1: 実機 OBS で設定形式を確定する（spike / 使い捨て） — ✅ 完了 2026-07-16

ADR-0041 が「本設計で最も外れやすい」と記録した箇所を、コードを書く前に潰す。**このタスクの成果物はコードではなく、Task 6 が使う観測値。**

**なぜ最初にやるか:** OBS のソース（`obs-text.cpp`）を読む限り `color` は `0x00RRGGBB`（プレーン RGB）で保存され、内部で `rgb_to_bgr()` される。しかしこれは読解であって観測ではない。**そして出荷デフォルトの `#ffffff` / `#000000` は回文なので、RGB/BGR を取り違えても気づけない。** 非対称色で確かめる必要がある。

**前提:** OBS が起動していて、Tools → obs-websocket Settings で WebSocket サーバが有効。パスワードを控えておく。シーンに `Text (GDI+)` ソースを 1 つ作り、名前を `vspeech-spike` にする。

## 実測結果（2026-07-16、この spike は実行済み。以下が Task 6 の入力）

環境: **OBS Studio 32.1.2 / obs-websocket 5.7.3 / rpcVersion 1**

| 観測項目 | 結果 |
|---|---|
| `inputKind` | **`text_gdiplus_v3`**（`SetInputSettings` は問題なく通る。v2/v3 の懸念は杞憂だった）|
| `color` の並び | **`0x00BBGGRR`（BGR）。ソース読解の RGB は誤りだった** |
| ソース不在時の `code` | **600**（ADR-0042 / 計画の前提どおり）|
| `align` / `valign` | **文字列で通る**（`"center"` / `"bottom"` がそのまま往復）|
| `font.flags` | `1` が往復する（`OBS_FONT_BOLD`）|
| 利用可能な text kind | `text_gdiplus_v3`, `text_ft2_source_v2` |

**色の測定（目視ではなく `GetSourceScreenshot` のピクセル）:**

| config | 保存した int | レンダリング結果 | 判定 |
|---|---|---|---|
| `#ff8000` | `0x0080FF` | `rgb(255,128,0)` | PASS |
| `#0080ff` | `0xFF8000` | `rgb(0,128,255)` | PASS |
| `#ffffff` | `0xFFFFFF` | `rgb(255,255,255)` | PASS（**回文なので無意味**）|

最初に `0xFF8000` を素の RGB として書き込んだところ `rgb(0,128,255)` になった。
→ **`hex_color_to_obs_int` は `(b << 16) | (g << 8) | r` でなければならない。**
`#ffffff` は反転してもしなくても PASS するので、この並びは非対称色でしか守れない。

**その他の観測（実装には影響しないが記録）:**
- 未設定のソースの `GetInputSettings` は `inputSettings: {}` を返す（既定値は暗黙）。
  → `validate_sources` が `GetInputSettings` で存在確認する方式は、素のソースでも機能する。
- **`align: "bogus"` は `code 100` で受理される。OBS は align を検証しない。**
  我々は値を生成する側なので実害は無いが、OBS からの fail-loud は期待できない。
- ユーザーの OBS には `sub` / `sub_rus` という `window_capture` が存在する。
  = 現行の tk 字幕ウィンドウを 2 枚キャプチャしている。OBS バックエンドはこれらを不要にする。

---

<details>
<summary>実行に使った spike（記録用。再実行は不要）</summary>

**Files:**
- Create: `<scratchpad>/obs_spike.py`（使い捨て。リポジトリにコミットしない）

- [ ] **Step 1: spike スクリプトを書く**

```python
"""使い捨て: 実機 OBS の text_gdiplus 設定の実際の形式を観測する。"""

import asyncio
import base64
import hashlib
import json
import sys

from websockets.asyncio.client import connect

URL = sys.argv[1] if len(sys.argv) > 1 else "ws://127.0.0.1:4455"
PASSWORD = sys.argv[2] if len(sys.argv) > 2 else ""
SOURCE = "vspeech-spike"


def auth_string(password: str, salt: str, challenge: str) -> str:
    secret = base64.b64encode(hashlib.sha256((password + salt).encode("utf-8")).digest())
    return base64.b64encode(
        hashlib.sha256(secret + challenge.encode("utf-8")).digest()
    ).decode("utf-8")


async def main():
    async with connect(URL) as ws:
        hello = json.loads(await ws.recv())
        print("Hello:", json.dumps(hello, indent=2, ensure_ascii=False))
        d = {"rpcVersion": 1}
        auth = hello["d"].get("authentication")
        if auth:
            d["authentication"] = auth_string(PASSWORD, auth["salt"], auth["challenge"])
        await ws.send(json.dumps({"op": 1, "d": d}))
        print("Identified:", await ws.recv())

        async def req(request_type, data=None):
            await ws.send(json.dumps({
                "op": 6,
                "d": {"requestType": request_type, "requestId": "spike", "requestData": data or {}},
            }))
            while True:
                m = json.loads(await ws.recv())
                if m["op"] == 7 and m["d"]["requestId"] == "spike":
                    return m["d"]

        # 1. 現状を読む -> inputKind と既定の形が分かる
        got = await req("GetInputSettings", {"inputName": SOURCE})
        print("\n=== GetInputSettings (before) ===")
        print(json.dumps(got, indent=2, ensure_ascii=False))

        # 2. 非対称色を書き込む。#ff8000 = 純RGB なら 16744448 (0xFF8000)
        #    BGR で解釈されるなら OBS 上では水色っぽく見えるはず。
        print("\n=== SetInputSettings (color=0xFF8000, bold, outline) ===")
        print(json.dumps(await req("SetInputSettings", {
            "inputName": SOURCE,
            "inputSettings": {
                "text": "色の確認 ABC",
                "font": {"face": "Meiryo UI", "size": 48, "flags": 1},
                "color": 0xFF8000,
                "opacity": 100,
                "outline": True,
                "outline_size": 1,
                "outline_color": 0x0000FF,
                "outline_opacity": 100,
                "align": "center",
                "valign": "bottom",
                "bk_color": 0x000000,
                "bk_opacity": 0,
                "extents": True,
                "extents_cx": 800,
                "extents_cy": 200,
                "extents_wrap": True,
            },
            "overlay": True,
        }), indent=2, ensure_ascii=False))

        # 3. 読み戻す
        print("\n=== GetInputSettings (after) ===")
        print(json.dumps(await req("GetInputSettings", {"inputName": SOURCE}), indent=2, ensure_ascii=False))

        # 4. 存在しないソース -> 失敗時の code を観測する (ADR-0042 が 600 を前提にしている)
        print("\n=== missing source ===")
        print(json.dumps(await req("GetInputSettings", {"inputName": "vspeech-does-not-exist"}), indent=2, ensure_ascii=False))


asyncio.run(main())
```

- [ ] **Step 2: 実行して観測する**

Run: `uv run --isolated --python 3.14 --with websockets python <scratchpad>/obs_spike.py ws://127.0.0.1:4455 <password>`

**記録すること（Task 6 の入力になる）:**
1. `GetInputSettings` の `inputKind`（`text_gdiplus_v2` か `text_gdiplus_v3` か）
2. **OBS の画面で文字がオレンジ（`#ff8000`）か水色（`#0080ff`）か。** オレンジなら `color` は RGB＝ソース読解どおり。水色なら BGR で、`0xBBGGRR` へ反転が必要
3. 輪郭が青（`#0000ff`）か赤か（同上の確認）
4. 太字になっているか（`flags: 1` = `OBS_FONT_BOLD` の検証）
5. 読み戻した `color` の値が書いた値と一致するか
6. 存在しないソースの `requestStatus.code`（ADR-0042 は **600** を前提にしている）
7. `align` / `valign` が文字列（`"center"` / `"bottom"`）で通るか、int を要求されるか

- [ ] **Step 3: 観測結果が読解と食い違ったら ADR を直す**

`color` が BGR だった / `code` が 600 でなかった / `align` が int だった場合、**Task 6 のマッピングを観測に合わせる**。ADR-0041 / 0042 の該当記述は `Proposed` なので、この時点で本文を実態に合わせて修正してよい（Task 9 で `Accepted` に昇格する前提）。

- [ ] **Step 4: 観測結果を計画に追記してコミット**

観測値をこの計画の Task 6 冒頭に追記する（後続タスクの実装者はこの計画しか読まない）。

```bash
git add docs/superpowers/plans/2026-07-16-subtitle-obs-websocket-backend.md
git commit -m "docs(subtitle): record observed text_gdiplus settings format from a real OBS"
```

</details>

---

### Task 2: websockets 依存と obs-websocket クライアント

**Files:**
- Modify: `pyproject.toml`（`dependencies` に 1 行）
- Create: `vspeech/lib/obs_ws.py`
- Test: `tests/test_obs_ws.py`

**Interfaces:**
- Produces:
  - `build_auth_string(password: str, salt: str, challenge: str) -> str`
  - `class ObsTransport(Protocol)` — `send(str)` / `recv() -> str | bytes` / `close()`
  - `class ObsWsClient` — `__init__(transport: ObsTransport, timeout: float = 5.0)`, `async identify(password: str) -> None`, `async request(request_type: str, request_data: dict[str, Any] | None = None) -> dict[str, Any]`
  - `class ObsProtocolError(Exception)` / `class ObsIdentifyError(ObsProtocolError)` / `class ObsRequestError(ObsProtocolError)` / `class ObsResourceNotFoundError(ObsRequestError)`

- [ ] **Step 1: 依存を追加する**

`pyproject.toml` の `dependencies` に追加（`audioop-lts` の行の前、アルファベット順は問わない — 既存もそうなっていない）:

```toml
    # obs-websocket 5.x クライアント (ADR-0043)。依存ゼロ。websockets の API 変更
    # (websockets.client.connect -> websockets.asyncio.client.connect) の影響は
    # lib/obs_ws.py に閉じ込める。
    "websockets>=14,<17",
```

Run: `uv sync --all-extras`
Expected: `websockets` が入る。**`uv sync` は vspeech 実行中だと `.venv` を置換できず bare os error 5 で失敗する。** 稼働中のパイプラインがある場合は先にユーザーへ確認する。

- [ ] **Step 2: 失敗するテストを書く**

`tests/test_obs_ws.py`:

```python
import json
from collections import deque

import pytest

from vspeech.lib.obs_ws import OP_IDENTIFY
from vspeech.lib.obs_ws import OP_REQUEST
from vspeech.lib.obs_ws import ObsIdentifyError
from vspeech.lib.obs_ws import ObsRequestError
from vspeech.lib.obs_ws import ObsResourceNotFoundError
from vspeech.lib.obs_ws import ObsWsClient
from vspeech.lib.obs_ws import build_auth_string

# obs-websocket 5.x の認証アルゴリズム:
#   1. base64(sha256(password + salt))    -> base64 secret
#   2. base64(sha256(secret + challenge))
# 期待値は simpleobsws (obs-websocket 本家 IRLToolkit) の実装と行レベルで一致
# することを確認した上で固定した回帰ベクタ (ADR-0043)。password は文字列
# "supersecretpassword"、salt/challenge はこのテスト専用の固定値で、実在の OBS
# のものではない (.gitleaks.toml で値ごと allowlist 済み)。
AUTH_PASSWORD = "supersecretpassword"
AUTH_SALT = "lM1GncleQOaCu9lT1yeUZhFYnqhsLLP1G5lAGo3ixaI="
AUTH_CHALLENGE = "+IxH4CnCiqpX1rM9scsNynZzbOe4KhDeYcTNS3PDaeY="
AUTH_EXPECTED = "1Ct943GAT+6YQUUX47Ia/ncufilbe6+oD6lY+5kaCu4="


class FakeObsServer:
    """スクリプト化した obs-websocket サーバ。ネットワークも OBS も使わない。

    「送られてきたものに応答する」形にしてある。おかげでテストはクライアントが
    採番した requestId を知る必要がなく、応答は send の時点で積まれるので
    待ち合わせも要らない。
    """

    def __init__(self, *, require_auth: bool = True, greet: bool = True):
        self.sent: list[dict] = []
        self.closed = False
        self._outgoing: deque[str] = deque()
        self._responses: deque[tuple[bool, int, dict]] = deque()
        if greet:
            hello: dict = {"obsWebSocketVersion": "5.5.0", "rpcVersion": 1}
            if require_auth:
                hello["authentication"] = {
                    "salt": AUTH_SALT,
                    "challenge": AUTH_CHALLENGE,
                }
            self._outgoing.append(json.dumps({"op": 0, "d": hello}))

    def script_response(
        self, *, ok: bool = True, code: int = 100, data: dict | None = None
    ):
        """次の Request に返す応答を積む。"""
        self._responses.append((ok, code, data or {}))

    def inject(self, message: dict):
        """次の応答より前に届く生メッセージ (イベント等) を積む。"""
        self._outgoing.append(json.dumps(message))

    async def send(self, message: str) -> None:
        m = json.loads(message)
        self.sent.append(m)
        if m["op"] == OP_IDENTIFY:
            self._outgoing.append(
                json.dumps({"op": 2, "d": {"negotiatedRpcVersion": 1}})
            )
        elif m["op"] == OP_REQUEST:
            ok, code, data = (
                self._responses.popleft() if self._responses else (True, 100, {})
            )
            self._outgoing.append(
                json.dumps(
                    {
                        "op": 7,
                        "d": {
                            "requestType": m["d"]["requestType"],
                            # 採番された id をそのまま返す = テスト側の受け渡し不要。
                            "requestId": m["d"]["requestId"],
                            "requestStatus": {"result": ok, "code": code},
                            "responseData": data,
                        },
                    }
                )
            )

    async def recv(self) -> str:
        if not self._outgoing:
            raise AssertionError("client recv'd more than the fake scripted")
        return self._outgoing.popleft()

    async def close(self) -> None:
        self.closed = True


def test_build_auth_string_matches_the_reference_vector():
    assert build_auth_string(AUTH_PASSWORD, AUTH_SALT, AUTH_CHALLENGE) == AUTH_EXPECTED


async def test_identify_sends_rpc_version_and_auth():
    server = FakeObsServer()
    await ObsWsClient(server).identify(AUTH_PASSWORD)
    assert server.sent == [
        {"op": 1, "d": {"rpcVersion": 1, "authentication": AUTH_EXPECTED}}
    ]


async def test_identify_omits_auth_when_server_does_not_ask():
    server = FakeObsServer(require_auth=False)
    await ObsWsClient(server).identify("")
    assert server.sent == [{"op": 1, "d": {"rpcVersion": 1}}]


async def test_identify_raises_when_server_wants_auth_but_password_is_empty():
    server = FakeObsServer()
    with pytest.raises(ObsIdentifyError, match="password"):
        await ObsWsClient(server).identify("")


async def test_identify_raises_when_hello_is_not_first():
    server = FakeObsServer(greet=False)
    server.inject({"op": 5, "d": {"eventType": "Surprise"}})
    with pytest.raises(ObsIdentifyError):
        await ObsWsClient(server).identify("")


async def test_request_sends_the_request_and_returns_response_data():
    server = FakeObsServer(require_auth=False)
    client = ObsWsClient(server)
    await client.identify("")
    server.script_response(data={"inputSettings": {"text": "hi"}})
    got = await client.request("GetInputSettings", {"inputName": "x"})
    assert got == {"inputSettings": {"text": "hi"}}
    assert server.sent[-1]["op"] == 6
    assert server.sent[-1]["d"]["requestType"] == "GetInputSettings"
    assert server.sent[-1]["d"]["requestData"] == {"inputName": "x"}


async def test_request_generates_a_unique_request_id_per_call():
    server = FakeObsServer(require_auth=False)
    client = ObsWsClient(server)
    await client.identify("")
    server.script_response()
    server.script_response()
    await client.request("A")
    await client.request("B")
    ids = [m["d"]["requestId"] for m in server.sent if m["op"] == 6]
    assert len(set(ids)) == 2


async def test_request_ignores_events_and_other_request_ids():
    server = FakeObsServer(require_auth=False)
    client = ObsWsClient(server)
    await client.identify("")
    server.inject({"op": 5, "d": {"eventType": "InputNameChanged"}})
    server.inject(
        {
            "op": 7,
            "d": {
                "requestType": "GetInputSettings",
                "requestId": "someone-elses-id",
                "requestStatus": {"result": True, "code": 100},
                "responseData": {"nope": True},
            },
        }
    )
    server.script_response(data={"mine": True})
    assert await client.request("GetInputSettings", {"inputName": "x"}) == {"mine": True}


async def test_request_raises_resource_not_found_on_600():
    server = FakeObsServer(require_auth=False)
    client = ObsWsClient(server)
    await client.identify("")
    server.script_response(ok=False, code=600)
    with pytest.raises(ObsResourceNotFoundError):
        await client.request("GetInputSettings", {"inputName": "nope"})


async def test_request_raises_generic_error_on_other_failures():
    server = FakeObsServer(require_auth=False)
    client = ObsWsClient(server)
    await client.identify("")
    server.script_response(ok=False, code=400)
    with pytest.raises(ObsRequestError) as e:
        await client.request("SetInputSettings", {"inputName": "x"})
    assert e.value.code == 400


async def test_request_returns_empty_dict_when_there_is_no_response_data():
    server = FakeObsServer(require_auth=False)
    client = ObsWsClient(server)
    await client.identify("")
    server.script_response(data={})
    assert await client.request("SetInputSettings", {"inputName": "x"}) == {}
```

- [ ] **Step 3: テストが失敗することを確認する**

Run: `uv run pytest tests/test_obs_ws.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vspeech.lib.obs_ws'`

- [ ] **Step 4: 実装する**

`vspeech/lib/obs_ws.py`:

```python
"""obs-websocket 5.x クライアント (ADR-0043)。

必要なのは Hello(0)/Identify(1)/Identified(2) のハンドシェイクと
Request(6)/RequestResponse(7) の往復だけで、イベント購読・バッチ・msgpack は
使わない。websockets の API 変更 (過去に websockets.client.connect ->
websockets.asyncio.client.connect の移行があった) の影響をこのファイルに
閉じ込めるため、呼び出し側は ObsTransport 越しにしか触らない。
"""

import base64
import hashlib
import json
from asyncio import wait_for
from typing import Any
from typing import Protocol
from uuid import uuid4

RPC_VERSION = 1

OP_HELLO = 0
OP_IDENTIFY = 1
OP_IDENTIFIED = 2
OP_REQUEST = 6
OP_REQUEST_RESPONSE = 7

STATUS_RESOURCE_NOT_FOUND = 600


class ObsProtocolError(Exception):
    """obs-websocket との対話が想定外の形になった。"""


class ObsIdentifyError(ObsProtocolError):
    """Identify が成立しなかった (認証失敗・RPC バージョン不一致など)。

    リトライしても直らない種類なので、呼び出し側は fail-loud に扱う (ADR-0042)。
    """


class ObsRequestError(ObsProtocolError):
    def __init__(self, request_type: str, code: int, comment: str):
        self.request_type = request_type
        self.code = code
        self.comment = comment
        super().__init__(f"{request_type} failed: code={code} {comment}")


class ObsResourceNotFoundError(ObsRequestError):
    """指定した input などが OBS に存在しない (code 600)。"""


class ObsTransport(Protocol):
    """websockets の ClientConnection が満たす最小の口。"""

    async def send(self, message: str) -> None: ...

    async def recv(self) -> str | bytes: ...

    async def close(self) -> None: ...


def build_auth_string(password: str, salt: str, challenge: str) -> str:
    """obs-websocket 5.x の認証文字列を作る。

    仕様の手順どおり:
      1. password + salt を sha256 して base64 -> base64 secret
      2. secret + challenge を sha256 して base64
    """
    secret = base64.b64encode(hashlib.sha256((password + salt).encode("utf-8")).digest())
    return base64.b64encode(
        hashlib.sha256(secret + challenge.encode("utf-8")).digest()
    ).decode("utf-8")


class ObsWsClient:
    def __init__(self, transport: ObsTransport, timeout: float = 5.0):
        self._transport = transport
        self._timeout = timeout

    async def _send(self, op: int, d: dict[str, Any]) -> None:
        await self._transport.send(json.dumps({"op": op, "d": d}))

    async def _recv(self) -> dict[str, Any]:
        raw = await wait_for(self._transport.recv(), timeout=self._timeout)
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            message = json.loads(raw)
        except ValueError as e:
            raise ObsProtocolError(f"OBS から不正な JSON: {e}") from e
        if not isinstance(message, dict) or "op" not in message:
            raise ObsProtocolError(f"OBS から不正なメッセージ: {message!r}")
        return message

    async def identify(self, password: str) -> None:
        message = await self._recv()
        if message["op"] != OP_HELLO:
            raise ObsIdentifyError(f"Hello を期待したが op={message['op']} が来た")
        auth = message["d"].get("authentication")
        d: dict[str, Any] = {"rpcVersion": RPC_VERSION}
        if auth:
            if not password:
                raise ObsIdentifyError(
                    "OBS が認証を要求していますが subtitle.obs.password が空です"
                )
            d["authentication"] = build_auth_string(
                password, auth["salt"], auth["challenge"]
            )
        await self._send(OP_IDENTIFY, d)
        message = await self._recv()
        if message["op"] != OP_IDENTIFIED:
            raise ObsIdentifyError(f"Identified を期待したが op={message['op']} が来た")

    async def request(
        self, request_type: str, request_data: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        request_id = str(uuid4())
        await self._send(
            OP_REQUEST,
            {
                "requestType": request_type,
                "requestId": request_id,
                "requestData": request_data or {},
            },
        )
        while True:
            message = await self._recv()
            # イベント (op 5) や他リクエストの応答は捨てる。
            if message["op"] != OP_REQUEST_RESPONSE:
                continue
            d = message["d"]
            if d.get("requestId") != request_id:
                continue
            status = d["requestStatus"]
            if not status.get("result"):
                code = status.get("code", 0)
                comment = status.get("comment", "")
                if code == STATUS_RESOURCE_NOT_FOUND:
                    raise ObsResourceNotFoundError(request_type, code, comment)
                raise ObsRequestError(request_type, code, comment)
            return d.get("responseData") or {}
```

- [ ] **Step 5: テストが通ることを確認する**

Run: `uv run pytest tests/test_obs_ws.py -v`
Expected: 11 passed

- [ ] **Step 6: 型検査とリント**

Run: `uv run ruff format . && uv run ruff check . && uv run ty check`
Expected: 新規の指摘なし

- [ ] **Step 7: コミット**

```bash
git add pyproject.toml uv.lock vspeech/lib/obs_ws.py tests/test_obs_ws.py
git commit -m "feat(subtitle): add an in-house obs-websocket 5.x client (ADR-0043)"
```

---

### Task 3: 純粋な状態機械を `lib/subtitle_state.py` へ抽出する

**Files:**
- Create: `vspeech/lib/subtitle_state.py`
- Modify: `vspeech/worker/subtitle.py`（移した定義を削除し、再 export する）
- Modify: `tests/test_subtitle_ingest.py:11-13`（import 先）
- Test: `tests/test_subtitle_state.py`（`age_panels` の新規テストのみ）

**Interfaces:**
- Produces:
  - `Text`（`value: str = ""`, `display_remain_sec: float = 0`）
  - `Texts`（`tag` / `anchor` / `config` / `bb_width` / `bb_height` / `display_remain_sec` / `values`、`coord_x` / `coord_y` / `texts` プロパティ）
  - `update_display_sec(current_sec, current_text, add_text, config) -> float`
  - `how_many_should_we_pop(texts: deque[Text], max_length: int) -> int`
  - `ingest_text(texts: dict[str, Texts], message: WorkerInput) -> Texts`
  - `age_panels(panels: dict[str, Texts], elapsed_sec: float) -> list[Texts]` **[新規]**
  - `next_expiry_sec(panels: dict[str, Texts]) -> float | None` **[新規]**

- [ ] **Step 1: `age_panels` / `next_expiry_sec` の失敗するテストを書く**

既存の `Text` / `Texts` / `ingest_text` は挙動不変の移動なので、既存テストが回帰を守る。新規追加分だけテストを書く。

`tests/test_subtitle_state.py`:

```python
from vspeech.config import SubtitleTextConfig
from vspeech.lib.subtitle_state import Text
from vspeech.lib.subtitle_state import Texts
from vspeech.lib.subtitle_state import age_panels
from vspeech.lib.subtitle_state import next_expiry_sec


def make_panels() -> dict[str, Texts]:
    return {
        "n": Texts(
            tag="text",
            anchor="s",
            config=SubtitleTextConfig(anchor="s"),
            bb_width=300,
            bb_height=200,
        ),
        "s": Texts(
            tag="translated",
            anchor="n",
            config=SubtitleTextConfig(anchor="n"),
            bb_width=300,
            bb_height=200,
        ),
    }


def test_age_panels_only_ages_the_head_entry():
    # tk バックエンドの 30fps ループは values[0] だけを減らす。壁時計でも同じ意味を保つ。
    panels = make_panels()
    panels["n"].values.append(Text(value="first", display_remain_sec=1.0))
    panels["n"].values.append(Text(value="second", display_remain_sec=5.0))
    age_panels(panels, 0.5)
    assert panels["n"].values[0].display_remain_sec == 0.5
    assert panels["n"].values[1].display_remain_sec == 5.0


def test_age_panels_pops_expired_head_and_returns_the_panel():
    panels = make_panels()
    panels["n"].values.append(Text(value="gone", display_remain_sec=0.25))
    panels["n"].values.append(Text(value="stays", display_remain_sec=5.0))
    changed = age_panels(panels, 0.25)
    assert changed == [panels["n"]]
    assert [t.value for t in panels["n"].values] == ["stays"]


def test_age_panels_never_goes_negative():
    panels = make_panels()
    panels["n"].values.append(Text(value="gone", display_remain_sec=0.1))
    age_panels(panels, 99.0)
    assert list(panels["n"].values) == []


def test_age_panels_returns_nothing_when_no_head_expires():
    panels = make_panels()
    panels["n"].values.append(Text(value="stays", display_remain_sec=5.0))
    assert age_panels(panels, 0.5) == []


def test_age_panels_ignores_empty_panels():
    panels = make_panels()
    assert age_panels(panels, 1.0) == []


def test_age_panels_handles_each_panel_independently():
    panels = make_panels()
    panels["n"].values.append(Text(value="n-gone", display_remain_sec=0.1))
    panels["s"].values.append(Text(value="s-stays", display_remain_sec=9.0))
    changed = age_panels(panels, 0.1)
    assert changed == [panels["n"]]
    assert [t.value for t in panels["s"].values] == ["s-stays"]


def test_next_expiry_sec_is_none_when_nothing_is_displayed():
    assert next_expiry_sec(make_panels()) is None


def test_next_expiry_sec_is_the_soonest_head_across_panels():
    panels = make_panels()
    panels["n"].values.append(Text(value="n", display_remain_sec=3.0))
    panels["s"].values.append(Text(value="s", display_remain_sec=1.5))
    assert next_expiry_sec(panels) == 1.5


def test_next_expiry_sec_ignores_non_head_entries():
    panels = make_panels()
    panels["n"].values.append(Text(value="head", display_remain_sec=4.0))
    panels["n"].values.append(Text(value="behind", display_remain_sec=0.1))
    assert next_expiry_sec(panels) == 4.0
```

- [ ] **Step 2: テストが失敗することを確認する**

Run: `uv run pytest tests/test_subtitle_state.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vspeech.lib.subtitle_state'`

- [ ] **Step 3: `vspeech/lib/subtitle_state.py` を作る**

`vspeech/worker/subtitle.py` の 27-39 行（`update_display_sec`）、141-188 行（`Text` / `Texts`）、210-241 行（`how_many_should_we_pop` / `ingest_text`）を**逐語で**移動し、`age_panels` / `next_expiry_sec` を足す。

```python
"""subtitle の純粋な状態機械 (ADR-0040)。

TK / OBS 両バックエンドがこれを共有するので、履歴・トリム・区切り文字・
表示時間の意味はバックエンドによらず同一になる。tkinter に依存しない。
"""

from collections import deque
from dataclasses import dataclass
from dataclasses import field

from vspeech.config import Anchor
from vspeech.config import SubtitleTextConfig
from vspeech.shared_context import WorkerInput


def update_display_sec(
    current_sec: float, current_text: str, add_text: str, config: SubtitleTextConfig
) -> float:
    min_sec = config.min_display_sec
    sec_per_letter = config.display_sec_per_letter
    max_text_len = config.max_text_len
    add_sec = max(
        len(add_text) * sec_per_letter,
        min_sec,
    )
    if len(current_text) + len(add_text) > max_text_len:
        return add_sec
    return current_sec + add_sec


@dataclass
class Text:
    value: str = ""
    display_remain_sec: float = 0


@dataclass
class Texts:
    tag: str
    anchor: Anchor
    config: SubtitleTextConfig
    bb_width: int
    bb_height: int
    display_remain_sec: float = 0
    values: deque[Text] = field(init=False)

    def __post_init__(self):
        self.values = deque([], maxlen=self.config.max_histories)

    @property
    def coord_x(self):
        if self.anchor == "center":
            return self.bb_width // 2
        elif "e" in self.anchor:
            return self.bb_width - self.config.margin
        elif "w" in self.anchor:
            return self.config.margin
        else:
            return self.bb_width // 2

    @property
    def coord_y(self):
        if self.anchor == "center":
            return self.bb_height // 2
        elif "s" in self.anchor:
            return self.bb_height - self.config.margin
        elif "n" in self.anchor:
            return self.config.margin
        else:
            return self.bb_height // 2

    @property
    def texts(self):
        if "s" in self.anchor:
            return self.config.delimiter.join(t.value for t in reversed(self.values))
        else:
            return self.config.delimiter.join(t.value for t in self.values)


def how_many_should_we_pop(texts: deque[Text], max_length: int):
    total_length = 0
    for idx, text in enumerate(reversed(texts)):
        total_length += len(text.value)
        if total_length > max_length:
            return len(texts) - (idx + 1)
    return 0


def ingest_text(texts: dict[str, Texts], message: WorkerInput) -> Texts:
    """Route an inbound message to its panel, append it, and trim overflow.

    Picks the panel named by the message's `position` param, falling back to the
    "n" panel when it is unset/unknown. Appends the text as a new entry whose
    display duration comes from `update_display_sec`, then drops the oldest
    entries that overflow `max_text_len`. Returns the panel that needs redrawing.
    """
    position = message.current_event.params.position
    ts = texts[position] if position in texts else texts["n"]
    t = Text()
    t.display_remain_sec = update_display_sec(
        current_sec=t.display_remain_sec,
        current_text=t.value,
        add_text=message.text,
        config=ts.config,
    )
    t.value = message.text
    ts.values.append(t)
    n_pop = how_many_should_we_pop(ts.values, max_length=ts.config.max_text_len)
    for _ in range(n_pop):
        ts.values.popleft()
    return ts


def age_panels(panels: dict[str, Texts], elapsed_sec: float) -> list[Texts]:
    """Age each panel's head entry by `elapsed_sec` and drop it once it expires.

    Mirrors the TK loop's aging: only `values[0]` counts down, so entries queue
    up behind the one on screen. TK drives this off a fixed 1/30s frame count;
    callers here pass real elapsed time. Returns the panels whose contents
    changed and therefore need re-rendering.
    """
    changed: list[Texts] = []
    for ts in panels.values():
        if not ts.values:
            continue
        t = ts.values[0]
        t.display_remain_sec = max(t.display_remain_sec - elapsed_sec, 0)
        if t.display_remain_sec <= 0:
            ts.values.popleft()
            changed.append(ts)
    return changed


def next_expiry_sec(panels: dict[str, Texts]) -> float | None:
    """Seconds until the soonest head entry expires, or None if nothing is shown."""
    remains = [ts.values[0].display_remain_sec for ts in panels.values() if ts.values]
    return min(remains) if remains else None
```

- [ ] **Step 4: `worker/subtitle.py` から移した定義を消し、import に置き換える**

`vspeech/worker/subtitle.py` の先頭 import 群を修正し、移動した 5 つの定義（`update_display_sec` / `Text` / `Texts` / `how_many_should_we_pop` / `ingest_text`）を削除して、代わりに:

```python
from vspeech.lib.subtitle_state import Text
from vspeech.lib.subtitle_state import Texts
from vspeech.lib.subtitle_state import ingest_text
```

`from collections import deque` / `from dataclasses import dataclass` / `from dataclasses import field` / `from vspeech.config import Anchor` は、この時点で `subtitle.py` から未使用になるものがあれば ruff が指摘する。指摘に従って消す。

- [ ] **Step 5: 既存テストの import 先を変える**

`tests/test_subtitle_ingest.py:11-13` を:

```python
from vspeech.lib.subtitle_state import Text
from vspeech.lib.subtitle_state import Texts
from vspeech.lib.subtitle_state import ingest_text
```

- [ ] **Step 6: テストが通ることを確認する**

Run: `uv run pytest tests/test_subtitle_state.py tests/test_subtitle_ingest.py tests/test_subtitle_wrap.py tests/test_subtitle_redraw.py -v`
Expected: 全て PASS（新規 9 + 既存 5 + 6 + 3 = 23 passed）

- [ ] **Step 7: 全体テストで回帰が無いことを確認する**

Run: `uv run pytest`
Expected: 既存 275 passed + 新規分。失敗ゼロ

- [ ] **Step 8: コミット**

```bash
git add vspeech/lib/subtitle_state.py vspeech/worker/subtitle.py tests/test_subtitle_state.py tests/test_subtitle_ingest.py
git commit -m "refactor(subtitle): extract the pure state machine to lib/subtitle_state"
```

---

### Task 4: tk バックエンドを分離し、ディスパッチャを作る

ここで **`worker/subtitle.py` が tkinter を import しなくなる**。spec の受入基準「tkinter を一切必要とせずに起動する」の土台。

**Files:**
- Create: `vspeech/worker/subtitle_tk.py`
- Modify: `vspeech/worker/subtitle.py`（ディスパッチャに作り替え）
- Modify: `vspeech/config.py`（`SubtitleWorkerType` / `SubtitleConfig.worker_type`）
- Modify: `tests/test_subtitle_wrap.py:1-6`, `tests/test_subtitle_redraw.py`（import 先）
- Test: `tests/test_subtitle_dispatch.py`

**Interfaces:**
- Consumes: `vspeech.lib.subtitle_state` の全て（Task 3）
- Produces:
  - `SubtitleWorkerType`（Enum: `TK = "TK"`, `OBS = "OBS"`）
  - `SubtitleConfig.worker_type: SubtitleWorkerType = SubtitleWorkerType.TK`
  - `vspeech.worker.subtitle_tk.subtitle_tk_worker(context: SharedContext, in_queue: Queue[WorkerInput]) -> None`
  - `vspeech.worker.subtitle.create_subtitle_task(tg: TaskGroup, context: SharedContext) -> Task`

- [ ] **Step 1: config に `worker_type` を足す**

`vspeech/config.py` の `TtsWorkerType`（92-94 行）の直後に:

```python
class SubtitleWorkerType(Enum):
    TK = "TK"
    OBS = "OBS"
```

`SubtitleConfig`（219 行〜）に 1 行足す（既定は `TK`＝既存 config の挙動を変えない）:

```python
class SubtitleConfig(BaseModel):
    enable: bool = False
    worker_type: SubtitleWorkerType = SubtitleWorkerType.TK
    window_width: int = 1600
    window_height: int = 120
    bg_color: str = "#00ff00"
    text: SubtitleTextConfig = Field(
        default_factory=lambda: SubtitleTextConfig(anchor="s")
    )
    translated: SubtitleTextConfig = Field(
        default_factory=lambda: SubtitleTextConfig(anchor="n")
    )
```

- [ ] **Step 2: 失敗するテストを書く**

`tests/test_subtitle_dispatch.py`:

```python
import sys

from vspeech.config import Config
from vspeech.config import SubtitleWorkerType


def test_subtitle_worker_type_defaults_to_tk():
    # 既存 config (worker_type 未指定) の挙動を変えない。
    assert Config().subtitle.worker_type == SubtitleWorkerType.TK


def test_subtitle_worker_type_round_trips_through_toml():
    config = Config()
    config.subtitle.worker_type = SubtitleWorkerType.OBS
    reloaded = Config.model_validate({"subtitle": {"worker_type": "OBS"}})
    assert reloaded.subtitle.worker_type == SubtitleWorkerType.OBS


def test_importing_the_subtitle_dispatcher_does_not_import_tkinter():
    """ヘッドレス目的の要。ディスパッチャ経由で tkinter が引き込まれないこと (ADR-0040)。

    tkinter は stdlib なので他経路で既に入っていることがある。ここでは
    「subtitle が tkinter に依存していないこと」ではなく「subtitle を import
    しても tkinter が新たに読み込まれないこと」を見たいので、一度落として
    から確かめる。
    """
    for name in list(sys.modules):
        if name == "tkinter" or name.startswith("tkinter."):
            del sys.modules[name]
    for name in list(sys.modules):
        if name.startswith("vspeech.worker.subtitle"):
            del sys.modules[name]

    import vspeech.worker.subtitle  # noqa: F401

    assert "tkinter" not in sys.modules
```

- [ ] **Step 3: テストが失敗することを確認する**

Run: `uv run pytest tests/test_subtitle_dispatch.py -v`
Expected: `test_importing_the_subtitle_dispatcher_does_not_import_tkinter` が FAIL（`subtitle.py` が今はまだ tkinter を import している）

- [ ] **Step 4: `vspeech/worker/subtitle_tk.py` を作る**

`vspeech/worker/subtitle.py` の現行内容から、tk 固有の部分を**逐語移動**する: `wrap_text_to_width` / `draw_text_with_outline` / `TRANSPARENT_BG_COLOR` / `WIN32_TRANSPARENT_COLOR` / `set_bg_color` / `redraw_panel` / `subtitle_worker` の本体 / `on_closing`。

**変更点は 2 つだけ**（それ以外はロジックに触らない）:
1. 関数名 `subtitle_worker` → `subtitle_tk_worker`
2. `Tk()` の生成が `create_subtitle_task` から関数内へ移る。`tk_root` 引数を取るのをやめ、自分で作る。`WM_DELETE_WINDOW` の配線もここへ移す（現行の「窓を閉じるとパイプラインが止まる」挙動を保つ）

```python
"""subtitle の TK バックエンド (ADR-0040)。

ADR-0040 の非ゴールにより、ここのロジックは OBS バックエンド追加の前後で
変えない。tkinter への依存はこのファイルだけに閉じ込める。
"""

from asyncio import CancelledError
from asyncio import Queue
from asyncio import QueueEmpty
from asyncio import current_task
from asyncio import sleep
from collections.abc import Callable
from functools import partial
from sys import platform
from tkinter import Canvas
from tkinter import Tk
from tkinter.font import Font
from typing import Any

from vspeech.config import Anchor
from vspeech.config import SubtitleTextConfig
from vspeech.exceptions import shutdown_worker
from vspeech.lib.subtitle_state import Texts
from vspeech.lib.subtitle_state import ingest_text
from vspeech.logger import logger
from vspeech.shared_context import SharedContext
from vspeech.shared_context import WorkerInput


def wrap_text_to_width(text: str, measure: Callable[[str], int], max_width: int) -> str:
    """Hard-wrap `text` so no line measures wider than `max_width`.

    `measure` maps a string to its rendered pixel width (e.g. `Font.measure`);
    injecting it keeps this function pure and Tk-free. `max_width <= 0` disables
    wrapping and returns `text` unchanged. Existing newlines are preserved and
    each source line is wrapped independently; a single character that alone
    exceeds `max_width` is still kept (never dropped) on its own line.

    Costs one `measure` call per character once a line does not fit, and each
    call is a Tcl round-trip: measured at ~0.44ms, so a wrapping redraw blocks
    the event loop for 26-87ms (1920px / Meiryo UI 24 / 100-200 chars), against
    0.53ms when the line fits and the early return above takes over. Only
    `max_text_len`, `font_size` and the window width decide which side of that
    cliff you land on. A binary search over the prefix length would give the
    same result in ~7 calls: `measure` is monotonic in prefix length, so the
    longest prefix that fits is found exactly. Deferred as out of scope for the
    OBS backend branch, which does not use this path at all (the OBS backend
    lets `extents_wrap` break lines inside OBS).
    """
    if max_width <= 0:
        return text
    wrapped_lines: list[str] = []
    for line in text.split("\n"):
        if not line:
            wrapped_lines.append("")
            continue
        if measure(line) <= max_width:
            wrapped_lines.append(line)
            continue
        current_line = ""
        for char in line:
            if not current_line:
                current_line = char
                continue
            test_line = current_line + char
            if measure(test_line) <= max_width:
                current_line = test_line
            else:
                wrapped_lines.append(current_line)
                current_line = char
        if current_line:
            wrapped_lines.append(current_line)
    return "\n".join(wrapped_lines)


def draw_text_with_outline(
    canvas: Canvas,
    text_coord_x: float,
    text_coord_y: float,
    texts: str,
    text_tag: str,
    anchor: Anchor,
    config: SubtitleTextConfig,
    max_width: int = 0,
):
    text_color = config.font_color
    outline_color = config.outline_color

    font_tuple = Font(
        family=config.font_family,
        size=config.font_size,
        weight="bold" if config.font_style.lower() == "bold" else "normal",
    )

    texts = wrap_text_to_width(texts, font_tuple.measure, max_width)

    justify_val = "center"
    if "e" in anchor:
        justify_val = "right"
    elif "w" in anchor:
        justify_val = "left"

    offset = 1
    for i in range(0, 4):
        x = text_coord_x - offset if i % 2 == 0 else text_coord_x + offset
        y = text_coord_y + offset if i < 2 else text_coord_y - offset
        canvas.create_text(
            x,
            y,
            text=texts,
            font=font_tuple,
            fill=outline_color,
            anchor=anchor,
            justify=justify_val,
            tags=text_tag,
        )
    canvas.create_text(
        text_coord_x,
        text_coord_y,
        text=texts,
        font=font_tuple,
        fill=text_color,
        anchor=anchor,
        justify=justify_val,
        tags=text_tag,
    )


TRANSPARENT_BG_COLOR = "systemTransparent"
WIN32_TRANSPARENT_COLOR = "#000001"


def set_bg_color(canvas: Canvas, bg_color: str):
    if bg_color == TRANSPARENT_BG_COLOR and platform == "win32":
        canvas.configure(bg=WIN32_TRANSPARENT_COLOR)
    else:
        canvas.configure(bg=bg_color)


def redraw_panel(canvas: Canvas, ts: Texts):
    """Clear the panel's previous text for its tag and draw its current state.

    Deletes before drawing so successive frames don't stack, and packs so the
    canvas re-lays out. `max_width` reserves a `margin`-wide gutter each side.
    """
    canvas.delete(ts.tag)
    draw_text_with_outline(
        canvas=canvas,
        texts=ts.texts,
        text_coord_x=ts.coord_x,
        text_coord_y=ts.coord_y,
        text_tag=ts.tag,
        anchor=ts.anchor,
        config=ts.config,
        max_width=ts.bb_width - ts.config.margin * 2,
    )
    canvas.pack()


def on_closing(gui_task: Any):
    if not gui_task.cancelled() and not gui_task.done():
        gui_task.cancel()


async def subtitle_tk_worker(
    context: SharedContext,
    in_queue: Queue[WorkerInput],
):
    tk_root = Tk()
    address = f"{context.config.listen_address}:{context.config.listen_port}"
    tk_root.title(f"vspeech:subtitle {address}")
    tk_root.protocol("WM_DELETE_WINDOW", partial(on_closing, current_task()))
    try:
        initial_width = context.config.subtitle.window_width
        initial_height = context.config.subtitle.window_height
        tk_root.geometry(f"{initial_width}x{initial_height}")
        tk_root.configure(borderwidth=0, highlightthickness=0)
        canvas = Canvas(
            tk_root,
            width=initial_width,
            height=initial_height,
            highlightthickness=0,
        )
        if (
            context.config.subtitle.bg_color == TRANSPARENT_BG_COLOR
            and platform == "win32"
        ):
            tk_root.wm_attributes("-transparentcolor", WIN32_TRANSPARENT_COLOR)
            tk_root.configure(bg=WIN32_TRANSPARENT_COLOR)
        else:
            tk_root.configure(bg=context.config.subtitle.bg_color)
        set_bg_color(canvas, bg_color=context.config.subtitle.bg_color)
        texts = {
            "n": Texts(
                tag="text",
                anchor=context.config.subtitle.text.anchor,
                config=context.config.subtitle.text,
                bb_height=tk_root.winfo_height(),
                bb_width=tk_root.winfo_width(),
            ),
            "s": Texts(
                tag="translated",
                anchor=context.config.subtitle.translated.anchor,
                config=context.config.subtitle.translated,
                bb_height=tk_root.winfo_height(),
                bb_width=tk_root.winfo_width(),
            ),
        }
        interval_sec = 1.0 / 30.0
        while True:
            set_bg_color(canvas, bg_color=context.config.subtitle.bg_color)
            for p in texts:
                if p == "n":
                    texts[p].config = context.config.subtitle.text
                elif p == "s":
                    texts[p].config = context.config.subtitle.translated
                texts[p].bb_width = tk_root.winfo_width()
                texts[p].bb_height = tk_root.winfo_height()
                if not texts[p].values:
                    continue
                t = texts[p].values[0]
                t.display_remain_sec = max(t.display_remain_sec - interval_sec, 0)
                if t.display_remain_sec <= 0:
                    texts[p].values.popleft()
                    redraw_panel(canvas, texts[p])
            tk_root.update()
            await sleep(interval_sec)
            try:
                message = in_queue.get_nowait()
                redraw_panel(canvas, ingest_text(texts, message))
            except QueueEmpty:
                pass
    except Exception as e:
        logger.exception(e)
        raise e
    except CancelledError as e:
        logger.info("subtitle worker cancelled")
        tk_root.destroy()
        raise shutdown_worker(e)
```

- [ ] **Step 5: `vspeech/worker/subtitle.py` をディスパッチャに作り替える**

ファイル全体を以下で置き換える。**tkinter を import しないこと**が要。

```python
"""subtitle worker のディスパッチャ (ADR-0040)。

transcription / tts と同じく `worker_type` でバックエンドへ振る。tkinter は
TK バックエンドの中だけに閉じ込めるので、このモジュールは import しない
(ヘッドレス構成で tkinter を要求しないため)。
"""

from asyncio import CancelledError
from asyncio import Queue
from asyncio import TaskGroup
from typing import assert_never

from vspeech.config import SubtitleWorkerType
from vspeech.exceptions import shutdown_worker
from vspeech.shared_context import EventType
from vspeech.shared_context import SharedContext
from vspeech.shared_context import WorkerInput


async def subtitle_worker(
    context: SharedContext,
    in_queue: Queue[WorkerInput],
):
    worker_type = context.config.subtitle.worker_type
    if worker_type == SubtitleWorkerType.TK:
        from vspeech.worker.subtitle_tk import subtitle_tk_worker

        await subtitle_tk_worker(context, in_queue=in_queue)
    elif worker_type == SubtitleWorkerType.OBS:
        from vspeech.worker.subtitle_obs import subtitle_obs_worker

        await subtitle_obs_worker(context, in_queue=in_queue)
    else:
        assert_never(worker_type)


def create_subtitle_task(
    tg: TaskGroup,
    context: SharedContext,
):
    worker = context.add_worker(
        event=EventType.subtitle, configs_depends_on=["subtitle"]
    )
    return tg.create_task(
        subtitle_worker(context, in_queue=worker.in_queue),
        name=worker.event.name,
    )
```

`CancelledError` / `shutdown_worker` はこのモジュールでは使わない（各バックエンドが自分で扱う）ので、ruff の未使用 import 指摘に従って消す。

- [ ] **Step 6: 既存テストの import 先を変える**

`tests/test_subtitle_wrap.py` の `from vspeech.worker.subtitle import wrap_text_to_width` を `from vspeech.worker.subtitle_tk import wrap_text_to_width` に。

`tests/test_subtitle_redraw.py` の `redraw_panel` / `draw_text_with_outline` の import と monkeypatch 対象を `vspeech.worker.subtitle_tk` に変える。**monkeypatch のパス文字列（`"vspeech.worker.subtitle.draw_text_with_outline"` 等）も忘れず変える** — ここを見落とすとテストは通るが実装を差し替えられていない。`Texts` / `Text` の import は `vspeech.lib.subtitle_state` へ。

- [ ] **Step 7: テストが通ることを確認する**

Run: `uv run pytest tests/test_subtitle_dispatch.py tests/test_subtitle_wrap.py tests/test_subtitle_redraw.py tests/test_subtitle_ingest.py tests/test_subtitle_state.py -v`
Expected: 全て PASS。特に `test_importing_the_subtitle_dispatcher_does_not_import_tkinter` が PASS

（この時点で `subtitle_obs` はまだ存在しないが、`subtitle.py` の import は関数内なのでモジュール import は成功する。）

- [ ] **Step 8: 全体テスト**

Run: `uv run pytest`
Expected: 失敗ゼロ

- [ ] **Step 9: TK バックエンドが実際に動くことを確認する（手動）**

既存の tk 構成が壊れていないことを実際に確認する。**テストは Tk を起動しないので、ここは目視でしか分からない。**

Run: `uv run python -m vspeech --config <subtitle が enable な config>`
Expected: これまでどおり字幕ウィンドウが出る。ウィンドウを閉じるとパイプラインが止まる

- [ ] **Step 10: コミット**

```bash
git add vspeech/config.py vspeech/worker/subtitle.py vspeech/worker/subtitle_tk.py tests/test_subtitle_dispatch.py tests/test_subtitle_wrap.py tests/test_subtitle_redraw.py
git commit -m "refactor(subtitle): split the TK backend out behind a worker_type dispatcher (ADR-0040)"
```

---

### Task 5: OBS の config と preflight

**Files:**
- Modify: `vspeech/config.py`（`SubtitleObsConfig`、`SubtitleConfig.obs`、`export_to_toml`）
- Modify: `vspeech/preflight.py`（`_check_subtitle` + `_CHECKERS` 登録）
- Test: `tests/test_preflight.py`（既存に追記）、`tests/test_subtitle_dispatch.py`（追記）

**Interfaces:**
- Consumes: `SubtitleWorkerType`（Task 4）
- Produces:
  - `SubtitleObsConfig`（`url: str = "ws://127.0.0.1:4455"`, `password: SecretStr = SecretStr("")`, `text_source: str = ""`, `translated_source: str = ""`）
  - `SubtitleConfig.obs: SubtitleObsConfig`
  - `preflight._check_subtitle(config: Config) -> list[ConfigProblem]`

- [ ] **Step 1: 既存の preflight テストの形を確認する**

Run: `uv run pytest tests/test_preflight.py -v --collect-only`

既存のテスト名と fixture の作法を読み、それに合わせる（この計画では既存ファイルの中身を仮定しない）。

- [ ] **Step 2: 失敗するテストを書く**

`tests/test_preflight.py` に追記:

```python
def test_subtitle_tk_backend_is_not_checked():
    # TK 構成に新しい失敗を持ち込まない (ADR-0042)。
    config = Config()
    config.subtitle.enable = True
    config.subtitle.worker_type = SubtitleWorkerType.TK
    config.subtitle.obs.url = ""
    assert _check_subtitle(config) == []


def test_disabled_subtitle_is_not_checked():
    config = Config()
    config.subtitle.enable = False
    config.subtitle.worker_type = SubtitleWorkerType.OBS
    config.subtitle.obs.url = ""
    assert _check_subtitle(config) == []


def test_obs_backend_requires_a_url():
    config = Config()
    config.subtitle.enable = True
    config.subtitle.worker_type = SubtitleWorkerType.OBS
    config.subtitle.obs.url = ""
    config.subtitle.obs.text_source = "t"
    config.subtitle.obs.translated_source = "s"
    problems = _check_subtitle(config)
    assert len(problems) == 1
    assert "url" in problems[0].detail


def test_obs_backend_rejects_a_non_websocket_url():
    config = Config()
    config.subtitle.enable = True
    config.subtitle.worker_type = SubtitleWorkerType.OBS
    config.subtitle.obs.url = "http://127.0.0.1:4455"
    config.subtitle.obs.text_source = "t"
    config.subtitle.obs.translated_source = "s"
    problems = _check_subtitle(config)
    assert len(problems) == 1
    assert "ws://" in problems[0].detail


def test_obs_backend_requires_both_source_names_and_reports_both():
    # ADR-0038 は「全問題を集約」する。1 個目で打ち切らない。
    config = Config()
    config.subtitle.enable = True
    config.subtitle.worker_type = SubtitleWorkerType.OBS
    problems = _check_subtitle(config)
    details = " ".join(p.detail for p in problems)
    assert "text_source" in details
    assert "translated_source" in details


def test_obs_backend_accepts_a_complete_config():
    config = Config()
    config.subtitle.enable = True
    config.subtitle.worker_type = SubtitleWorkerType.OBS
    config.subtitle.obs.text_source = "vspeech-text"
    config.subtitle.obs.translated_source = "vspeech-translated"
    assert _check_subtitle(config) == []
```

`tests/test_subtitle_dispatch.py` に追記:

```python
def test_obs_password_survives_a_toml_round_trip():
    """export_to_toml は SecretStr をハードコードで展開している。新しい secret を
    足したらここも足さないと、GUI の保存が config を壊す。"""
    from vspeech.config import Config

    config = Config()
    config.subtitle.enable = True
    config.subtitle.worker_type = SubtitleWorkerType.OBS
    config.subtitle.obs.password = SecretStr("hunter2")
    dumped = config.export_to_toml()
    assert "hunter2" in dumped
    assert "**" not in dumped
    reloaded = Config.read_config_from_file(
        _named_bytes_io(dumped.encode("utf-8"), "config.toml")
    )
    assert reloaded.subtitle.obs.password.get_secret_value() == "hunter2"


def _named_bytes_io(data: bytes, name: str):
    import io

    buf = io.BytesIO(data)
    buf.name = name
    return buf
```

- [ ] **Step 3: テストが失敗することを確認する**

Run: `uv run pytest tests/test_preflight.py -k subtitle tests/test_subtitle_dispatch.py -v`
Expected: FAIL — `ImportError: cannot import name '_check_subtitle'` / `AttributeError: 'SubtitleConfig' object has no attribute 'obs'`

- [ ] **Step 4: config を実装する**

`vspeech/config.py` の `SubtitleConfig` の直前に:

```python
class SubtitleObsConfig(BaseModel):
    url: str = Field(
        default="ws://127.0.0.1:4455",
        description="obs-websocket server URL (Tools -> obs-websocket Settings)",
    )
    password: SecretStr = Field(
        default=SecretStr(""), description="obs-websocket server password"
    )
    text_source: str = Field(
        default="",
        description="name of the OBS Text (GDI+) source that shows transcription",
    )
    translated_source: str = Field(
        default="",
        description="name of the OBS Text (GDI+) source that shows translation",
    )

    @field_serializer("password", when_used="json")
    def serialize_password(self, v: SecretStr) -> str:
        return v.get_secret_value()
```

`SubtitleConfig` に 1 行:

```python
    obs: SubtitleObsConfig = Field(default_factory=SubtitleObsConfig)
```

`export_to_toml`（419-432 行）に subtitle を足す。**これを忘れると `toml.dumps` が SecretStr を書けずに落ちるか、マスクされた値を書き込む:**

```python
    def export_to_toml(self):
        encoded = self.model_dump()
        conf_dict = {
            **encoded,
            "ami": {**encoded["ami"], "appkey": self.ami.appkey.get_secret_value()},
            "gcp": {
                **encoded["gcp"],
                "service_account_info": {
                    k: v.get_secret_value()
                    for k, v in self.gcp.service_account_info.items()
                },
            },
            "subtitle": {
                **encoded["subtitle"],
                "obs": {
                    **encoded["subtitle"]["obs"],
                    "password": self.subtitle.obs.password.get_secret_value(),
                },
            },
        }
        return toml.dumps(conf_dict, encoder=CustomTomlEncoder(dict, separator="\n"))
```

- [ ] **Step 5: preflight を実装する**

`vspeech/preflight.py` の import に足す:

```python
from vspeech.config import SubtitleWorkerType
```

`_check_vc` の後に:

```python
def _check_subtitle(config: Config) -> list[ConfigProblem]:
    if not config.subtitle.enable:
        return []
    if config.subtitle.worker_type != SubtitleWorkerType.OBS:
        return []  # TK は接続先を持たない
    w = "subtitle"
    obs = config.subtitle.obs
    problems: list[ConfigProblem] = []
    if not obs.url:
        problems.append(
            ConfigProblem(w, "OBS バックエンドには subtitle.obs.url が必須ですが空です")
        )
    elif not obs.url.startswith(("ws://", "wss://")):
        problems.append(
            ConfigProblem(
                w, f"subtitle.obs.url '{obs.url}' は ws:// か wss:// で始まる必要があります"
            )
        )
    for name, value in (
        ("subtitle.obs.text_source", obs.text_source),
        ("subtitle.obs.translated_source", obs.translated_source),
    ):
        if not value:
            problems.append(
                ConfigProblem(w, f"OBS バックエンドには {name} が必須ですが空です")
            )
    # 認証の成立とソースの実在は層B (接続してからでないと未起動と区別できない, ADR-0042)。
    return problems
```

`_CHECKERS` に足す:

```python
_CHECKERS: list[Checker] = [
    _check_transcription,
    _check_translation,
    _check_tts,
    _check_vc,
    _check_recording,
    _check_playback,
    _check_subtitle,
]
```

- [ ] **Step 6: テストが通ることを確認する**

Run: `uv run pytest tests/test_preflight.py tests/test_subtitle_dispatch.py -v`
Expected: 全て PASS

- [ ] **Step 7: コミット**

```bash
git add vspeech/config.py vspeech/preflight.py tests/test_preflight.py tests/test_subtitle_dispatch.py
git commit -m "feat(subtitle): add OBS backend config and its preflight checks (ADR-0041, ADR-0042)"
```

---

### Task 6: config → text_gdiplus 設定のマッピング（純関数）

**ADR-0041 が「最も外れやすい」と記録した箇所。Task 1 の観測結果をここに反映する。**

> **Task 1 の実測値（OBS 32.1.2 / obs-websocket 5.7.3、2026-07-16）:**
> - `inputKind`: **`text_gdiplus_v3`**
> - `color` の並び: **BGR = `0x00BBGGRR`**（RGB だと思っていたのは誤り。`0xFF8000` を書いたら `rgb(0,128,255)` になった）
> - ソース不在時の `code`: **600**
> - `align` / `valign`: **文字列**（`"center"` / `"bottom"`）
> - `font.flags` の bold ビット: **1**
>
> 以下の実装はこの実測に合わせてある。**`#ffffff` / `#000000` はバイト順を反転しても
> 通ってしまうので、この並びを守れるのは非対称な色のテストだけ。**

**Files:**
- Create: `vspeech/lib/obs_text_settings.py`
- Test: `tests/test_obs_text_settings.py`

**Interfaces:**
- Consumes: `SubtitleTextConfig` / `SubtitleConfig` / `Anchor`（`vspeech.config`）
- Produces:
  - `hex_color_to_obs_int(hex_color: str) -> int`
  - `anchor_to_align(anchor: Anchor) -> str`
  - `anchor_to_valign(anchor: Anchor) -> str`
  - `build_text_settings(text_config: SubtitleTextConfig, subtitle_config: SubtitleConfig) -> dict[str, Any]`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_obs_text_settings.py`:

```python
import pytest

from vspeech.config import SubtitleConfig
from vspeech.config import SubtitleTextConfig
from vspeech.lib.obs_text_settings import anchor_to_align
from vspeech.lib.obs_text_settings import anchor_to_valign
from vspeech.lib.obs_text_settings import build_text_settings
from vspeech.lib.obs_text_settings import hex_color_to_obs_int


def test_hex_color_to_obs_int_reverses_rgb_to_bgr():
    # OBS は 0x00BBGGRR で保存する (実機 32.1.2 で GetSourceScreenshot により測定)。
    # #ffffff / #000000 は回文なので取り違えても素通りする -- この 1 本と次の 1 本
    # だけがバイト順を守っている (ADR-0041)。
    assert hex_color_to_obs_int("#ff8000") == 0x0080FF


def test_hex_color_to_obs_int_reversal_is_not_incidental():
    assert hex_color_to_obs_int("#0080ff") == 0xFF8000


def test_hex_color_to_obs_int_accepts_shipped_defaults():
    assert hex_color_to_obs_int("#ffffff") == 0xFFFFFF
    assert hex_color_to_obs_int("#000000") == 0x000000


def test_hex_color_to_obs_int_is_case_insensitive_and_hash_optional():
    assert hex_color_to_obs_int("FF8000") == 0xFF8000
    assert hex_color_to_obs_int("#Ff8000") == 0xFF8000


@pytest.mark.parametrize("bad", ["", "#fff", "#gggggg", "#ff80000", "nope"])
def test_hex_color_to_obs_int_rejects_junk(bad: str):
    with pytest.raises(ValueError):
        hex_color_to_obs_int(bad)


@pytest.mark.parametrize(
    ("anchor", "expected"),
    [
        ("nw", "left"),
        ("w", "left"),
        ("sw", "left"),
        ("ne", "right"),
        ("e", "right"),
        ("se", "right"),
        ("n", "center"),
        ("s", "center"),
        ("center", "center"),
    ],
)
def test_anchor_to_align_matches_the_tk_justify_rule(anchor, expected):
    # tk の draw_text_with_outline と同じ規則: e -> right, w -> left, else center
    assert anchor_to_align(anchor) == expected


@pytest.mark.parametrize(
    ("anchor", "expected"),
    [
        ("nw", "top"),
        ("n", "top"),
        ("ne", "top"),
        ("sw", "bottom"),
        ("s", "bottom"),
        ("se", "bottom"),
        ("w", "center"),
        ("e", "center"),
        ("center", "center"),
    ],
)
def test_anchor_to_valign(anchor, expected):
    assert anchor_to_valign(anchor) == expected


def test_build_text_settings_maps_every_tk_key():
    subtitle = SubtitleConfig(window_width=1920, window_height=120)
    text = SubtitleTextConfig(
        anchor="s",
        font_family="Meiryo UI",
        font_size=24,
        font_style="bold",
        font_color="#ff8000",
        outline_color="#0000ff",
        margin=4,
    )
    got = build_text_settings(text, subtitle)
    assert got["font"] == {"face": "Meiryo UI", "size": 24, "flags": 1}
    # BGR: #ff8000 -> 0x0080FF, #0000ff -> 0xFF0000 (実機で測定, ADR-0041)
    assert got["color"] == 0x0080FF
    assert got["opacity"] == 100
    assert got["outline"] is True
    assert got["outline_size"] == 1
    assert got["outline_color"] == 0xFF0000
    assert got["outline_opacity"] == 100
    assert got["align"] == "center"
    assert got["valign"] == "bottom"
    assert got["extents"] is True
    assert got["extents_cx"] == 1920 - 4 * 2
    assert got["extents_cy"] == 120 - 4 * 2
    assert got["extents_wrap"] is True


def test_build_text_settings_non_bold_clears_the_flag():
    text = SubtitleTextConfig(font_style="normal")
    assert build_text_settings(text, SubtitleConfig())["font"]["flags"] == 0


def test_build_text_settings_bold_is_case_insensitive_like_tk():
    # tk: "bold" if config.font_style.lower() == "bold" else "normal"
    text = SubtitleTextConfig(font_style="BOLD")
    assert build_text_settings(text, SubtitleConfig())["font"]["flags"] == 1


def test_build_text_settings_transparent_bg_becomes_zero_opacity():
    subtitle = SubtitleConfig(bg_color="systemTransparent")
    got = build_text_settings(SubtitleTextConfig(), subtitle)
    assert got["bk_opacity"] == 0


def test_build_text_settings_opaque_bg_is_honoured_like_tk():
    # tk で bg_color="#00ff00" なら緑の背景になる。OBS でも同じにする。
    # (緑は回文なのでバイト順は守れない -- 下の非対称ケースがそれを見る)
    subtitle = SubtitleConfig(bg_color="#00ff00")
    got = build_text_settings(SubtitleTextConfig(), subtitle)
    assert got["bk_color"] == 0x00FF00
    assert got["bk_opacity"] == 100


def test_build_text_settings_bg_colour_is_also_bgr():
    subtitle = SubtitleConfig(bg_color="#ff8000")
    got = build_text_settings(SubtitleTextConfig(), subtitle)
    assert got["bk_color"] == 0x0080FF
    assert got["bk_opacity"] == 100


def test_build_text_settings_never_sets_a_negative_extent():
    # margin が窓より大きい病的な config でも OBS に負値を送らない。
    subtitle = SubtitleConfig(window_width=4, window_height=4)
    got = build_text_settings(SubtitleTextConfig(margin=100), subtitle)
    assert got["extents_cx"] >= 1
    assert got["extents_cy"] >= 1


def test_build_text_settings_does_not_set_text():
    # テキストは別経路 (毎回変わる) で push する。スタイルと混ぜない。
    assert "text" not in build_text_settings(SubtitleTextConfig(), SubtitleConfig())
```

- [ ] **Step 2: テストが失敗することを確認する**

Run: `uv run pytest tests/test_obs_text_settings.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vspeech.lib.obs_text_settings'`

- [ ] **Step 3: 実装する**

`vspeech/lib/obs_text_settings.py`:

```python
"""config -> OBS `text_gdiplus` 設定 dict のマッピング (ADR-0041)。

config を表示スタイルの権威とするため、tk 版の設定キーをすべて OBS 側の
設定へ写す。OBS のバージョンに依存する形式 (色の int の並び、font flags の
ビット) はこのファイルだけに閉じ込める。
"""

import re
from typing import Any

from vspeech.config import Anchor
from vspeech.config import SubtitleConfig
from vspeech.config import SubtitleTextConfig

# obs-properties.h の enum obs_font_style。
OBS_FONT_BOLD = 1

# tk の TRANSPARENT_BG_COLOR と同じ番兵。tk では win32 の -transparentcolor に
# 化けるが、OBS では背景の不透明度 0 に写す (カラーキー自体が不要になる)。
TRANSPARENT_BG_COLOR = "systemTransparent"

_HEX_COLOR = re.compile(r"\A#?([0-9a-fA-F]{6})\Z")


def hex_color_to_obs_int(hex_color: str) -> int:
    """`#rrggbb` を OBS が設定に持つ int へ変換する。

    OBS は色を **0x00BBGGRR (BGR)** で保存する。実機 (OBS 32.1.2 /
    obs-websocket 5.7.3) で測定した: `0xFF8000` を書き込むと rgb(0,128,255)
    にレンダリングされ、`0x0080FF` を書き込むと rgb(255,128,0) になる。

    出荷デフォルトの #ffffff / #000000 は回文なので、この並びを取り違えても
    素通りする。並びを守れるのは非対称な色のテストだけ。
    """
    m = _HEX_COLOR.match(hex_color)
    if not m:
        raise ValueError(f"'{hex_color}' は #rrggbb 形式の色ではありません")
    digits = m.group(1)
    r, g, b = (int(digits[i : i + 2], 16) for i in (0, 2, 4))
    return (b << 16) | (g << 8) | r


def anchor_to_align(anchor: Anchor) -> str:
    """tk の `draw_text_with_outline` の justify 規則をそのまま写す。"""
    if "e" in anchor:
        return "right"
    if "w" in anchor:
        return "left"
    return "center"


def anchor_to_valign(anchor: Anchor) -> str:
    """tk の `coord_y` が anchor の n/s 成分で決めているのと同じ区分。"""
    if "n" in anchor:
        return "top"
    if "s" in anchor:
        return "bottom"
    return "center"


def build_text_settings(
    text_config: SubtitleTextConfig, subtitle_config: SubtitleConfig
) -> dict[str, Any]:
    """1 パネル分のスタイル設定を組む。`text` は含めない (別経路で push する)。"""
    bg = subtitle_config.bg_color
    transparent = bg == TRANSPARENT_BG_COLOR
    margin = text_config.margin
    return {
        "font": {
            "face": text_config.font_family,
            "size": text_config.font_size,
            "flags": OBS_FONT_BOLD if text_config.font_style.lower() == "bold" else 0,
        },
        "color": hex_color_to_obs_int(text_config.font_color),
        "opacity": 100,
        "outline": True,
        # tk は 1px オフセットの 4 隅コピーで輪郭を描く。それに相当する太さ。
        "outline_size": 1,
        "outline_color": hex_color_to_obs_int(text_config.outline_color),
        "outline_opacity": 100,
        "align": anchor_to_align(text_config.anchor),
        "valign": anchor_to_valign(text_config.anchor),
        "bk_color": 0x000000 if transparent else hex_color_to_obs_int(bg),
        "bk_opacity": 0 if transparent else 100,
        "extents": True,
        "extents_cx": max(subtitle_config.window_width - margin * 2, 1),
        "extents_cy": max(subtitle_config.window_height - margin * 2, 1),
        "extents_wrap": True,
    }
```

- [ ] **Step 4: テストが通ることを確認する**

Run: `uv run pytest tests/test_obs_text_settings.py -v`
Expected: 全て PASS

- [ ] **Step 5: コミット**

```bash
git add vspeech/lib/obs_text_settings.py tests/test_obs_text_settings.py
git commit -m "feat(subtitle): map subtitle config onto text_gdiplus settings (ADR-0041)"
```

---

### Task 7: OBS バックエンド

**Files:**
- Create: `vspeech/worker/subtitle_obs.py`
- Test: `tests/test_subtitle_obs.py`

**Interfaces:**
- Consumes: `ObsWsClient` / `ObsIdentifyError` / `ObsResourceNotFoundError`（Task 2）、`subtitle_state` の全て（Task 3）、`build_text_settings`（Task 6）
- Produces: `subtitle_obs_worker(context: SharedContext, in_queue: Queue[WorkerInput]) -> None`

**設計の要（Task 2 のレビューを受けて改訂）:**
- `connect()` は fail-open。接続拒否・切断は外側の `except` が拾って再接続する（ADR-0042）。
- **`worker_startup` をここで使ってはいけない。** ADR-0038 の層B は通常 `worker_startup` を使うが、あれは `except Exception` で**すべて**を `WorkerStartupError` に変える。この worker では identify 中のタイムアウト（＝リトライすれば直る）まで fail-loud に化けてしまい、「観測できたものだけ即死」（ADR-0042）に反する。代わりに、**観測済みかつ回復不能な 2 種類だけ**を型で拾って `WorkerStartupError` を直接送出する:

```python
    try:
        await client.identify(obs.password.get_secret_value())
        await validate_sources(client, obs)
    except (ObsIdentifyError, ObsResourceNotFoundError) as e:
        # 認証失敗とソース不在は、繋がった上で観測できて、かつリトライしても
        # 直らない。ここだけが fail-loud (ADR-0042)。
        raise WorkerStartupError("subtitle", str(e)) from e
    # 他の ObsProtocolError (タイムアウト・不正メッセージ) と OSError /
    # WebSocketException は下の except が拾って再接続する = fail-open。
```

- **例外は `logger` 経由でのみ出すこと（`print` や直接のストリーム書き込みは不可）。** `lib/obs_ws.py` の監査で判明: OBS が `comment` に孤立サロゲートを入れて送ると `str(e)` にそれが残る（`comment` の可読性のためスライスで縛っており、`repr` のようにエスケープしないため）。契約も長さ上限も破らないが、utf-8 ストリームへ直接書くと `UnicodeEncodeError` になる。`logging` のハンドラはこれを握り潰すので `logger.warning(...)` なら安全（実測確認済み）。
- セッション中の切断・タイムアウト・不正メッセージは外側の `except (OSError, WebSocketException, ObsProtocolError)` で拾って fail-open。**`ObsProtocolError` を必ず含めること**: `lib/obs_ws.py` は recv のタイムアウトと壊れたメッセージをこの型に包む。包まれていないと素の `TimeoutError` / `KeyError` が worker を貫通し、TaskGroup ごとプロセスが死ぬ（＝字幕の都合で音声が止まる。spec の受入基準「OBS を再起動しても音声パイプラインは動き続ける」を破る）。
- `WorkerStartupError` は `OSError` でも `WebSocketException` でも `ObsProtocolError` でもないので、fail-open の `except` を素通りして上まで飛ぶ。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_subtitle_obs.py`:

```python
from asyncio import Queue
from uuid import uuid4

import pytest

from vspeech.config import Config
from vspeech.config import EventType
from vspeech.config import SubtitleWorkerType
from vspeech.lib.obs_ws import ObsResourceNotFoundError
from vspeech.lib.subtitle_state import Texts
from vspeech.shared_context import EventAddress
from vspeech.shared_context import Params
from vspeech.shared_context import SharedContext
from vspeech.shared_context import SoundInput
from vspeech.shared_context import WorkerInput
from vspeech.worker.subtitle_obs import make_panels
from vspeech.worker.subtitle_obs import push_styles
from vspeech.worker.subtitle_obs import push_text
from vspeech.worker.subtitle_obs import validate_sources


class FakeObsClient:
    """ObsWsClient の狭い口だけを真似る。ネットワークも OBS も無し。"""

    def __init__(self, missing: set[str] | None = None):
        self.calls: list[tuple[str, dict]] = []
        self.missing = missing or set()

    async def request(self, request_type: str, request_data=None) -> dict:
        data = request_data or {}
        self.calls.append((request_type, data))
        name = data.get("inputName")
        if name in self.missing:
            raise ObsResourceNotFoundError(request_type, 600, "not found")
        if request_type == "GetInputSettings":
            return {"inputKind": "text_gdiplus_v3", "inputSettings": {}}
        return {}

    def settings_for(self, source: str) -> list[dict]:
        return [
            d["inputSettings"]
            for t, d in self.calls
            if t == "SetInputSettings" and d.get("inputName") == source
        ]


def make_config() -> Config:
    config = Config()
    config.subtitle.enable = True
    config.subtitle.worker_type = SubtitleWorkerType.OBS
    config.subtitle.obs.text_source = "vspeech-text"
    config.subtitle.obs.translated_source = "vspeech-translated"
    return config


def make_message(text: str, position=None) -> WorkerInput:
    return WorkerInput(
        input_id=uuid4(),
        current_event=EventAddress(EventType.subtitle, params=Params(position=position)),
        following_events=[],
        text=text,
        sound=SoundInput.invalid(),
        file_path="",
        filters=[],
    )


def test_make_panels_uses_the_same_two_panels_as_tk():
    panels = make_panels(make_config().subtitle)
    assert set(panels) == {"n", "s"}
    assert panels["n"].anchor == "s"
    assert panels["s"].anchor == "n"


async def test_validate_sources_passes_when_both_exist():
    client = FakeObsClient()
    await validate_sources(client, make_config().subtitle.obs)
    assert [d["inputName"] for _, d in client.calls] == [
        "vspeech-text",
        "vspeech-translated",
    ]


async def test_validate_sources_raises_when_a_source_is_missing():
    client = FakeObsClient(missing={"vspeech-translated"})
    with pytest.raises(ObsResourceNotFoundError):
        await validate_sources(client, make_config().subtitle.obs)


async def test_push_text_sends_the_joined_panel_text_to_its_source():
    config = make_config()
    panels = make_panels(config.subtitle)
    from vspeech.lib.subtitle_state import ingest_text

    ts = ingest_text(panels, make_message("こんにちは"))
    client = FakeObsClient()
    await push_text(client, config.subtitle.obs, panels, ts)
    assert client.settings_for("vspeech-text") == [{"text": "こんにちは"}]


async def test_push_text_routes_the_s_panel_to_the_translated_source():
    config = make_config()
    panels = make_panels(config.subtitle)
    from vspeech.lib.subtitle_state import ingest_text

    ts = ingest_text(panels, make_message("hello", position="s"))
    client = FakeObsClient()
    await push_text(client, config.subtitle.obs, panels, ts)
    assert client.settings_for("vspeech-translated") == [{"text": "hello"}]


async def test_push_text_sends_empty_string_when_the_panel_drained():
    config = make_config()
    panels = make_panels(config.subtitle)
    client = FakeObsClient()
    await push_text(client, config.subtitle.obs, panels, panels["n"])
    assert client.settings_for("vspeech-text") == [{"text": ""}]


async def test_push_text_uses_overlay_so_it_does_not_clobber_style():
    config = make_config()
    panels = make_panels(config.subtitle)
    client = FakeObsClient()
    await push_text(client, config.subtitle.obs, panels, panels["n"])
    assert all(d["overlay"] is True for t, d in client.calls if t == "SetInputSettings")


async def test_push_styles_sends_both_panels_with_config_values():
    config = make_config()
    config.subtitle.text.font_color = "#ff8000"
    config.subtitle.translated.font_size = 22
    panels = make_panels(config.subtitle)
    client = FakeObsClient()
    await push_styles(client, config.subtitle, panels)
    text_settings = client.settings_for("vspeech-text")[0]
    translated_settings = client.settings_for("vspeech-translated")[0]
    assert text_settings["color"] == 0xFF8000
    assert text_settings["valign"] == "bottom"
    assert translated_settings["font"]["size"] == 22
    assert translated_settings["valign"] == "top"
```

- [ ] **Step 2: テストが失敗することを確認する**

Run: `uv run pytest tests/test_subtitle_obs.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vspeech.worker.subtitle_obs'`

- [ ] **Step 3: 実装する**

`vspeech/worker/subtitle_obs.py`:

```python
"""subtitle の OBS バックエンド (ADR-0040 / 0041 / 0042)。

obs-websocket のクライアントとして OBS の Text (GDI+) ソースへ字幕を push
する。OBS の構造 (シーン・input の存在・配置) には触らず、ユーザーが作った
input の設定値だけを更新する (ADR-0041)。

失敗の扱いは「観測できたものだけ即死」(ADR-0042):
  - 接続できない / 切断 / タイムアウト / 不正メッセージ -> fail-open
    (warn once + バックオフ再接続)。字幕は落ちるが音声は生き続ける。
  - 認証失敗 / ソース不在 -> fail-loud (WorkerStartupError)
繋がるまでは両者を区別できないので、繋がるまで待つ。

ADR-0038 の層B は通常 exceptions.worker_startup を使うが、ここでは使わない:
あれは except Exception で全てを WorkerStartupError に変えるため、identify 中の
タイムアウト (リトライで直る) まで fail-loud に化ける。回復不能と観測できた
2 型だけを拾って WorkerStartupError を送出する。
"""

from asyncio import CancelledError
from asyncio import Queue
from asyncio import TimeoutError as AsyncTimeoutError
from asyncio import sleep
from asyncio import wait_for
from time import monotonic
from typing import Any
from typing import Protocol

from websockets.asyncio.client import connect
from websockets.exceptions import WebSocketException

from vspeech.config import SubtitleConfig
from vspeech.config import SubtitleObsConfig
from vspeech.exceptions import WorkerStartupError
from vspeech.exceptions import shutdown_worker
from vspeech.lib.obs_ws import ObsIdentifyError
from vspeech.lib.obs_ws import ObsProtocolError
from vspeech.lib.obs_ws import ObsResourceNotFoundError
from vspeech.lib.obs_ws import ObsWsClient
from vspeech.lib.obs_text_settings import build_text_settings
from vspeech.lib.subtitle_state import Texts
from vspeech.lib.subtitle_state import age_panels
from vspeech.lib.subtitle_state import ingest_text
from vspeech.lib.subtitle_state import next_expiry_sec
from vspeech.logger import logger
from vspeech.shared_context import SharedContext
from vspeech.shared_context import WorkerInput

INITIAL_BACKOFF_SEC = 0.5
MAX_BACKOFF_SEC = 5.0


class ObsRequester(Protocol):
    async def request(
        self, request_type: str, request_data: dict[str, Any] | None = None
    ) -> dict[str, Any]: ...


def make_panels(config: SubtitleConfig) -> dict[str, Texts]:
    """tk バックエンドと同じ 2 パネル。

    bb_width/bb_height は tk の Canvas 実寸のための値で、OBS 側は extents で
    表現するのでレイアウトには使わない。共有 dataclass の必須項目なので
    config の窓寸法をそのまま入れておく。
    """
    return {
        "n": Texts(
            tag="text",
            anchor=config.text.anchor,
            config=config.text,
            bb_width=config.window_width,
            bb_height=config.window_height,
        ),
        "s": Texts(
            tag="translated",
            anchor=config.translated.anchor,
            config=config.translated,
            bb_width=config.window_width,
            bb_height=config.window_height,
        ),
    }


def _source_of(panel_key: str, obs: SubtitleObsConfig) -> str:
    return obs.text_source if panel_key == "n" else obs.translated_source


def _panel_key(panels: dict[str, Texts], ts: Texts) -> str:
    for key, panel in panels.items():
        if panel is ts:
            return key
    raise KeyError("panel not found")


async def validate_sources(client: ObsRequester, obs: SubtitleObsConfig) -> None:
    """両ソースが OBS に実在することを確かめる。

    存在しなければ ObsResourceNotFoundError が上がり、呼び出し側の
    worker_startup が WorkerStartupError へ変える (fail-loud, ADR-0042)。
    """
    for source in (obs.text_source, obs.translated_source):
        await client.request("GetInputSettings", {"inputName": source})


async def push_styles(
    client: ObsRequester, config: SubtitleConfig, panels: dict[str, Texts]
) -> None:
    """config のスタイルを両ソースへ流し込む (ADR-0041: config が権威)。"""
    for key, ts in panels.items():
        await client.request(
            "SetInputSettings",
            {
                "inputName": _source_of(key, config.obs),
                "inputSettings": build_text_settings(ts.config, config),
                "overlay": True,
            },
        )


async def push_text(
    client: ObsRequester,
    obs: SubtitleObsConfig,
    panels: dict[str, Texts],
    ts: Texts,
) -> None:
    """パネルの現在の文字列を対応するソースへ送る。

    `Texts.texts` が区切り文字での結合と "s" アンカーの逆順を済ませているので、
    ここは送るだけ。空なら空文字を送って消す。
    """
    await client.request(
        "SetInputSettings",
        {
            "inputName": _source_of(_panel_key(panels, ts), obs),
            "inputSettings": {"text": ts.texts},
            "overlay": True,
        },
    )


async def _push_all_text(
    client: ObsRequester, obs: SubtitleObsConfig, panels: dict[str, Texts]
) -> None:
    for ts in panels.values():
        await push_text(client, obs, panels, ts)


def _refresh_panel_configs(context: SharedContext, panels: dict[str, Texts]) -> None:
    panels["n"].config = context.config.subtitle.text
    panels["n"].anchor = context.config.subtitle.text.anchor
    panels["s"].config = context.config.subtitle.translated
    panels["s"].anchor = context.config.subtitle.translated.anchor


async def _run_session(
    context: SharedContext,
    client: ObsRequester,
    in_queue: Queue[WorkerInput],
    panels: dict[str, Texts],
) -> None:
    """繋がっている間の本ループ。30fps のビジーループは持たない。

    次に消える字幕の時刻までを timeout にして待つので、何も起きていない間は
    1 回も起きない。
    """
    last_tick = monotonic()
    while True:
        if context.need_reload:
            context.reset_need_reload()
            _refresh_panel_configs(context, panels)
            await push_styles(client, context.config.subtitle, panels)
            await _push_all_text(client, context.config.subtitle.obs, panels)
        timeout = next_expiry_sec(panels)
        message: WorkerInput | None = None
        try:
            message = await wait_for(in_queue.get(), timeout=timeout)
        except AsyncTimeoutError:
            pass
        now = monotonic()
        for ts in age_panels(panels, now - last_tick):
            await push_text(client, context.config.subtitle.obs, panels, ts)
        last_tick = now
        if message is not None:
            ts = ingest_text(panels, message)
            await push_text(client, context.config.subtitle.obs, panels, ts)
        if not context.running.is_set():
            await context.running.wait()


async def subtitle_obs_worker(
    context: SharedContext,
    in_queue: Queue[WorkerInput],
):
    panels = make_panels(context.config.subtitle)
    backoff = INITIAL_BACKOFF_SEC
    warned = False
    try:
        while True:
            obs = context.config.subtitle.obs
            try:
                async with connect(obs.url) as ws:
                    client = ObsWsClient(ws)
                    try:
                        await client.identify(obs.password.get_secret_value())
                        await validate_sources(client, obs)
                    except (ObsIdentifyError, ObsResourceNotFoundError) as e:
                        # 繋がった上で観測できて、かつリトライしても直らない。
                        # ここだけが fail-loud (ADR-0042)。他の ObsProtocolError
                        # (タイムアウト・不正メッセージ) は下の except へ落ちる。
                        raise WorkerStartupError("subtitle", str(e)) from e
                    logger.info("subtitle worker [obs] connected to %s", obs.url)
                    backoff = INITIAL_BACKOFF_SEC
                    warned = False
                    _refresh_panel_configs(context, panels)
                    await push_styles(client, context.config.subtitle, panels)
                    await _push_all_text(client, obs, panels)
                    await _run_session(context, client, in_queue, panels)
            except (OSError, WebSocketException, ObsProtocolError) as e:
                # OBS 未起動・切断・タイムアウト・不正メッセージ。字幕は落ちるが
                # 音声パイプラインは巻き込まない。ObsProtocolError を external に
                # 落とさないこと: 素の TimeoutError/KeyError が worker を貫通すると
                # TaskGroup ごとプロセスが死ぬ。
                if not warned:
                    logger.warning(
                        "subtitle worker [obs] cannot reach %s (%s); "
                        "retrying in the background. Subtitles are not shown "
                        "until OBS is up.",
                        obs.url,
                        e,
                    )
                    warned = True
                else:
                    logger.debug("subtitle worker [obs] still unreachable: %s", e)
                await sleep(backoff)
                backoff = min(backoff * 2, MAX_BACKOFF_SEC)
    except CancelledError as e:
        logger.info("subtitle worker cancelled")
        raise shutdown_worker(e)
```

- [ ] **Step 4: テストが通ることを確認する**

Run: `uv run pytest tests/test_subtitle_obs.py -v`
Expected: 全て PASS

- [ ] **Step 5: 型検査とリント**

Run: `uv run ruff format . && uv run ruff check . && uv run ty check`
Expected: 新規の指摘なし

- [ ] **Step 6: コミット**

```bash
git add vspeech/worker/subtitle_obs.py tests/test_subtitle_obs.py
git commit -m "feat(subtitle): add the OBS backend (ADR-0040, ADR-0041, ADR-0042)"
```

---

### Task 8: config.toml.example とドキュメント

**Files:**
- Modify: `config.toml.example`
- Modify: `CLAUDE.md`

- [ ] **Step 1: `config.toml.example` の `[subtitle]` に足す**

現行 78-83 行の `[subtitle]` セクションに `worker_type` を、そして新しい `[subtitle.obs]` セクションを足す。**このファイルは既存セクションにコメントが無いが、ここは新設の外部依存なので最小限のコメントを付ける**（OBS 側の準備が必要なことは config だけでは分からない）。

```toml
[subtitle]
enable = true
# "TK" = 専用ウィンドウを出す (ディスプレイが要る)。
# "OBS" = obs-websocket 経由で OBS の Text (GDI+) ソースへ送る (ディスプレイ不要)。
worker_type = "TK"
window_width = 1920
window_height = 120
bg_color = "systemTransparent"

# worker_type = "OBS" のときだけ使う。
# 事前に OBS 側で: Tools -> obs-websocket Settings で WebSocket サーバを有効にし、
# 下の 2 つの名前で Text (GDI+) ソースを作っておく (vspeech はソースを作らない)。
# 見た目は下の [subtitle.text] / [subtitle.translated] が権威で、OBS 側の
# 手動変更は接続時と reload 時に上書きされる。
[subtitle.obs]
url = "ws://127.0.0.1:4455"
password = ""
text_source = "vspeech-text"
translated_source = "vspeech-translated"
```

- [ ] **Step 2: `CLAUDE.md` の worker_type の記述を更新する**

「The `transcription` and `tts` workers dispatch to sub-implementations by `worker_type`」の文に subtitle を足す:

```
The `transcription`, `tts` and `subtitle` workers dispatch to sub-implementations by `worker_type` (ACP/GCP/WHISPER, VR2/VOICEVOX, TK/OBS) — add new backends there. The `subtitle` OBS backend talks to OBS as an obs-websocket client and pushes both text and style, so `tkinter` is only imported by the TK backend ([ADR-0040](docs/adr/0040-subtitle-obs-backend-via-worker-type.md) through [0043](docs/adr/0043-obs-websocket-client-in-house.md)).
```

- [ ] **Step 3: example が実際に読めることを確認する**

Run: `uv run python -c "from vspeech.config import Config; f=open('config.toml.example','rb'); c=Config.read_config_from_file(f); print(c.subtitle.worker_type, c.subtitle.obs.text_source)"`
Expected: `SubtitleWorkerType.TK vspeech-text`

- [ ] **Step 4: コミット**

```bash
git add config.toml.example CLAUDE.md
git commit -m "docs(subtitle): document the OBS backend in the example config and CLAUDE.md"
```

---

### Task 9: 実機検証と ADR の後始末

**テストは OBS を使わないので、ここでしか分からないことがある。** ADR-0038 の教訓（「entrypoint smoke が 275 テスト全てが見逃した logger の cp1252 バグを捕まえた」）と同じ位置づけ。

- [ ] **Step 1: ゲートを通す**

Run: `uv run poe check`
Expected: 既知の許容済み 2 件（torch CVE / vr2_config deadcode）以外の失敗ゼロ

- [ ] **Step 2: OBS を起動せずにパイプラインが上がることを確認する**

**ADR-0042 の中心的な主張の検証。** OBS を落とした状態で:

Run: `uv run python -m vspeech --config <worker_type="OBS" の config>`
Expected:
- プロセスが起動して他の worker が動く
- `subtitle worker [obs] cannot reach ws://... retrying` が **1 回だけ** 出る（毎回出ない = warn once）
- プロセスが落ちない

- [ ] **Step 3: 後から OBS を起動して字幕が出ることを確認する**

vspeech を動かしたまま OBS を起動する。
Expected: `subtitle worker [obs] connected to ws://...` が出て、字幕が OBS に表示される

- [ ] **Step 4: スタイルが config どおりか目視する**

色の並びは Task 1 で `GetSourceScreenshot` により測定済み（BGR 確定）なので、ここは目視で足りる。念のため `font_color = "#ff8000"` にするとオレンジで出るはず（水色なら Task 6 のマッピングが壊れている）。

Expected: フォント・サイズ・太字・輪郭・位置（anchor）・折り返しが config どおり。**背景が透過で、輪郭にカラーキー由来の縁が無い**（tk 版との差が出るところ）

- [ ] **Step 5: 配信中の OBS 再起動に耐えることを確認する**

vspeech を動かしたまま OBS を再起動する。
Expected: 音声パイプライン（vc / playback）は無傷。OBS 復帰後に字幕が自動で戻り、スタイルも再適用される

- [ ] **Step 6: fail-loud を確認する**

`text_source` をわざと typo にして起動する。
Expected: `[subtitle] ...` を含むログを出して exit する（無限 warn ではない）

`password` をわざと間違えて起動する。
Expected: 同上

- [ ] **Step 7: reload が効くことを確認する**

vspeech を動かしたまま config の `font_size` を変えて reload を投げる。
Expected: OBS 上の表示サイズが変わる

- [ ] **Step 8: TK バックエンドの非退行を確認する**

`worker_type = "TK"` に戻して起動する。
Expected: これまでどおり字幕ウィンドウが出て、閉じるとパイプラインが止まる

- [ ] **Step 9: ADR を突合して昇格する**

adr-writing の「final-review 前に突合する」を回す。ADR-0040 / 0041 / 0042 / 0043 を一巡し:
- 実装が決定を裏づけた → Status を `Proposed` → `Accepted`（1 行だけ）
- 実装が決定を覆した（例: `color` が BGR だった / ソース不在の code が 600 でなかった）→ 昇格させず、実態を記した新 ADR で supersede

`docs/adr/README.md` の索引の Status 列も合わせる。

- [ ] **Step 10: コミット**

```bash
git add docs/adr/
git commit -m "docs(adr): promote ADR-0040..0043 to Accepted after hardware validation"
```

---

## Self-Review

**1. Spec coverage（受入基準 11 項目）:**

| 受入基準 | 実装するタスク | 検証するタスク |
|---|---|---|
| tkinter 無しで起動し OBS に表示 | 4, 7 | 4-Step2（テスト）, 9-Step3 |
| 既存 config が無改変で動く | 4-Step1（既定 TK） | 4-Step9, 9-Step8 |
| フォント等の変更が反映される | 6, 7 | 6（テスト）, 9-Step4 |
| 表示時間・履歴・トリム・区切り・position が tk と同じ | 3（状態機械の共有） | 3（テスト）, 7（テスト） |
| OBS 未起動でも起動でき後から出る | 7 | 9-Step2, 9-Step3 |
| OBS 再起動に音声が耐え字幕が復帰 | 7 | 9-Step5 |
| 認証失敗・ソース不在で終了 | 7（worker_startup） | 7（テスト）, 9-Step6 |
| url/ソース名未設定で起動時に列挙して終了 | 5 | 5（テスト） |
| reload でスタイルが追従 | 7（`_run_session`） | 9-Step7 |
| 透過時に縁が出ない | 6（`bk_opacity=0`） | 9-Step4 |
| 他 worker を数十 ms 待たせない | 7（wrap 不使用・ビジーループ廃止） | 9-Step4（目視）※下記 |

**ギャップ 1つ:** 最後の「数十 ms 待たせない」を自動で守るテストが無い。OBS バックエンドは `wrap_text_to_width` を呼ばず 30fps ループも持たないので構造的に満たされるが、回帰を検出はできない。**受け入れる**: 直接測るには実 OBS が要り、構造で担保されている（`subtitle_obs.py` は `subtitle_tk` を import しない）。

**2. Placeholder scan:** Task 6 の観測値記入欄は意図的な空欄（Task 1 の成果物を書き写す指示）で、TBD ではない。それ以外に placeholder なし。

**3. Type consistency:**
- `ingest_text` / `Texts` / `Text`: Task 3 で定義 → Task 4（tk）, 7（obs）が同じ名前で使用 ✓
- `age_panels` / `next_expiry_sec`: Task 3 で定義 → Task 7 の `_run_session` で使用 ✓
- `build_text_settings(text_config, subtitle_config)`: Task 6 で定義 → Task 7 の `push_styles` が `build_text_settings(ts.config, config)` で呼ぶ ✓
- `ObsWsClient.request(request_type, request_data)`: Task 2 で定義 → Task 7 の `ObsRequester` Protocol と `FakeObsClient` が同じ signature ✓
- `ObsResourceNotFoundError`: Task 2 で定義 → Task 7 の `validate_sources` が送出、テストが捕捉 ✓
- `SubtitleWorkerType` / `SubtitleObsConfig`: Task 4 / 5 で定義 → Task 7, 8 で使用 ✓
- `subtitle_obs_worker(context, in_queue)`: Task 7 で定義 → Task 4 のディスパッチャが同じ引数で呼ぶ ✓
