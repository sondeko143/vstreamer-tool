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
import reprlib
from asyncio import wait_for
from typing import Any
from typing import Protocol
from uuid import uuid4

from websockets.exceptions import ConnectionClosed
from websockets.frames import Close

RPC_VERSION = 1

OP_HELLO = 0
OP_IDENTIFY = 1
OP_IDENTIFIED = 2
OP_REQUEST = 6
OP_REQUEST_RESPONSE = 7

STATUS_RESOURCE_NOT_FOUND = 600

# 生の repr() は危険: OBS (ピア) が選べる JSON のネスト深さに比例して Python
# の呼び出しスタックを消費するため、json.loads() 自体は生き延びる深さでも
# repr() だけが RecursionError で落ちる窓がある (オブジェクトのネストで
# 概ね深さ 9000-9600、残りの C スタックに応じて動く。配列のネストではこの窓
# は開かない — json.loads() 側が先に力尽きるため)。このモジュールは
# ObsProtocolError の本文を組み立てる最中にまさにその repr() を呼ぶので、
# 素の repr() のままだと例外を作ろうとして別の (許容外の) 例外を漏らして
# しまう。reprlib.Repr は深さを自前のカウンタで打ち切るため安全。
#
# maxstring/maxother が制限するのは「葉 (leaf) 1 つあたり」の長さだけで、
# 総出力長には効かない。maxlevel=6 の
# 中でも maxlist=6/maxdict=4 は「幅」を許すため、6 段のネストの中に最大
# 6^6 ≈ 46656 個もの葉 (各 ≤200 文字) が並びうる。実測: 深さ 4 段・幅 6 の
# 入力 (約 328 KB のフレーム) から 262 KB の例外文字列が組み上がる —
# json.loads() が生き延びる深さ (RecursionError の窓) にも、
# 1 リーフの長さにも触れていない。つまり maxstring/maxother は「1 MiB の
# ピアフレームがそのまま 1 MiB の例外文字列 (ひいてはログ行) になるのを
# 防ぐ」という副次効果を持たない。総出力長を抑えるのは _bounded_repr() の
# 役目で、このモジュールの例外メッセージはすべて _SAFE_REPR.repr() を直接
# 使わず _bounded_repr() 経由にすること。
_SAFE_REPR = reprlib.Repr(maxlevel=6, maxstring=200, maxother=200)

# _bounded_repr() が返す文字列の上限。9 箇所ある呼び出し元のうち最長の
# プレフィックス文字列 (request_type を含むもの) がおよそ 60 文字なので、
# 300 + len("…(truncated)") を足しても例外メッセージ全体は余裕を持って
# 500 文字を下回る。
_BOUNDED_REPR_MAX_CHARS = 300


def _bounded_repr(x: Any) -> str:
    """`_SAFE_REPR.repr(x)` を、レンダリング後の文字列そのものに対して
    さらに固定の総文字数まで切り詰めて返す。

    `_SAFE_REPR` 単体は深さ (`maxlevel`) と葉 1 つあたりの長さ
    (`maxstring`/`maxother`) しか制限しないため、幅 (`maxlist`/`maxdict`)
    の分だけ葉が並ぶと、深さに関係なく合計は簡単に数百 KB になりうる
    (上のモジュールコメント参照)。ここで最終的にレンダリング済みの文字列
    そのものを切り詰めることで、幅にも深さにも依存しない総量の上限を
    保証する。
    """
    rendered = _SAFE_REPR.repr(x)
    if len(rendered) <= _BOUNDED_REPR_MAX_CHARS:
        return rendered
    return rendered[:_BOUNDED_REPR_MAX_CHARS] + "…(truncated)"


class ObsProtocolError(Exception):
    """obs-websocket との対話が想定外の形になった。"""


class ObsIdentifyError(ObsProtocolError):
    """Identify が成立しなかった (認証失敗、Hello/Identified 以外の op が来た、など)。

    RPC バージョン不一致はここでは検査しない: obs-websocket 側がそれを検出
    すると接続そのものを閉じるため、この関数まで来た時点では起こり得ず、
    検査しても死んだコードになる。

    リトライしても直らない種類なので、呼び出し側は fail-loud に扱う (ADR-0042)。
    """


class ObsRequestError(ObsProtocolError):
    def __init__(self, request_type: str, code: int, comment: str):
        self.request_type = request_type
        self.code = code
        self.comment = comment
        # comment は OBS 側の自由文字列で長さの取り決めが無い最後の
        # unbounded な peer->message 経路だった: 呼び出し側 (subtitle worker) はこれをリトライループ
        # 上で毎回ログするので、悪意/単に冗長なピアがリトライのたびに巨大な
        # ログ行を吐かせられる。属性 (self.comment) は呼び出し側が生の値を
        # 読めるよう素通しのまま残し、例外メッセージだけ切り詰める。通常長の
        # comment はそのまま読めるよう、_SAFE_REPR の repr() 化 (クォート付き
        # で読みにくくなる) ではなくスライスで済ませる。
        #
        # comment はアノテーション上 str だが、このクラスは公開クラスであり
        # このモジュール外の呼び出し側 (呼び出し側が独自に組み立てる場合も
        # 含む) が任意の値で直接構築しうる。ピア経由の 2 箇所の構築元は
        # どちらも呼び出し前に isinstance(comment, str) を検査済みなので
        # ここに非 str が来ることはないが、`len(comment)` を無検査で呼ぶと
        # 例えば `ObsRequestError("X", 1, None)` が素の TypeError で死ぬ。
        # isinstance で分岐し、非 str でも
        # コンストラクタ自体は決して例外を漏らさない total な形に戻す。
        if isinstance(comment, str):
            bounded_comment = (
                comment if len(comment) <= 200 else comment[:200] + "…(truncated)"
            )
        else:
            bounded_comment = _bounded_repr(comment)
        super().__init__(
            f"{request_type} failed: code={_bounded_repr(code)} {bounded_comment}"
        )


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
    secret = base64.b64encode(
        hashlib.sha256((password + salt).encode("utf-8")).digest()
    )
    return base64.b64encode(
        hashlib.sha256(secret + challenge.encode("utf-8")).digest()
    ).decode("utf-8")


# obs-websocket が自らハンドシェイクを拒否したことを表明する専用の close
# code 帯 (5.x のプロトコル文書 WebSocketCloseCode がここに全ての拒否理由
# を割り当てている。パスワード誤り = 4009 が実測での唯一の実例で、RPC
# バージョン不一致・不正な Identify なども同じ帯を使う)。この帯に入る close
# だけが「再接続しても直らない」と型で示せる signal で、identify() はこれ
# だけを ObsIdentifyError に変換する。
_HANDSHAKE_REJECTION_CLOSE_CODES = range(4000, 5000)


def _handshake_rejection(e: ConnectionClosed) -> Close | None:
    """`e` が obs-websocket 自身によるハンドシェイク拒否の close なら、その
    close frame (code/reason) を返す。そうでなければ None。

    `e.rcvd` は相手 (OBS) から実際に届いた close frame。`e.sent` (自分側が
    送った close) は判定に使わない -- 拒否理由を表明するのは相手だけなので。
    close frame を一切受け取らずに切れた場合 (トランスポートの生の切断、
    `e.rcvd is None`。例: 1006 相当や、プロセスが即死してハンドシェイクの
    途中で TCP だけ落ちたケース) は判定材料が無いので拒否とは扱わない --
    「まだ繋がっていないだけ」かもしれず、リトライで直りうる。
    """
    rcvd = e.rcvd
    if rcvd is None:
        return None
    if rcvd.code not in _HANDSHAKE_REJECTION_CLOSE_CODES:
        return None
    return rcvd


class ObsWsClient:
    """obs-websocket 5.x の薄いクライアント。

    Hello(0)/Identify(1)/Identified(2) のハンドシェイクと Request(6)/
    RequestResponse(7) の往復のみをサポートする。イベント購読・バッチ
    リクエスト・msgpack シリアライザは扱わない (ADR-0043)。想定外の応答は
    すべてこのモジュールの例外 (`ObsProtocolError` とそのサブクラス) として
    送出し、呼び出し側が `OSError` / `websockets` の例外と区別して扱えるように
    する (ADR-0042)。
    """

    def __init__(self, transport: ObsTransport, timeout: float = 5.0):
        self._transport = transport
        self._timeout = timeout

    async def _send(self, op: int, d: dict[str, Any]) -> None:
        await self._transport.send(json.dumps({"op": op, "d": d}))

    async def _recv(self) -> dict[str, Any]:
        try:
            raw = await wait_for(self._transport.recv(), timeout=self._timeout)
        except TimeoutError as e:
            raise ObsProtocolError(
                f"OBS からの応答が {self._timeout} 秒以内に来なかった"
            ) from e
        if isinstance(raw, bytes):
            try:
                raw = raw.decode("utf-8")
            except UnicodeDecodeError as e:
                raise ObsProtocolError(f"OBS から UTF-8 でないフレーム: {e}") from e
        try:
            message = json.loads(raw)
        except (ValueError, RecursionError) as e:
            # json.loads には 2 通りの壊れ方がある: 不正な JSON 構文
            # (JSONDecodeError, ValueError のサブクラス) と、深すぎるネスト
            # (RecursionError, RuntimeError のサブクラスで ValueError では
            # 拾えない)。後者は "[" * N + "]" * N のような数万バイトの ASCII
            # 文字列だけで作れ、websockets のデフォルト max_size (1 MiB) を
            # 素通りする transport-valid な入力なので、ここで一緒に拾う。
            raise ObsProtocolError(f"OBS から不正な JSON: {e}") from e
        if not isinstance(message, dict) or "op" not in message:
            raise ObsProtocolError(
                f"OBS から不正なメッセージ: {_bounded_repr(message)}"
            )
        if not isinstance(message.get("d"), dict):
            raise ObsProtocolError(
                f"OBS からのメッセージに 'd' が無い: {_bounded_repr(message)}"
            )
        return message

    async def identify(self, password: str) -> None:
        """接続直後の Hello を受け取り、必要なら認証して Identify を送り、
        Identified を待つ。

        接続ごとに一度だけ呼ぶ想定。obs-websocket は Hello/Identified の形式
        異常だけでなく、ハンドシェイクそのものの拒否 (認証失敗・RPC バージョン
        不一致・不正な Identify など) もエラーメッセージでは返さない -- 代わりに
        WebSocket を 4000-4999 (private use) の close code で切る (実測: OBS
        32.1.2 / obs-websocket 5.7.3、誤ったパスワードで code 4009
        "Authentication failed.")。この関数は、検出できる失敗を
        すべて `ObsIdentifyError` (`ObsProtocolError` のサブクラス) として
        送出する -- リトライしても直らない失敗だと呼び出し側 (ADR-0042) が型で
        見分けられるようにするため。

        一方、close code が無い (`ConnectionClosed.rcvd is None`、例えば接続が
        ハンドシェイクの途中で生の TCP レベルで落ちた場合) 切断や、4000-4999
        帯の外の close (1006 のようなトランスポートレベルの異常切断など) は
        ハンドシェイクの拒否ではなく、単に OBS にまだ繋がっていないだけかも
        しれない -- リトライで直りうるので `ObsIdentifyError` には変換せず、
        `websockets` の `ConnectionClosed` (`WebSocketException` のサブクラス)
        のまま呼び出し側へ伝播させる。呼び出し側はそれを fail-open (バックオフ
        再接続) として扱う (ADR-0042)。
        """
        try:
            message = await self._recv()
            if message["op"] != OP_HELLO:
                # message['op'] は _recv() が「キーとして存在する」ことしか保証して
                # いない生のピア値。f-string の
                # {x} は format() 経由で結局 dict.__repr__ を呼ぶので、!r を使って
                # いなくても repr() の再帰ハザードと
                # 無制限長ハザードを踏む。_bounded_repr() で両方封じる (深さは
                # reprlib の maxlevel、総幅は _bounded_repr() 自身の切り詰めで)。
                raise ObsIdentifyError(
                    f"Hello を期待したが op={_bounded_repr(message['op'])} が来た"
                )
            hello_data = message["d"]
            d: dict[str, Any] = {"rpcVersion": RPC_VERSION}
            # 「authentication キーが無い」と「あるが偽値」を区別する: obs-websocket
            # は認証が要る場合にのみこのキーを載せる。真偽 (`if auth:`) で判定すると
            # `{}` / `[]` / `0` / `false` / `""` を「認証不要」と誤読して無認証の
            # Identify を送ってしまい、相手が実際には認証必須なら 4008 で切られて
            # 呼び出し側が延々リトライすることになる (壊れたハンドシェイクはリトライ
            # しても直らない)。聞くべきは値の真偽ではなくキーの有無。
            if "authentication" in hello_data:
                auth = hello_data["authentication"]
                if not password:
                    raise ObsIdentifyError(
                        "OBS が認証を要求していますが subtitle.obs.password が空です"
                    )
                if not isinstance(auth, dict):
                    raise ObsIdentifyError(
                        f"OBS の authentication が不正な形: {_bounded_repr(auth)}"
                    )
                salt = auth.get("salt")
                challenge = auth.get("challenge")
                if not isinstance(salt, str) or not isinstance(challenge, str):
                    raise ObsIdentifyError(
                        "OBS の authentication に salt/challenge が無い:"
                        f" {_bounded_repr(auth)}"
                    )
                try:
                    d["authentication"] = build_auth_string(password, salt, challenge)
                except UnicodeError as e:
                    # isinstance(salt, str) / isinstance(challenge, str) は UTF-8
                    # エンコード可能であることまでは保証しない (例:
                    # json.loads('"\\ud800"') は非対の surrogate を含む str を返す)。
                    # build_auth_string() 自身はこの節の探索対象外の純粋関数として
                    # 残す (専用ユニットテストを持つ) ので、ここで包んで
                    # identify 時の失敗として fail-loud にする (ADR-0042)。
                    raise ObsIdentifyError(
                        f"OBS の authentication の salt/challenge が UTF-8 として不正: {e}"
                    ) from e
            await self._send(OP_IDENTIFY, d)
            message = await self._recv()
            if message["op"] != OP_IDENTIFIED:
                # 上の Hello ガードと同じハザード。
                raise ObsIdentifyError(
                    f"Identified を期待したが op={_bounded_repr(message['op'])} が来た"
                )
        except ConnectionClosed as e:
            # obs-websocket はエラーメッセージを送らず、close code で拒否を
            # 表明する (このメソッドの docstring 参照)。
            rejection = _handshake_rejection(e)
            if rejection is None:
                # ハンドシェイクの拒否ではない、ただの (リトライで直りうる)
                # 切断。ObsIdentifyError には変換せず、ConnectionClosed の
                # まま呼び出し側の fail-open 経路へ渡す。
                raise
            raise ObsIdentifyError(
                f"OBS がハンドシェイクを拒否した: {rejection}"
            ) from e

    async def request(
        self, request_type: str, request_data: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """obs-websocket に Request を送り、対応する RequestResponse を待って
        `responseData` を返す。

        **並行呼び出しに対して安全ではない。** 2 回の呼び出しが同時に走ると、
        どちらも同じ `_recv()` ストリームを消費するため互いの応答を取り違え
        うる。呼び出し側 (subtitle worker) が 1 リクエストずつ順に呼ぶことを
        前提にした設計であり、直す必要のあるバグではなく明文化した制約。
        """
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
            # 集約デッドライン無し (per-recv の self._timeout のみ)。無関係な
            # メッセージを timeout より速く送り続ける相手だとここが詰まりうる
            # が、対象はローカル OBS でありリスクは低いと判断して見送った。
            # 再現する運用が出たらここに集約デッドラインを足す。
            message = await self._recv()
            # イベント (op 5) や他リクエストの応答は捨てる。
            if message["op"] != OP_REQUEST_RESPONSE:
                continue
            d = message["d"]
            if d.get("requestId") != request_id:
                continue
            status = d.get("requestStatus")
            if not isinstance(status, dict):
                raise ObsProtocolError(
                    f"{request_type} の応答に requestStatus が無い: {_bounded_repr(d)}"
                )
            if not status.get("result"):
                code = status.get("code", 0)
                comment = status.get("comment", "")
                # ObsRequestError.code/comment のアノテーション (int/str) を
                # 嘘にしないための型検査。これが無いと 2 つ実害が出る:
                # (1) `code == STATUS_RESOURCE_NOT_FOUND` は int の 600 としか
                #     一致しないので、相手が "600" (str) を送ると
                #     ObsResourceNotFoundError が汎用の ObsRequestError に
                #     こっそり降格し、呼び出し側の「リソースが無い」専用の
                #     fail-loud 経路が発火しなくなる。
                # (2) 検査を通さないと e.code / e.comment が str/list/dict/None
                #     になりうり、呼び出し側の `e.comment.lower()` のような
                #     アノテーション通りのコードが素の AttributeError で死ぬ。
                # ここは requestStatus 自体が壊れているケースなので、
                # ObsRequestError ではなく ObsProtocolError で fail-loud にする
                # (`code`/`comment` を捏造して ObsRequestError を作ると同じ嘘を
                # 一段先送りするだけ)。
                # isinstance(True, int) は True になる (bool は int のサブクラス)
                # ため、素の isinstance(code, int) だけだと相手が code に
                # JSON の true/false を送ってきても素通りし、
                # ObsRequestError.code に (600 とは絶対に一致しない) bool が
                # 入ってしまう。これは「ガードが証明していることが足りない」
                # という、このモジュールで繰り返し踏んでいる形そのものなので
                # bool を明示的に除外する。
                if (
                    not isinstance(code, int)
                    or isinstance(code, bool)
                    or not isinstance(comment, str)
                ):
                    raise ObsProtocolError(
                        f"{request_type} の応答の requestStatus.code/comment が"
                        f" 不正な形: {_bounded_repr(status)}"
                    )
                if code == STATUS_RESOURCE_NOT_FOUND:
                    raise ObsResourceNotFoundError(request_type, code, comment)
                raise ObsRequestError(request_type, code, comment)
            response_data = d.get("responseData")
            if response_data is None:
                return {}
            if not isinstance(response_data, dict):
                # requestStatus の isinstance 検査と非対称にしない: responseData
                # だけ無検査だと、呼び出し側 (subtitle worker) の
                # `result["inputSettings"]` のようなアクセスで同じバグ形が
                # 1 フレーム先送りされて素の TypeError になる。
                raise ObsProtocolError(
                    f"{request_type} の応答の responseData が不正な形:"
                    f" {_bounded_repr(response_data)}"
                )
            return response_data
