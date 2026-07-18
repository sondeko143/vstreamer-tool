from asyncio import get_running_loop
from collections.abc import Sequence
from typing import Any

import grpc
from google.auth import default as google_auth_default
from google.auth.compute_engine import Credentials as CeCredentials
from google.auth.compute_engine import IDTokenCredentials as CeIdTokenCredentials
from google.auth.credentials import Credentials as BaseCredentials
from google.auth.credentials import with_scopes_if_required
from google.auth.exceptions import TransportError
from google.auth.transport.grpc import AuthMetadataPlugin
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
from google.oauth2.service_account import IDTokenCredentials
from grpc import aio
from requests import Session
from requests.adapters import HTTPAdapter
from requests.adapters import Retry

from vspeech.config import GcpConfig
from vspeech.config import ServiceAccountInfo

type GcpIDTokenCredentials = IDTokenCredentials | CeIdTokenCredentials

# トークン更新 (oauth2.googleapis.com) 用の retry。
#
# 更新は概ねトークンの有効期限 (約 1 時間) ごとにしか起きないので、その間
# プールに残った TLS 接続は Google 側にとっくに閉じられている。urllib3 は
# 貸し出し前に `is_connection_dropped()` を見るが、それで拾えるのは「貸す前に
# 既に死んでいた」接続だけで、本番で観測したのは POST を書いた後に RST が来る
# 窓 (traceback は `getresponse()` -> `recv_into` で落ちていた)。チェックと
# 書き込みの間に死ぬので死活チェックでは塞げず、retry でしか救えない。
#
# allowed_methods に POST を入れるのが要: urllib3 の既定 (DEFAULT_ALLOWED_METHODS)
# は非冪等な POST を retry 対象から外すが、トークン更新はまさに POST なので、
# 既定のままでは何も retry されない。ここでの二重送信は「トークンをもう 1 本
# 貰う」だけで害が無く、逆に retry しなければこの窓は永久に塞がらない。
# GET も入れておく (メタデータサーバ経路)。urllib3 は「全メソッド」を意味する
# allowed_methods=False も受けるが、それは型スタブ上 Collection[str] | None に
# 無く ty が弾くうえ、明示した方が「POST を意図して含めた」と読める。
_AUTH_RETRY = Retry(
    total=3,
    connect=3,
    read=3,
    # HTTP ステータスでの retry は google.auth 自身の役目 (`_client` の
    # ExponentialBackoff が 429/503 などを解釈して retry し、invalid_grant は
    # 然るべき例外にする) なので、ここでは接続層だけを見る。
    status=0,
    # status=0 だけでは足りない。urllib3 の `Retry.is_retry()` は
    # respect_retry_after_header (既定 True) が有効で Retry-After ヘッダが
    # 付いていると、status_forcelist を一切見ずに retry すべきと判断する。
    # そこへ status=0 が重なると 0 -> -1 で即 is_exhausted になり、429/503 が
    # 「retry もされず、応答本文も失われた RetryError」に化ける
    # (実測: Retry-After 付き 503 が attempts=1 で RetryError)。つまり
    # google.auth に任せるどころか、google.auth が持っていた retry を奪う。
    # ヘッダの尊重ごと切って、ステータスの扱いを完全に google.auth へ返す。
    respect_retry_after_header=False,
    allowed_methods=frozenset({"GET", "POST"}),
    backoff_factor=0.25,
)

# トークン更新 1 回あたりの上限秒。
#
# google.auth はトークンエンドポイントへの POST に timeout を渡さないので、
# `Request.__call__` の既定 (_DEFAULT_TIMEOUT = 120 秒) がそのまま効く。retry を
# 足すと試行回数ぶん最悪時間が伸び、応答を返さない (blackhole) エンドポイント
# 相手では 4 x 120 + backoff = 約 481 秒、gRPC は認証 plugin を呼び出しごとの
# 別スレッドで走らせ RPC の deadline を過ぎてもそのスレッドは生き残るため、
# 詰まったスレッドが積み上がる。
#
# 1 試行を 20 秒で切れば最悪でも 4 x 20 + 1.5 = 約 81.5 秒で、retry を入れた後
# でも修正前 (120 秒) より短くなる。Google のトークンエンドポイントの正常応答は
# 1 秒未満なので 20 秒は十分に緩い。
_AUTH_REQUEST_TIMEOUT_SEC = 20.0


def _cap(value: float | None) -> float:
    return (
        _AUTH_REQUEST_TIMEOUT_SEC
        if value is None
        else min(value, _AUTH_REQUEST_TIMEOUT_SEC)
    )


class _BoundedTimeoutAdapter(HTTPAdapter):
    """呼び出し側が渡した timeout を `_AUTH_REQUEST_TIMEOUT_SEC` で頭打ちにする。

    google.auth 側に timeout を渡す口が無い (トークン更新の呼び出しは
    `_client._token_endpoint_request` の中で完結している) ので、session 側で
    上限をかける。requests は timeout を float か (connect, read) のタプルで
    渡してくるため両方を扱う。
    """

    def send(self, request, stream=False, timeout=None, *args, **kwargs):
        if isinstance(timeout, tuple):
            connect, read = timeout
            timeout = (_cap(connect), _cap(read))
        else:
            timeout = _cap(timeout)
        return super().send(request, stream=stream, timeout=timeout, *args, **kwargs)


def build_auth_session() -> Session:
    """トークンエンドポイント用の `requests.Session`。

    `google.auth.transport.requests.Request()` は session を渡さないと素の
    `requests.Session()` (retry 無し) を作る。そこに死んだプール接続を掴む窓が
    あるので、retry を積んだ session を自前で用意して渡せるようにする。
    """
    session = Session()
    adapter = _BoundedTimeoutAdapter(max_retries=_AUTH_RETRY)
    # http:// も本番で効く経路であって、テスト専用ではない: compute engine の
    # credentials はメタデータサーバ (http://metadata.google.internal) を
    # この session で叩く。
    #
    # 加えて、**両方に同一の adapter を張ること**自体が要件でもある: 別々に
    # すると、テストが叩く http:// 側だけ retry が付いていて本番の https:// 側は
    # 素、という状態がテスト GREEN のまま成立する。
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# GAPIC の transport が自分でチャネルを作るときに渡す options。
#
# チャネルをこちらから渡すとその分岐ごと飛ばされ、受信上限が gRPC 既定の 4 MiB に
# 戻る (実測: 5 MiB の応答が RESOURCE_EXHAUSTED)。Translate と Speech の
# transport はどちらもこの 2 つを渡していることを確認済みだが、**新しい
# クライアントを足すときは、その transport が本当にこれと同じものを渡して
# いるか確かめること** -- 違うものを渡していたら、この定数を使い回すと
# 静かに落とすことになる。
GAPIC_DEFAULT_CHANNEL_OPTIONS: tuple[tuple[str, Any], ...] = (
    ("grpc.max_send_message_length", -1),
    ("grpc.max_receive_message_length", -1),
)


def create_auth_metadata_plugin(
    credentials: BaseCredentials, host: str, scopes: Sequence[str]
) -> AuthMetadataPlugin:
    """gRPC の認証メタデータ plugin を、retry 付き session で更新するよう組む。

    `google.api_core.grpc_helpers._create_composite_credentials` が既定でやって
    いることと同じだが、あちらは `Request()` を引数無しで作る (session を注入
    する口が無い) ため、そこだけを差し替えるために手で組み直している。

    `with_scopes_if_required` と `default_host` は api_core に合わせる:
    前者を省くとサービスアカウントがスコープ無しでトークンを取り拒否され、
    後者を省くとサービスアカウントの self-signed JWT 経路が無効になる。
    """
    scoped = with_scopes_if_required(credentials, scopes=None, default_scopes=scopes)
    request = Request(session=build_auth_session())
    return AuthMetadataPlugin(scoped, request, default_host=host)


def create_auth_channel(
    credentials: BaseCredentials,
    host: str,
    scopes: Sequence[str],
    options: Sequence[tuple[str, Any]] = (),
) -> aio.Channel:
    """`create_auth_metadata_plugin` の認証を載せた grpc.aio チャネル。

    `options` は呼び出し側 (GAPIC transport を使う側) が必ず渡すこと。
    transport が自分でチャネルを作るときに渡している options は、チャネルを
    こちらが渡した瞬間に適用されなくなる -- 既定の 4 MiB 受信上限に戻る。
    """
    # チャネルは生成時点の実行中ループに束縛される (`grpc/aio/_channel.py` の
    # `get_working_loop()`)。ループが走っていないと例外にはならず、走っていない
    # 新しいループが黙って結び付けられ、その後の RPC が永久に返らなくなる。
    # 今の呼び出し元は async 関数だけだが、preflight (ADR-0045) は同期なので
    # そこに GCP の生存確認を足した誰かが静かにこの罠を踏みうる。声を上げさせる。
    get_running_loop()
    call_credentials = grpc.metadata_call_credentials(
        create_auth_metadata_plugin(credentials, host=host, scopes=scopes)
    )
    channel_credentials = grpc.composite_channel_credentials(
        grpc.ssl_channel_credentials(), call_credentials
    )
    return aio.secure_channel(f"{host}:443", channel_credentials, options=list(options))


def unescape_private_key(service_account_info: ServiceAccountInfo):
    decoded = {k: v.get_secret_value() for k, v in service_account_info.items()}
    if "private_key" in service_account_info:
        return {
            **decoded,
            "private_key": decoded["private_key"].replace("\\n", "\n"),
        }
    return decoded


def get_credentials(config: GcpConfig) -> tuple[Credentials | CeCredentials, str]:
    """サービスアカウントの読み込みに `scopes=` を渡さないこと (ADR-0048)。

    スコープを付けないサービスアカウントは、トークンを audience ベースの
    self-signed JWT としてローカルで組み立てる -- ネットワークに出ない。
    `scopes=` を渡した瞬間その分岐から外れ、1 時間ごとに
    oauth2.googleapis.com へ往復するようになる。つまり ADR-0048 がまさに
    問題にしている経路が、設定を足しただけで静かに復活する
    (`service_account.py` の `_use_self_signed_jwt` 参照)。
    """
    if config.service_account_file_path:
        file_path = config.service_account_file_path.expanduser()
        cred = Credentials.from_service_account_file(file_path)
        return cred, cred.project_id
    elif config.service_account_info:
        decoded = unescape_private_key(config.service_account_info)
        cred = Credentials.from_service_account_info(decoded)
        return cred, cred.project_id
    elif config.use_ce_credentials:
        cred = CeCredentials()
        return cred, ""
    else:
        cred, project_id = google_auth_default()
        return cred, project_id or ""


def get_id_token_credentials(
    config: GcpConfig,
) -> GcpIDTokenCredentials | None:
    if config.service_account_file_path:
        file_path = config.service_account_file_path.expanduser()
        return IDTokenCredentials.from_service_account_file(
            filename=file_path, target_audience=""
        )
    elif config.service_account_info:
        decoded = unescape_private_key(config.service_account_info)
        return IDTokenCredentials.from_service_account_info(
            info=decoded, target_audience=""
        )
    elif config.use_ce_credentials:
        # Constructing CeIdTokenCredentials synchronously probes the GCE
        # metadata server (metadata.google.internal). On non-GCE hosts that
        # blocks for several seconds while DNS/connection retries exhaust
        # before raising TransportError, stalling sender worker startup. Only
        # pay that cost when the user has explicitly opted into CE credentials.
        try:
            # ここの Request は 1 回の更新で 3 本の HTTP を張る経路に埋まる:
            # `iam.Signer` がこれを抱え込み (`iam.py`)、更新のたびに
            # メタデータサーバへの GET と iamcredentials.googleapis.com への
            # signBlob POST を、この後のトークン POST より **先に** 行う。
            # 素の Request() だとその 2 本が retry 無しのままなので、同じ
            # stale-pool の窓が残る (ADR-0048)。
            return CeIdTokenCredentials(
                request=Request(session=build_auth_session()), target_audience=""
            )
        except TransportError:
            return None
    else:
        return None
