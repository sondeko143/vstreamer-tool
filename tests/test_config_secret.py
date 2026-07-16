import json
from typing import TypeAliasType
from typing import get_args
from typing import get_origin

from pydantic import BaseModel
from pydantic import SecretStr

from vspeech.config import Config


def test_secret_roundtrip_via_model_dump_json():
    cfg = Config.model_validate(
        {
            "ami": {"appkey": "topsecret"},
            "gcp": {"service_account_info": {"private_key": "gcpsecret"}},
        }
    )
    assert cfg.ami.appkey.get_secret_value() == "topsecret"
    assert cfg.gcp.service_account_info["private_key"].get_secret_value() == "gcpsecret"

    dumped = cfg.model_dump_json()
    # field_serializer(when_used="json") により秘密値は平文で出力される（GUI→本体の受け渡しに必須）
    data = json.loads(dumped)
    assert data["ami"]["appkey"] == "topsecret"
    assert data["gcp"]["service_account_info"]["private_key"] == "gcpsecret"

    reloaded = Config.model_validate(data)
    assert reloaded.ami.appkey.get_secret_value() == "topsecret"
    secret = reloaded.gcp.service_account_info["private_key"]
    assert secret.get_secret_value() == "gcpsecret"


def _resolve_type_alias(annotation: object) -> object:
    """PEP 695 `type X = ...` aliases (e.g. `ServiceAccountInfo`) don't unwrap
    under `get_origin`/`get_args` on their own -- `model_fields[...].annotation`
    hands back the `TypeAliasType` object itself. Follow `.__value__` until we
    reach the real annotation."""
    while isinstance(annotation, TypeAliasType):
        annotation = annotation.__value__
    return annotation


def _iter_secret_str_fields(model: type[BaseModel], prefix: str = ""):
    """Walk ``model``'s pydantic v2 schema (``model_fields``) recursively and
    yield ``(dotted_path, kind)`` for every field that is, or contains,
    ``SecretStr``:

    - ``kind == "scalar"`` for a direct (optionally ``| None``) ``SecretStr`` field.
    - ``kind == "dict"`` for a ``dict[str, SecretStr]`` field.

    Nested ``BaseModel`` fields (direct or inside a union) are recursed into.
    This is schema introspection, not a hardcoded field list, so a new
    ``SecretStr`` anywhere in ``Config`` is discovered automatically.
    """
    for name, field in model.model_fields.items():
        path = f"{prefix}{name}"
        annotation = _resolve_type_alias(field.annotation)
        args = get_args(annotation)
        candidates = [annotation, *(_resolve_type_alias(a) for a in args)]
        for candidate in candidates:
            if candidate is SecretStr:
                yield path, "scalar"
                break
            candidate_origin = get_origin(candidate)
            if candidate_origin is dict:
                dict_args = get_args(candidate)
                if (
                    len(dict_args) == 2
                    and _resolve_type_alias(dict_args[1]) is SecretStr
                ):
                    yield path, "dict"
                    break
            if isinstance(candidate, type) and issubclass(candidate, BaseModel):
                yield from _iter_secret_str_fields(candidate, prefix=f"{path}.")
                break


def test_every_secret_str_field_survives_export_to_toml():
    """export_to_toml() は SecretStr を手作業でフィールド名指定して展開している
    (vspeech/config.py の export_to_toml 参照)。新しい SecretStr フィールドを
    足してその手作業リストへの追加を忘れると、model_dump() が返す生の SecretStr
    オブジェクトが toml.dumps に渡り、マスクされた値 ("**********") か
    "SecretStr(...)" の repr が書き出される -- GUI の保存経路がユーザーの実
    config ファイルを静かに壊す。

    ここでは Config.model_fields を再帰的に歩いて SecretStr 型のフィールドを
    機械的に列挙する (ハードコードされた一覧ではない)。見つけた各フィールドに
    識別可能な sentinel をセットし、export_to_toml() の出力に平文で現れる
    こと、マスク済みマーカーや SecretStr の repr が一切現れないことを検証する。
    """
    fields = list(_iter_secret_str_fields(Config))
    assert fields, (
        "walker found no SecretStr fields in Config -- it is almost certainly "
        "broken (Config.ami.appkey alone should have matched)"
    )

    config = Config()
    sentinels: dict[str, str] = {}
    for path, kind in fields:
        sentinel = f"SENTINEL_{path.replace('.', '_').upper()}"
        sentinels[path] = sentinel
        *parents, leaf = path.split(".")
        target = config
        for part in parents:
            target = getattr(target, part)
        if kind == "scalar":
            setattr(target, leaf, SecretStr(sentinel))
        else:  # "dict"
            secret_dict: dict[str, SecretStr] = getattr(target, leaf)
            secret_dict[f"{leaf}_key"] = SecretStr(sentinel)

    dumped = config.export_to_toml()

    for path, sentinel in sentinels.items():
        assert sentinel in dumped, (
            f"SecretStr field {path!r} did not survive export_to_toml() in "
            "plaintext -- export_to_toml()'s hand-unwrap list is missing it"
        )
    assert "**********" not in dumped, (
        "export_to_toml() output contains pydantic's masked SecretStr marker "
        "('**********') -- some SecretStr field was dumped unwrapped"
    )
    assert "SecretStr(" not in dumped, (
        "export_to_toml() output contains a SecretStr repr -- some SecretStr "
        "field was dumped unwrapped"
    )
