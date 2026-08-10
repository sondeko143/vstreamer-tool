from collections.abc import Callable
from typing import Protocol

import sounddevice as sd
from pydantic import AliasChoices
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from vspeech.config import PlaybackConfig
from vspeech.config import RecordingConfig
from vspeech.config import SampleFormat
from vspeech.config import StreamVcConfig
from vspeech.exceptions import DeviceNotFoundError
from vspeech.exceptions import DeviceRateUnresolvedError
from vspeech.logger import logger


class HostAPIInfo(BaseModel):
    index: int
    name: str


class DeviceInfo(BaseModel):
    # sounddevice's device dict is snake_case. Only the host api key is `hostapi`, so
    # pick it up under an alias.
    model_config = ConfigDict(populate_by_name=True)

    host_api: int = Field(validation_alias=AliasChoices("hostapi", "host_api"))
    max_input_channels: int
    max_output_channels: int
    name: str
    index: int


def list_all_devices(input: bool = False, output: bool = False):
    results: dict[str, int] = {}
    host_apis = sd.query_hostapis()
    for raw in sd.query_devices():
        d = DeviceInfo.model_validate(dict(raw))
        if input and d.max_input_channels <= 0:
            continue
        if output and d.max_output_channels <= 0:
            continue
        host_name = host_apis[d.host_api]["name"]
        results[f"{host_name}: {d.name}"] = d.index
    return results


def get_device_info(index: int) -> DeviceInfo:
    return DeviceInfo.model_validate(dict(sd.query_devices(index)))


def search_host_api_by_type(name: str):
    for i, host_api in enumerate(sd.query_hostapis()):
        if host_api["name"] == name:
            return HostAPIInfo(index=i, name=host_api["name"])


def search_device_by_name(
    name: str | None,
    host_api_index: int | None,
    input: bool = False,
    output: bool = False,
):
    if not name:
        default_input, default_output = sd.default.device
        if input:
            return get_device_info(default_input)
        elif output:
            return get_device_info(default_output)
        return None
    for raw in sd.query_devices():
        device = DeviceInfo.model_validate(dict(raw))
        if host_api_index is not None and host_api_index != device.host_api:
            continue
        if input and device.max_input_channels <= 0:
            continue
        if output and device.max_output_channels <= 0:
            continue
        if name in device.name:
            return device


def search_device(
    host_api_type: str | None,
    name: str | None,
    input: bool = False,
    output: bool = False,
):
    host_api_index = None
    if host_api_type:
        info = search_host_api_by_type(host_api_type)
        if info:
            host_api_index = info.index
    return search_device_by_name(name, host_api_index, input=input, output=output)


def _resolve_device(
    *,
    index: int | None,
    index_key: str,
    host_api_type: str | None,
    host_api_key: str,
    name: str | None,
    name_key: str,
    input: bool = False,
    output: bool = False,
) -> DeviceInfo:
    """The shared body of `resolve_input_device` / `resolve_output_device` (ADR-0038).

    An explicit index wins; otherwise search by host_api / name. If nothing is found,
    raise DeviceNotFoundError (its message carries the caller's config key).
    """
    if index is not None:
        try:
            return get_device_info(index)
        except Exception as e:
            raise DeviceNotFoundError(f"{index_key}={index} が無効です: {e}") from e
    device = search_device(
        host_api_type=host_api_type,
        name=name,
        input=input,
        output=output,
    )
    if device is None:
        kind = "入力" if input else "出力"
        raise DeviceNotFoundError(
            f"{kind}デバイスが見つかりません "
            f"({host_api_key}={host_api_type!r}, {name_key}={name!r})"
        )
    return device


def resolve_input_device(config: RecordingConfig) -> DeviceInfo:
    """Resolve the recording input device. Raises DeviceNotFoundError if absent.

    preflight and the recording worker go through the same path (ADR-0038).
    """
    return _resolve_device(
        index=config.input_device_index,
        index_key="recording.input_device_index",
        host_api_type=config.input_host_api_name,
        host_api_key="recording.input_host_api_name",
        name=config.input_device_name,
        name_key="recording.input_device_name",
        input=True,
    )


def resolve_output_device(config: PlaybackConfig) -> DeviceInfo:
    """Resolve the playback output device. Raises DeviceNotFoundError if absent.

    preflight and the playback worker go through the same path (ADR-0038).
    """
    return _resolve_device(
        index=config.output_device_index,
        index_key="playback.output_device_index",
        host_api_type=config.output_host_api_name,
        host_api_key="playback.output_host_api_name",
        name=config.output_device_name,
        name_key="playback.output_device_name",
        output=True,
    )


def resolve_stream_vc_input_device(config: StreamVcConfig) -> DeviceInfo:
    """Resolve streaming VC's own input device (preflight and capture share the path)."""
    return _resolve_device(
        index=config.input_device_index,
        index_key="stream_vc.input_device_index",
        host_api_type=config.input_host_api_name,
        host_api_key="stream_vc.input_host_api_name",
        name=config.input_device_name,
        name_key="stream_vc.input_device_name",
        input=True,
    )


def resolve_stream_vc_output_device(config: StreamVcConfig) -> DeviceInfo:
    """Resolve streaming VC's output device (preflight and playback share the path)."""
    return _resolve_device(
        index=config.output_device_index,
        index_key="stream_vc.output_device_index",
        host_api_type=config.output_host_api_name,
        host_api_key="stream_vc.output_host_api_name",
        name=config.output_device_name,
        name_key="stream_vc.output_device_name",
        output=True,
    )


_WASAPI_HOST_API = "Windows WASAPI"


def _wasapi_counterpart_rates(name: str, *, input: bool) -> dict[int, set[str]]:
    """WASAPI devices whose name starts with `name`, grouped by mix rate.

    PortAudio's WMME/DirectSound backends report a hardcoded 44100 for every device,
    so their `default_samplerate` cannot be trusted. Their device names, however, are
    the WASAPI names truncated to 31 characters, which makes the WASAPI row for the
    same endpoint findable by prefix (ADR-0071).

    Returns a mapping from mix rate to the set of matched WASAPI device names that
    reported it. The caller uses the number of keys to judge uniqueness, and the
    matched names to say which WASAPI row a resolved rate was borrowed from (or to
    list every rate it disagreed on, if more than one key comes back).
    """
    host_apis = sd.query_hostapis()
    matches: dict[int, set[str]] = {}
    for raw in sd.query_devices():
        raw_dict = dict(raw)
        device = DeviceInfo.model_validate(raw_dict)
        if host_apis[device.host_api]["name"] != _WASAPI_HOST_API:
            continue
        if input and device.max_input_channels <= 0:
            continue
        if not input and device.max_output_channels <= 0:
            continue
        if device.name.startswith(name):
            rate = int(round(raw_dict["default_samplerate"]))
            matches.setdefault(rate, set()).add(device.name)
    return matches


def resolve_device_rate(
    device: DeviceInfo, override: int | None, *, input: bool, config_key: str
) -> tuple[int, str]:
    """The rate to open `device` at, plus a human-readable note on how it was decided.

    Order: explicit config -> the device's own default_samplerate when it is a WASAPI
    device -> the mix rate of its WASAPI counterpart (ADR-0071). Anything ambiguous
    raises rather than guessing: opening at the wrong rate silently reinstates the OS
    resampler that ADR-0070 exists to remove. A resolved rate of 0 or less (some host
    APIs report this for a device in a bad state) is treated as unresolved too, so a
    broken endpoint fails loud here instead of surfacing later as an opaque English
    ValueError or PortAudio open error.
    """
    if override is not None:
        return override, f"{config_key} で明示"
    host_apis = sd.query_hostapis()
    host_api_name = host_apis[device.host_api]["name"]
    kind = "入力" if input else "出力"
    if host_api_name == _WASAPI_HOST_API:
        for raw in sd.query_devices():
            raw_dict = dict(raw)
            # Guard against a device table that shifted under us (e.g. after a
            # sd._terminate()/_initialize() cycle): an index match with a different
            # name is not this device, so treat it the same as "not found" (M4).
            if raw_dict["index"] != device.index or raw_dict["name"] != device.name:
                continue
            rate = int(round(raw_dict["default_samplerate"]))
            if rate <= 0:
                raise DeviceRateUnresolvedError(
                    f"WASAPI デバイス '{device.name}' の default_samplerate が "
                    f"{rate} で異常です。{config_key} に明示してください"
                )
            return rate, "WASAPI のミックス形式"
        raise DeviceRateUnresolvedError(
            f"WASAPI デバイス '{device.name}' (index={device.index}) が"
            f"デバイス一覧に見つかりません。{config_key} に明示してください"
        )
    matches = _wasapi_counterpart_rates(device.name, input=input)
    if len(matches) == 1:
        rate = next(iter(matches))
        matched_name = ", ".join(sorted(matches[rate]))
        if rate <= 0:
            raise DeviceRateUnresolvedError(
                f"{kind}デバイス '{device.name}' ({host_api_name}) の WASAPI 同名デバイス "
                f"'{matched_name}' の default_samplerate が {rate} で異常です。"
                f"{config_key} に明示してください"
            )
        return rate, f"WASAPI の '{matched_name}' から逆引き ({host_api_name} デバイス)"
    if not matches:
        detail = "対応する WASAPI デバイスが見つかりません"
    else:
        detail = f"対応する WASAPI デバイスのレートが一致しません ({sorted(matches)})"
    raise DeviceRateUnresolvedError(
        f"{kind}デバイス '{device.name}' ({host_api_name}) の実レートを判定できません: "
        f"{detail}。Windows のサウンド設定で「既定の形式」を確認し "
        f"{config_key} に明示してください"
    )


class ReportsSampleRate(Protocol):
    """The little `open_device_stream` needs back from a freshly built stream.

    Both `sd.RawInputStream` and `sd.RawOutputStream` satisfy it, which is what lets one
    helper serve the input and the output boundaries without knowing which it is holding.
    """

    def start(self) -> None: ...

    @property
    def samplerate(self) -> float: ...


def open_device_stream[StreamT: ReportsSampleRate](
    *,
    device: DeviceInfo,
    override: int | None,
    input: bool,
    config_key: str,
    opening: str,
    subject: str,
    open_stream: Callable[[int], StreamT],
    pipeline_rate: int | None = None,
) -> tuple[StreamT, int]:
    """Open `device` at its own native rate; return the stream and that rate.

    All four device boundaries (streaming VC in/out, utterance recording/playback) open a
    device the same way, and the *order* of the steps is what makes the open honest, so it
    lives here once instead of four times:

    1. resolve the rate right next to the device that was just resolved, so an open stays
       a single decision point and the rate has no second, cached copy to drift from. It
       re-decides nothing within one process: sd.query_devices() is cached at PortAudio
       init and nothing here re-initialises it (see the deferred note in
       worker/playback.py's search_appropriate_device), so a reopen resolves the same rate;
    2. log what is about to be attempted *before* the open, so a failing open still says
       which device, which rate, and how that rate was decided;
    3. build the stream -- the caller's own call, because blocksize / channels / dtype /
       latency differ per boundary -- and start it;
    4. say out loud when PortAudio reports a rate other than the one it was asked for.

    Callers keep converting at the **requested** rate, never at the reported one: the
    polyphase ratio has to be built from a sane number (44099 -> 16000 would mean 16000
    phases), so a hardware rate that differs by a hair is only a slow drift in the audio --
    invisible unless step 4 says so.

    `opening` leads the info line ("use input device"), `subject` names the boundary in the
    warning ("recording"). They are two arguments rather than one because the two sentences
    were worded per boundary before this helper existed, and their wording is what an
    operator greps a log for. `pipeline_rate` is the fixed rate the boundary converts to,
    and is appended to the info line together with whether that means a conversion; the
    playback boundaries pass None because their other side arrives with the audio.

    DeviceRateUnresolvedError from step 1 escapes unhandled: it is a config problem no
    retry can fix (ADR-0071), and every caller deliberately keeps it out of its device
    -fault retry path.
    """
    rate, how = resolve_device_rate(
        device, override, input=input, config_key=config_key
    )
    if pipeline_rate is None:
        logger.info(
            "%s %s: %s @%dHz (%s)", opening, device.index, device.name, rate, how
        )
    else:
        logger.info(
            "%s %s: %s @%dHz (%s) -> %dHz (%s)",
            opening,
            device.index,
            device.name,
            rate,
            how,
            pipeline_rate,
            "プロセス内で変換" if rate != pipeline_rate else "変換なし",
        )
    stream = open_stream(rate)
    stream.start()
    reported = float(stream.samplerate)
    if abs(reported - rate) > 0.5:
        logger.warning(
            "%s device reports %.4fHz for a requested %dHz; "
            "converting at the requested rate",
            subject,
            reported,
            rate,
        )
    return stream, rate


def get_sd_dtype(format: SampleFormat) -> str:
    if format == SampleFormat.UINT8:
        return "uint8"
    if format == SampleFormat.INT8:
        return "int8"
    if format == SampleFormat.INT16:
        return "int16"
    if format == SampleFormat.INT24:
        return "int24"
    if format == SampleFormat.FLOAT32:
        return "float32"

    raise ValueError(f"Invalid format: {format}")
