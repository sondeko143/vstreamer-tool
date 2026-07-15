import sounddevice as sd
from pydantic import AliasChoices
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from vspeech.config import PlaybackConfig
from vspeech.config import RecordingConfig
from vspeech.config import SampleFormat
from vspeech.exceptions import DeviceNotFoundError


class HostAPIInfo(BaseModel):
    index: int
    name: str


class DeviceInfo(BaseModel):
    # sounddevice の device dict は snake_case。host api の key だけ `hostapi` なので別名で拾う。
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


def get_device_name(index: int) -> str:
    return DeviceInfo.model_validate(dict(sd.query_devices(index))).name


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


def resolve_input_device(config: RecordingConfig) -> DeviceInfo:
    """録音入力デバイスを解決する。見つからなければ DeviceNotFoundError。

    preflight と recording worker が同じ経路を通る (ADR-0038)。index 指定が
    あれば優先し、無ければ host_api / name で検索する。
    """
    if config.input_device_index is not None:
        try:
            return get_device_info(config.input_device_index)
        except Exception as e:
            raise DeviceNotFoundError(
                f"recording.input_device_index={config.input_device_index} "
                f"が無効です: {e}"
            ) from e
    device = search_device(
        host_api_type=config.input_host_api_name,
        name=config.input_device_name,
        input=True,
    )
    if device is None:
        raise DeviceNotFoundError(
            "入力デバイスが見つかりません "
            f"(recording.input_host_api_name={config.input_host_api_name!r}, "
            f"recording.input_device_name={config.input_device_name!r})"
        )
    return device


def resolve_output_device(config: PlaybackConfig) -> DeviceInfo:
    """再生出力デバイスを解決する。見つからなければ DeviceNotFoundError。"""
    if config.output_device_index is not None:
        try:
            return get_device_info(config.output_device_index)
        except Exception as e:
            raise DeviceNotFoundError(
                f"playback.output_device_index={config.output_device_index} "
                f"が無効です: {e}"
            ) from e
    device = search_device(
        host_api_type=config.output_host_api_name,
        name=config.output_device_name,
        output=True,
    )
    if device is None:
        raise DeviceNotFoundError(
            "出力デバイスが見つかりません "
            f"(playback.output_host_api_name={config.output_host_api_name!r}, "
            f"playback.output_device_name={config.output_device_name!r})"
        )
    return device


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
