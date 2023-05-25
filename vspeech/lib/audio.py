from typing import Dict
from typing import Optional

from humps import camelize
from pyaudio import PyAudio
from pyaudio import paFloat32
from pyaudio import paInt8
from pyaudio import paInt16
from pyaudio import paInt24
from pyaudio import paUInt8
from pydantic import BaseModel

from vspeech.config import SampleFormat


class HostAPIInfo(BaseModel):
    index: int
    name: str

    class Config:
        alias_generator = camelize


class DeviceInfo(BaseModel):
    host_api: int
    max_input_channels: float
    max_output_channels: float
    name: str
    index: int

    class Config:
        alias_generator = camelize


def list_all_devices(input: bool = False, output: bool = False):
    p = PyAudio()
    results: Dict[str, int] = {}
    for i in range(p.get_device_count()):
        d = DeviceInfo.parse_obj(p.get_device_info_by_index(i))
        if input and d.max_input_channels <= 0:
            continue
        if output and d.max_output_channels <= 0:
            continue
        h = HostAPIInfo.parse_obj(p.get_host_api_info_by_index(d.host_api))
        results[f"{h.name}: {d.name}"] = d.index
    p.terminate()
    return results


def get_device_name(p: PyAudio, index: int):
    return DeviceInfo.parse_obj(p.get_device_info_by_index(index)).name


def get_device_info(p: PyAudio, index: int):
    return DeviceInfo.parse_obj(p.get_device_info_by_index(index))


def search_host_api_by_type(p: PyAudio, name: str):
    for i in range(p.get_host_api_count()):
        host_api = HostAPIInfo.parse_obj(p.get_host_api_info_by_index(i))
        if host_api.name == name:
            return host_api


def search_device_by_name(
    p: PyAudio,
    name: Optional[str],
    host_api_index: Optional[int],
    input: bool = False,
    output: bool = False,
):
    if not name:
        if input:
            return DeviceInfo.parse_obj(p.get_default_input_device_info())
        elif output:
            return DeviceInfo.parse_obj(p.get_default_output_device_info())
        return None
    for i in range(p.get_device_count()):
        device = DeviceInfo.parse_obj(p.get_device_info_by_index(i))
        if host_api_index and host_api_index != device.host_api:
            continue
        if input and device.max_input_channels <= 0:
            continue
        if output and device.max_output_channels <= 0:
            continue
        if name in device.name:
            return device


def search_device(
    p: PyAudio,
    host_api_type: Optional[str],
    name: Optional[str],
    input: bool = False,
    output: bool = False,
):
    host_api_index = None
    if host_api_type:
        info = search_host_api_by_type(p, host_api_type)
        if info:
            host_api_index = info.index
    return search_device_by_name(p, name, host_api_index, input=input, output=output)


def get_pa_format(format: SampleFormat) -> int:
    if format == SampleFormat.UINT8:
        return paUInt8
    if format == SampleFormat.INT8:
        return paInt8
    if format == SampleFormat.INT16:
        return paInt16
    if format == SampleFormat.INT24:
        return paInt24
    if format == SampleFormat.FLOAT32:
        return paFloat32

    raise ValueError(f"Invalid format: {format}")
