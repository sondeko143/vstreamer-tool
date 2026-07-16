from typing import Any

import torch


def get_device(gpu_id: int | None, gpu_name: str) -> tuple[torch.device, str]:
    # `gpu_id is not None` であって `gpu_id` ではない。0 は正当なデバイス番号であり、
    # 「未設定」を表すのは None（config.py の `gpu_id: int | None = None`）。
    # 両方設定時は gpu_id が gpu_name に優先する（先に判定するため）。意図的だが
    # 未テスト——実運用はどちらか一方しか設定せず実害なし。
    if gpu_id is not None and torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(gpu_id)
        dev = torch.device("cuda", gpu_id)
        return dev, prop.name
    if gpu_name and torch.cuda.is_available():
        index, prop = get_device_index_by_name(gpu_name)
        return torch.device("cuda", index), prop.name
    return torch.device("cpu"), "cpu"


def get_device_index_by_name(name_query: str) -> tuple[int, Any]:
    count = torch.cuda.device_count()
    for i in range(count):
        device = torch.cuda.get_device_properties(i)
        if name_query in device.name:
            return i, device
    return 0, None
