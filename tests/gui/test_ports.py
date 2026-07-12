import pytest

import gui.ports as ports_mod
from gui.ports import allocate_free_port


def test_allocate_skips_claimed_and_busy(monkeypatch):
    busy = {8081}
    monkeypatch.setattr(
        ports_mod, "is_port_free", lambda port, host="127.0.0.1": port not in busy
    )
    # 8080 is claimed, 8081 is OS-busy -> first free is 8082
    assert allocate_free_port(claimed={8080}, base=8080) == 8082


def test_allocate_first_when_all_free(monkeypatch):
    monkeypatch.setattr(ports_mod, "is_port_free", lambda port, host="127.0.0.1": True)
    assert allocate_free_port(claimed=set(), base=8080) == 8080


def test_allocate_raises_when_none(monkeypatch):
    monkeypatch.setattr(ports_mod, "is_port_free", lambda port, host="127.0.0.1": False)
    with pytest.raises(RuntimeError):
        allocate_free_port(claimed=set(), base=8080, limit=8082)
