import vspeech.lib.gcp as gcp_mod
from vspeech.config import GcpConfig


def test_id_token_credentials_skips_metadata_when_ce_disabled(monkeypatch):
    """Default config (no service account, use_ce_credentials=False) must not
    probe the GCE metadata server. That probe blocks the sender worker for
    several seconds on non-GCE hosts when metadata.google.internal can't
    resolve."""

    def _explode(*_args, **_kwargs):
        raise AssertionError("CeIdTokenCredentials should not be constructed")

    monkeypatch.setattr(gcp_mod, "CeIdTokenCredentials", _explode)

    config = GcpConfig()  # defaults: no SA file/info, use_ce_credentials=False
    assert config.use_ce_credentials is False
    assert gcp_mod.get_id_token_credentials(config) is None


def test_id_token_credentials_uses_ce_when_enabled(monkeypatch):
    """When the user opts in via use_ce_credentials=True, CE credentials are
    constructed as before."""
    sentinel = object()
    calls = []

    def _fake_ce(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(gcp_mod, "CeIdTokenCredentials", _fake_ce)

    config = GcpConfig(use_ce_credentials=True)
    assert gcp_mod.get_id_token_credentials(config) is sentinel
    assert len(calls) == 1
