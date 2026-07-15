import pytest
from pydantic import ValidationError

from vspeech.config import RecordingConfig


def test_recording_positive_bounds_reject_zero():
    with pytest.raises(ValidationError):
        RecordingConfig(rate=0)
    with pytest.raises(ValidationError):
        RecordingConfig(channels=0)
    with pytest.raises(ValidationError):
        RecordingConfig(chunk=0)
    # 既定は許容
    RecordingConfig()
