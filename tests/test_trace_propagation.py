from uuid import uuid4

from vspeech.config import EventType
from vspeech.config import SampleFormat
from vspeech.shared_context import EventAddress
from vspeech.shared_context import SoundOutput
from vspeech.shared_context import WorkerInput
from vspeech.shared_context import WorkerOutput


def _output_with_trace():
    out = WorkerOutput(
        input_id=uuid4(),
        followings=[[EventAddress(event=EventType.vc, remote="//r")]],
        sound=SoundOutput(
            data=b"abc", rate=16000, format=SampleFormat.INT16, channels=1
        ),
        text="hi",
    )
    out.trace_id = "trace-xyz"
    out.origin_ts = 1234.5
    return out


def test_to_pb_encodes_trace():
    cmd = _output_with_trace().to_pb("//r")
    assert cmd.operand.trace_id == "trace-xyz"
    assert cmd.operand.origin_ts == 1234.5


def test_from_command_decodes_trace():
    cmd = _output_with_trace().to_pb("//r")
    inputs = WorkerInput.from_command(cmd)
    assert inputs[0].trace_id == "trace-xyz"
    assert inputs[0].origin_ts == 1234.5


def test_from_input_copies_trace():
    out = _output_with_trace()
    inputs = WorkerInput.from_command(out.to_pb("//r"))
    copied = WorkerOutput.from_input(inputs[0])
    assert copied.trace_id == "trace-xyz"
    assert copied.origin_ts == 1234.5


def test_absent_trace_defaults():
    out = WorkerOutput(
        input_id=uuid4(),
        followings=[[EventAddress(event=EventType.subtitle, remote="//r")]],
        text="t",
    )
    inputs = WorkerInput.from_command(out.to_pb("//r"))
    assert inputs[0].trace_id == ""
    assert inputs[0].origin_ts == 0.0
