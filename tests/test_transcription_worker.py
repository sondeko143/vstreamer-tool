from asyncio import Queue
from asyncio import wait_for
from uuid import uuid4

from pytest_httpx import HTTPXMock

from vspeech.config import AmiConfig
from vspeech.config import EventType
from vspeech.config import SampleFormat
from vspeech.config import TranscriptionConfig
from vspeech.shared_context import EventAddress
from vspeech.shared_context import SoundInput
from vspeech.shared_context import WorkerInput
from vspeech.worker.transcription import transcript_worker_ami


async def put_queue(queue: Queue[WorkerInput]):
    await queue.put(
        WorkerInput(
            input_id=uuid4(),
            current_event=EventAddress(EventType.transcription),
            following_events=[],
            text="",
            sound=SoundInput(
                data=b"00000", rate=16000, format=SampleFormat.INT16, channels=1
            ),
            file_path="",
            filters=[],
        )
    )


async def test_ami(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        json={
            "results": [
                {
                    "tokens": [
                        {
                            "written": "%えーと%",
                            "confidence": 0.99,
                            "starttime": 1120,
                            "endtime": 1392,
                            "spoken": "えーと",
                        },
                        {
                            "written": "%うーん%",
                            "confidence": 0.78,
                            "starttime": 1648,
                            "endtime": 1872,
                            "spoken": "うーん",
                        },
                        {
                            "written": "ちょっと",
                            "confidence": 0.15,
                            "starttime": 1872,
                            "endtime": 2048,
                            "spoken": "ちょっと",
                        },
                    ],
                    "confidence": 0.89,
                    "starttime": 800,
                    "endtime": 2992,
                    "tags": [],
                    "rulename": "",
                    "text": "%えーと%%うーん%ちょっと",
                }
            ],
            "utteranceid": "yyyymmdd/mm/nnnnnnnnnnnnnnnnnnnnnnnn_yyyymmdd_hhmmss[nolog]",
            "text": "%えーと%%うーん%ちょっと",
            "code": "",
            "message": "",
        }
    )
    config = TranscriptionConfig()
    ami_config = AmiConfig(appkey="", engine_uri="https://dummy")
    queue = Queue[WorkerInput]()
    await put_queue(queue)
    try:
        output = await wait_for(
            anext(
                transcript_worker_ami(
                    config=config, ami_config=ami_config, in_queue=queue
                )
            ),
            10,
        )
        assert output.text == "えーとうーんちょっと"
    except TimeoutError:
        assert False
