import io
import json
import wave
from pprint import pformat
from signal import SIGINT, signal
from types import FrameType
from typing import Any, List, Optional, Union

import click
import httpx
import pyaudio  # Soundcard audio I/O access library
import scipy.io.wavfile
from websocket import WebSocket, create_connection

from config import Config
from logger import configure_logger, logger


def recording_chunk(stream: pyaudio.Stream, config: Config):
    # start Recording
    frames: List[bytes] = []

    # Record for RECORD_SECONDS
    for _ in range(0, int(config.rate / config.chunk * config.record_seconds)):
        data = stream.read(config.chunk)
        frames.append(data)
    # Stop Recording
    return frames


def get_amp_max(frames: List[bytes], config: Config, sample_size: int):
    tempwav = io.BytesIO()
    with wave.open(tempwav, "wb") as waveFile:
        waveFile.setnchannels(config.channels)
        waveFile.setsampwidth(sample_size)
        waveFile.setframerate(config.rate)
        waveFile.writeframes(b"".join(frames))
    tempwav.seek(0)
    _, amp_arr = scipy.io.wavfile.read(tempwav)
    MAX_WAV16_AMP = 32767
    amps: List[Any] = []
    for amp in amp_arr:
        amps.append(amp[0] / MAX_WAV16_AMP)
        amps.append(amp[1] / MAX_WAV16_AMP)
    return max(amps)


def write_to_file(
    frames: List[bytes],
    wav_file: Union[io.BytesIO, str],
    config: Config,
    sample_size: int,
):
    with wave.open(wav_file, "wb") as waveFile:
        waveFile.setnchannels(config.channels)
        waveFile.setsampwidth(sample_size)
        waveFile.setframerate(config.rate)
        waveFile.writeframes(b"".join(frames))


def recording_loop(stream: pyaudio.Stream, config: Config, sample_size: int):
    recorded_frames: List[bytes] = []
    total_recording_seconds = 0
    status = "waiting"
    while status != "finished":
        frames = recording_chunk(stream=stream, config=config)
        amp_max = get_amp_max(frames, config=config, sample_size=sample_size)
        silence = amp_max < config.silence_threshold
        if status == "waiting":
            if silence:
                pass
            else:
                recorded_frames = frames
                status = "recording"
        elif status == "recording":
            recorded_frames += frames
            total_recording_seconds += config.record_seconds
            if total_recording_seconds > config.max_recording_seconds:
                status = "finished"
            elif silence:
                status = "finished"
            else:
                pass
        if status != "waiting":
            logger.info(f"{status}: {total_recording_seconds}")
    return recorded_frames


def voice_recognize(recorded_frames: List[bytes], config: Config, sample_size: int):
    tempwav = io.BytesIO()
    write_to_file(
        frames=recorded_frames, wav_file=tempwav, config=config, sample_size=sample_size
    )
    data = {
        "d": config.ami_engine_name,
        "u": config.ami_appkey,
    }
    files = {"a": tempwav}
    r = httpx.post(config.ami_engine_uri, data=data, files=files)
    res_body = r.json()
    logger.info(pformat(res_body))
    return res_body["text"]


def open_connection_to_yukari_net(config: Config):
    return create_connection(f"ws://localhost:{config.yukari_net_port}")


def send_text_to_yukari_net(ws: WebSocket, text: str):
    ws.send(f"0:{text}")


loop_end = False


def handler(signum: int, frame: Optional[FrameType]):
    logger.info("Signal handler called with signal", signum)
    global loop_end
    loop_end = True


@click.command()
@click.option("-c", "--conf", default="./config.json")
def main(conf: str):
    with open(conf, "r") as f:
        config_obj = json.load(f)
        config = Config(**config_obj)
    configure_logger(config)

    # Startup pyaudio instance
    audio = pyaudio.PyAudio()
    ws = open_connection_to_yukari_net(config)
    sample_size = audio.get_sample_size(config.format)

    signal(SIGINT, handler)
    logger.info("Start main loop")
    try:
        while not loop_end:
            stream = audio.open(
                format=config.format,
                channels=config.channels,
                rate=config.rate,
                input=True,
                frames_per_buffer=config.chunk,
            )
            recorded_frames = recording_loop(
                stream=stream, config=config, sample_size=sample_size
            )
            stream.stop_stream()
            stream.close()
            text = voice_recognize(
                recorded_frames=recorded_frames, config=config, sample_size=sample_size
            )
            send_text_to_yukari_net(ws, text)
    finally:
        logger.info("terminate")
        ws.close()
        audio.terminate()


if __name__ == "__main__":
    main()
