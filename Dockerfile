# Python image to use.
FROM python:3.10-slim AS builder
ENV PYTHONUNBUFFERED True
WORKDIR /app
RUN apt-get update && apt-get install -y gcc g++ curl
RUN python -m venv .venv
COPY requirements-pod.txt .
RUN .venv/bin/python -m pip install --trusted-host pypi.python.org -r requirements-pod.txt
RUN curl -vsSfL https://github.com/VOICEVOX/voicevox_core/releases/download/0.14.3/download-linux-x64 -o download
RUN chmod +x download
RUN ./download

FROM python:3.10-slim
ENV PYTHONUNBUFFERED True
WORKDIR /app
COPY --from=builder /app/.venv ./.venv
COPY --from=builder /app/voicevox_core ./voicevox_core
RUN cp voicevox_core/*.so.* ./ 
COPY vspeech ./vspeech
ENTRYPOINT ["/app/.venv/bin/python", "-m", "vspeech"]
