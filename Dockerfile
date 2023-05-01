# Python image to use.
FROM python:3.10-slim AS builder
ENV PYTHONUNBUFFERED True
WORKDIR /app
RUN apt-get update && apt-get install -y gcc g++
RUN pip install --trusted-host pypi.python.org poetry
RUN poetry config virtualenvs.in-project true
COPY pyproject.toml poetry.lock .
RUN poetry install --only main --no-root

FROM python:3.10-slim
ENV PYTHONUNBUFFERED True
WORKDIR /app
COPY --from=builder /app/.venv ./.venv
COPY vspeech ./vspeech
COPY voicevox_core/open_jtalk_dic_utf_8-1.11 ./voicevox_core/open_jtalk_dic_utf_8-1.11
COPY docker/ ./
ENTRYPOINT ["/app/.venv/bin/python", "-m", "vspeech"]
