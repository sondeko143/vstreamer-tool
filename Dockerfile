# requirements-pod.txt は requires-python = ">=3.12,<3.13" で解決した uv.lock から
# 生成されるので、イメージもその範囲に収まっていなければならない。
FROM python:3.14-slim AS builder
ENV PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update && apt-get install -y gcc g++ curl
RUN python -m venv .venv
COPY requirements-pod.txt .
RUN .venv/bin/python -m pip install --trusted-host pypi.python.org -r requirements-pod.txt
# voicevox-core 0.16 の wheel は ONNX Runtime・OpenJTalk 辞書・.vvm を同梱しないので
# ダウンローダで取る。出力先 ./voicevox は VoicevoxConfig の既定パス
# (./voicevox/dict/open_jtalk_dic_utf_8-1.11 と ./voicevox/models/vvms) に合わせてある。
RUN curl -vsSfL https://github.com/VOICEVOX/voicevox_core/releases/download/0.16.4/download-linux-x64 -o download
RUN chmod +x download
# `yes y` は必須。onnxruntime と models はそれぞれ利用規約への同意を対話で求めるので、
# 素の `./download` は非対話ビルドで "unexpected end of file" になって落ちる。
RUN yes y | ./download -o voicevox --exclude c-api --devices cpu

FROM python:3.14-slim
ENV PYTHONUNBUFFERED=1
# Onnxruntime.load_once() は libvoicevox_onnxruntime.so.<version> を dlopen する。
# バージョン付きの名前を焼き込まずに済むよう、検索パスだけを通す。
ENV LD_LIBRARY_PATH=/app/voicevox/onnxruntime/lib
WORKDIR /app
COPY --from=builder /app/.venv ./.venv
COPY --from=builder /app/voicevox ./voicevox
COPY vspeech ./vspeech
ENTRYPOINT ["/app/.venv/bin/python", "-m", "vspeech"]
