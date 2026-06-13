ARTIFACTS=

ARTIFACTS += requirements-pod.txt

.PHONY: all
all: ${ARTIFACTS}

requirements-pod.txt: uv.lock pyproject.toml
	uv export --no-default-groups --extra voicevox --no-hashes --no-emit-project -o requirements-pod.txt

.PHONY: voicevox-assets
voicevox-assets:
	curl -sSfL https://github.com/VOICEVOX/voicevox_core/releases/download/0.16.4/download-windows-x64.exe -o download-voicevox.exe
	./download-voicevox.exe -o tests/assets/voicevox --exclude c-api --devices cuda
	@echo "デバイスを変える場合は --devices (cpu/cuda/directml) を指定。フラグは ./download-voicevox.exe --help を参照。"

.PHONY: clean
clean:
	rm -f ${ARTIFACTS}
