ARTIFACTS=

ARTIFACTS += requirements-pod.txt

.PHONY: all
all: ${ARTIFACTS}

requirements-pod.txt: uv.lock pyproject.toml
	uv export --no-default-groups --extra voicevox --no-hashes --no-emit-project -o requirements-pod.txt

.PHONY: clean
clean:
	rm -f ${ARTIFACTS}
