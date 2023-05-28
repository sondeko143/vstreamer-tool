ARTIFACTS=

ARTIFACTS += requirements-pod.txt

PYTHON ?= python3

.PHONY: all
all: ${ARTIFACTS}

requirements-pod.txt: poetry.lock
	${PYTHON} -m poetry export -E voicevox --without-hashes > requirements-pod.txt

.PHONY: clean
clean:
	rm -f ${ARTIFACTS}
