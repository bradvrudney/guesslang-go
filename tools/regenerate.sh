#!/bin/sh
# Rebuilds model/model.bin and testdata/reference.json from the original
# guesslang (2.2.1, TensorFlow 2.5.0), run in Docker. Needs Docker and python3.
# This is only needed to refresh the model or the reference answers; neither
# Python nor TensorFlow is needed to build or use the package.
set -e
cd "$(dirname "$0")/.."
docker run --rm -v "$PWD":/work -w /work python:3.9-slim sh -c \
  'pip install -q tensorflow==2.5.0 guesslang==2.2.1 && python tools/extract.py /work && chown -R '"$(id -u):$(id -g)"' /work'
python3 -I tools/pack.py .
