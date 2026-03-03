#!/bin/bash
# Compile all proto files in this directory and fix imports to be relative.
# Prerequisites: pip install grpcio-tools
set -euo pipefail
cd "$(dirname "$0")"

python -m grpc_tools.protoc --proto_path=. --python_out=. *.proto

# Fix bare imports → relative imports in generated _pb2.py files
for f in *_pb2.py; do
    sed -i 's/^import \([a-z_]*_pb2\)/from . import \1/' "$f"
done

echo "Done. Generated files:"
ls -1 *_pb2.py
