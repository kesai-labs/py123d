#!/bin/bash
# Compile all proto files in this directory (modules and .pyi stubs) and fix imports to be relative.
# The modules are generated with protoc 6.31.1 (grpcio-tools 1.74.0) to match the `waymo` extra.
# Prerequisites: pip install grpcio-tools
set -euo pipefail
cd "$(dirname "$0")"

python -m grpc_tools.protoc --proto_path=. --python_out=. --pyi_out=. *.proto

# Fix bare imports → relative imports in generated _pb2.py files
for f in *_pb2.py *_pb2.pyi; do
    sed -i 's/^import \([a-z_]*_pb2\)/from . import \1/' "$f"
done

echo "Done. Generated files:"
ls -1 *_pb2.py *_pb2.pyi
