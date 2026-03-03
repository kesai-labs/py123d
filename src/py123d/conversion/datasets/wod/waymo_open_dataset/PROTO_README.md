# Imported Waymo Open Dataset Protos

Imported from [waymo-open-dataset](https://github.com/waymo-research/waymo-open-dataset) **v1.6.7**.

## What's imported

- **8 proto files** in `protos/` — flattened from their original directory structure, import paths rewritten to be flat.
- **3 Python utility files** in `utils/` — `frame_utils.py`, `range_image_utils.py`, `transform_utils.py` with import paths rewritten.

## Recompiling protos

If you need to update the generated `_pb2.py` files (e.g. after editing a `.proto` file):

```bash
# Prerequisites
pip install grpcio-tools

# Compile
cd src/py123d/conversion/datasets/wod/waymo_open_dataset/protos/
bash compile_protos.sh
```

The generated `_pb2.py` files are committed so end users don't need `grpcio-tools`.

## Updating to a new Waymo version

1. Clone the new waymo-open-dataset release
2. Copy the 8 proto files, flatten imports (remove directory prefixes)
3. Copy the 3 utility Python files, rewrite imports
4. Run `compile_protos.sh`
5. Test with `python scripts/validate_wod_protos.py`

## License

Apache License 2.0 — see `LICENSE` in this directory.
