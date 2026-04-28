<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://kesai.eu/py123d/_static/123D_logo_transparent_white.svg" width="500">
    <source media="(prefers-color-scheme: light)" srcset="https://kesai.eu/py123d/_static/123D_logo_transparent_black.svg" width="500">
    <img alt="Logo" src="https://kesai.eu/py123d/_static/123D_logo_transparent_black.svg" width="500">
  </picture>
  <h2 align="center">123D: A Unified Library for Multi-Modal Autonomous Driving Data</h2>
  <h3 align="center"><a href="https://youtu.be/Q4q29fpXnx8">Video</a> | <a href="https://kesai.eu/py123d/">Documentation</a></h3>
</h1>

<p align="center">
  <a href="https://pypi.org/project/py123d/"><img alt="PyPI Version" src="https://img.shields.io/pypi/v/py123d?color=blue"></a>
  <a href="https://pypi.org/project/py123d/"><img alt="Python Versions" src="https://img.shields.io/pypi/pyversions/py123d"></a>
  <a href="https://github.com/kesai-labs/py123d/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-green"></a>
</p>

**One library for autonomous driving datasets.** 123D converts raw data from Argoverse 2, nuScenes, nuPlan, KITTI-360, PandaSet, and Waymo into a fast, unified [Apache Arrow](https://arrow.apache.org/) format, and then gives you a single API to read cameras, lidar, HD maps, and labels across all of them.

## Features

- **Dataset download**: Fetch supported datasets from their official sources via the CLI, and optionally convert directly into the unified format.
- **Hydra-based conversion CLI**: YAML configs to manage your data pipelines.
- **Apache Arrow storage**: columnar, memory-mapped, zero-copy reads. Fast and memory efficient.
- **Multiple sensor codecs**: MP4/JPEG/PNG for cameras; LAZ/Draco/Arrow IPC for lidar.
- **No sensor duplication**: Converted logs reference source camera/lidar files via relative paths, so you don't store sensors twice.
- **Unified API**: Read cameras, lidar, maps, and labels through a single interface, regardless of the source dataset.
- **Built-in visualization**: interactive 3D viewer ([Viser](https://viser.studio/main/)), and [matplotlib](https://matplotlib.org/) plotting.

## Installation

```bash
pip install py123d
```

Per-dataset extras (e.g. `py123d[av2]`, `py123d[nuscenes]`, `py123d[waymo]`) install the parser dependencies for each dataset on demand. See the [Demo](#demo) below for an example.

## Demo

Demo using the Argoverse 2 Sensor dataset, which is publicly readable from S3 and requires no cloud authentication.

The `av2-sensor-stream` config downloads the requested logs/maps into a managed temp directory, converts them into our self-contained Arrow format, and cleans up the source files afterwards. `PY123D_DATA_ROOT` controls where the converted logs/maps are written. The script below installs the AV2 extra, converts the first 3 validation logs (~250 MB each), and launches the Viser viewer:

```bash
# 1. Install
pip install py123d[av2]

export PY123D_DATA_ROOT=/path/to/py123d_data

# 2. Download + Convert
py123d-conversion dataset=av2-sensor-stream \
  dataset.parser.splits='[av2-sensor_val]' \
  dataset.parser.downloader.num_logs=3

# 3. Launch Viewer
py123d-viser scene_filter=av2-sensor
```

Open `http://localhost:8080` to browse the converted scenes interactively.

## Viewer

<p align="center">
  <img src="assets/viser.png" alt="Viser 3D Viewer" width="800">
</p>

## Supported Datasets

| Dataset | Cameras | LiDARs | Map | 3D Boxes | Traffic Lights |
|:--------|:-------:|:------:|:---:|:--------:|:--------------:|
| [Argoverse 2 - Sensor](https://www.argoverse.org/) | 9 | 2 | ✓ | ✓ | ✗ |
| [nuScenes](https://www.nuscenes.org/) | 6 | 1 | ✓ | ✓ | ✗ |
| [nuPlan](https://www.nuscenes.org/nuplan) | 8 | 5 | ✓ | ✓ | ✓ |
| [KITTI-360](https://www.cvlibs.net/datasets/kitti-360/) | 4 | 1 | ✓ | ✓ | ✗ |
| [PandaSet](https://pandaset.org/) | 6 | 2 | ✗ | ✓ | ✗ |
| [Waymo Open - Perception](https://waymo.com/open/) | 5 | 5 | ✓ | ✓ | ✗ |
| [Waymo Open - Motion](https://waymo.com/open/) | ✗ | ✗ | ✓ | ✓ | ✓ |
| [CARLA / LEAD](https://github.com/kesai-labs/lead) | config. | config. | ✓ | ✓ | ✓ |
| [NVIDIA Physical AI AV](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) *(experimental)* | 7 | 1 | ✗ | ✓ | ✗ |


## Changelog

<details open>
<summary><b>v0.3.0</b> (2026-04-28)</summary>

- Refactored dataset download interface, with new download/stream options for nuScenes, PandaSet (HF mirror), AV2-sensor, WOD-perception, WOMD ([#126](https://github.com/kesai-labs/py123d/pull/126)), and nuPlan.
- Added ncore dataset support with parser, downloader, and on-the-fly conversion ([#125](https://github.com/kesai-labs/py123d/pull/125)).
- Waymo Open Motion: remaining WOMD splits, WOMD-specific fields via custom modalities, and skip-logs / skip-map options for storage-constrained runs.
- Map improvements: speed bumps as surface map objects in Waymo ([#130](https://github.com/kesai-labs/py123d/pull/130)); `align_road_edges_to_traffic` for traffic-aligned road edges ([#123](https://github.com/kesai-labs/py123d/pull/123)).
- Parser fixes across PandaSet (lidar/ego poses, extrinsics), nuPlan ([#128](https://github.com/kesai-labs/py123d/pull/128)), KITTI-360 labels, nuScenes map path in stream mode, WOD motion streaming, and `pai-av` / `ncore` labels.
- Runtime and packaging: corrected sync-table entries when inferring dynamics; Ray executor compatibility; `google-cloud-storage` moved to the Waymo extra.

Includes all fixes from v0.2.1 and v0.2.2. No breaking changes to the public API, Arrow schema, or CLI entry points.

</details>

<details>
<summary><b>v0.2.0</b> (2026-04-14)</summary>

- Transferred repository to [KE:SAI]( https://kesai.eu).

- Aligned ego and agent dynamics to a unified global/ego-frame convention, with velocity/acceleration inference in `LogWriter` from poses for `EgoState` and `BoxDetectionsSE3` ([#119](https://github.com/kesai-labs/py123d/pull/119), [#120](https://github.com/kesai-labs/py123d/pull/120)).
- Improved OpenDRIVE maps: 3D road-edge lifting, lane-boundary reconstruction, and cleaner map-metadata location handling ([#121](https://github.com/kesai-labs/py123d/pull/121)).
- Parser and visualization fixes: NuScenes interpolated parser defaults to 10 Hz sync with camera-pose interpolation; PandaSet extrinsic/undistortion fixes; viser fixes ([#117](https://github.com/kesai-labs/py123d/pull/117)); new matplotlib camera-rig and lidar-reprojection utilities.

No breaking changes to the public API, Arrow schema, or CLI entry points.

</details>

<details>
<summary><b>v0.1.0</b> (2026-03-22)</summary>

- Asynchronous (native-rate) data storage: modalities are now written at their original capture rate, not just at the a frame-wise rate.
- New parser architecture with `BaseLogParser.iter_modalities_async` for native-rate iteration alongside the existing synchronized path.
- Added NVIDIA Physical AI AV dataset support (experimental).
- Added standalone OpenDRIVE / CARLA map parser.
- Refactored `conversion/` module into `parser/` with consistent naming across all dataset parsers.
- Refactored Viser 3D viewer. Adds more control and dark mode.
- Added `LaneType`, `IntersectionType`, `StopZoneType` to map data structure.
- Replaced Waymo heavy dependencies with lightweight protobufs.
- Various fixes to camera-to-global transforms across all datasets.

</details>

<details>
<summary><b>v0.0.9</b> (2026-02-09)</summary>

- Added Waymo Open Motion Dataset support.
- Replaced gpkg map implementation with Arrow-based format for improved performance.
- Added sensor names and timestamps to camera and Lidar data across all datasets.
- Added ego-to-camera transforms in static metadata.
- Implemented geometry builders for PoseSE2/PoseSE3 from arbitrary rotation/translation representations.
- Added support for loading merged point clouds in API.
- Improved map querying speed and OpenDrive lane connectivity handling.
- Added recommended conversion options to dataset YAML configuration files.
- Fixed PandaSet static extrinsics and KITTI-360 timestamp handling.
- Fixed memory issues when converting large datasets (e.g., nuPlan).

</details>

<details>
<summary><b>v0.0.8</b> (2025-11-21)</summary>

- Release of package and documentation.
- Demo data for tutorials.

</details>

## Citation

```bibtex
@software{Contributors123D,
  title   = {123D: A Unified Library for Multi-Modal Autonomous Driving Data},
  author  = {123D Contributors},
  year    = {2026},
  url     = {https://github.com/kesai-labs/py123d},
  license = {Apache-2.0}
}
```


## License

123D is released under the [Apache License 2.0](LICENSE).
