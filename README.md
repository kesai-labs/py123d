<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://autonomousvision.github.io/py123d/_static/123D_logo_transparent_white.svg" width="500">
    <source media="(prefers-color-scheme: light)" srcset="https://autonomousvision.github.io/py123d/_static/123D_logo_transparent_black.svg" width="500">
    <img alt="Logo" src="https://autonomousvision.github.io/py123d/_static/123D_logo_transparent_black.svg" width="500">
  </picture>
  <h2 align="center">123D: A Unified Library for Multi-Modal Autonomous Driving Data</h1>
  <h3 align="center"><a href="https://youtu.be/Q4q29fpXnx8">Video</a> | <a href="https://autonomousvision.github.io/py123d/">Documentation</a>
</h1>


## Features

- Unified API for driving data, including sensor data, maps, and labels.
- Support for multiple sensors storage formats.
- Fast dataformat based on [Apache Arrow](https://arrow.apache.org/).
- Visualization tools with [matplotlib](https://matplotlib.org/) and [Viser](https://viser.studio/main/).

## Viewer

<p align="center">
  <img src="assets/viser.png" alt="Viser 3D Viewer" width="800" style="border-radius: 12px; box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);">
</p>


## Changelog

- **`[2026-02-09]`** v0.0.9
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

- **`[2025-11-21]`** v0.0.8 (silent release)
  - Release of package and documentation.
  - Demo data for tutorials.
