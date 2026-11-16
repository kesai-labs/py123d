# Installation

## Pip-Install
You can simply install `py123d` for Python versions >=3.9 via PyPI with
```bash
pip install py123d
```
or as editable pip package with
```bash
mkdir -p $HOME/py123d_workspace; cd $HOME/py123d_workspace # Optional
git clone git@github.com:kesai-labs/py123d.git
cd py123d
pip install -e .
```

## File Structure & Storage
The 123D library converts driving datasets to a unified format. By default, all data is stored in directory of the environment variable `$PY123D_DATA_ROOT`.
For example, you can use.

```bash
export PY123D_DATA_ROOT="$HOME/py123d_workspace/data"
```
which can be added to your `~/.bashrc` or to your bash scripts. Optionally, you can adjust all dataset paths in the hydra config: `py123d/script/config/common/default_dataset_paths.yaml`.

Streaming conversions (`dataset=<name>-stream`) download the source data into `$PY123D_DATA_ROOT/.download/<name>` and delete that folder when the conversion ends (and `.download` itself once it is empty); pass `keep_downloads=true` to keep it. The system temp directory is not used. To download onto another disk, set
```bash
export PY123D_DOWNLOAD_ROOT="/path/to/scratch"
```
Each streaming config has its own folder, so conversions of different datasets can run in parallel. Two conversions of the *same* streaming config share one folder and must not run at the same time, unless each gets its own download root (e.g. `export PY123D_DOWNLOAD_ROOT="$PY123D_DATA_ROOT/.download/$SLURM_JOB_ID"`).

The 123D conversion includes:
- **Logs:** Each log is a directory containing per-modality `.arrow` files (e.g. ego state, cameras, lidar, bounding boxes, traffic lights) and a `sync.arrow` synchronization table. Each modality is stored at its native capture rate. When using MP4 camera compression, the `.mp4` files are also stored in the log directory.
- **Maps:** The maps are static and store our unified HD-Map API. Maps can either be defined per-log (e.g. in AV2, Waymo) or globally for a certain location (e.g. nuPlan, nuScenes, CARLA). We use `.arrow` files to store maps.
- **Sensors:** Camera and lidar data can either be (1) read from the original dataset via relative paths stored in the `.arrow` files or (2) embedded as binary data in the `.arrow` files. Cameras additionally support (3) MP4 compression, stored alongside the `.arrow` files in the log directory.

For example, when converting `nuplan-mini` with MP4 compression and using `PY123D_DATA_ROOT="$HOME/py123d_workspace/data"`, the file structure would look the following way:
```
~/py123d_workspace/
├── data/
│   ├── logs/
│   │   ├── nuplan-mini_test/
│   │   │   ├── 2021.05.25.14.16.10_veh-35_01690_02183/
│   │   │   │   ├── sync.arrow
│   │   │   │   ├── ego_state_se3.arrow
│   │   │   │   ├── box_detections_se3.arrow
│   │   │   │   ├── traffic_light_detections.arrow
│   │   │   │   ├── camera.pcam_f0.arrow
│   │   │   │   ├── ...
│   │   │   │   ├── camera.pcam_b0.arrow
│   │   │   │   ├── camera.pcam_f0.mp4       (optional, if MP4 compression)
│   │   │   │   ├── ...
│   │   │   │   └── lidar.lidar_merged.arrow
│   │   │   └── ...
│   │   ├── nuplan-mini_train/
│   │   │   └── ...
│   │   └── ...
│   └── maps/
│       ├── nuplan/
│       │   ├── nuplan_sg-one-north.arrow
│       │   ├── ...
│       │   └── nuplan_us-pa-pittsburgh-hazelwood.arrow
│       └── ...
└── py123d/ (repository)
    └── ...
```

During a conversion, logs are first written to `$PY123D_DATA_ROOT/logs/.incomplete` (maps to `$PY123D_DATA_ROOT/maps/.incomplete`) and only moved to the structure above once they are complete, so a conversion that fails or is interrupted does not leave half-written logs behind. At the end, the conversion reports whether everything completed and lists all maps and logs that did not. Maps and logs that did not complete are still closed, so that what was written can be inspected, and stay in `.incomplete` until they are converted again. Pass `keep_incomplete=false` to delete them when the conversion ends.
