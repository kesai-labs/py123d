"""Writes the kedec format-1 fixture files of test_kedec_v1_fixtures.py (run once; the files are committed).

Each fixture is a blob (.kedec) and the decoded arrays at the time the format was frozen (.npz). The test checks that
every later decoder reads the same values from the same bytes. Do not regenerate the files for a format change:
add a new format version and new fixtures instead.

Usage: python tests/unit/common/io/lidar/kedec/fixtures/make_kedec_v1_fixtures.py
"""

from pathlib import Path

import numpy as np

from py123d.common.io.lidar.kedec_lidar_io import encode_point_cloud_as_kedec_binary, load_point_cloud_from_kedec_binary
from py123d.common.io.lidar.point_cloud_codec_config import PointCloudCodecConfig

HERE = Path(__file__).parent


def _scan(rng, pose, n_beams, n_columns, beam_and_column):
    import sys

    sys.path.insert(0, str(HERE.parent))
    from synthetic_scans import spinning_scan

    points, features = spinning_scan(rng, pose, n_beams=n_beams, n_columns=n_columns)
    if not beam_and_column:
        del features["column"]
    return points, features


def main():
    rng = np.random.default_rng(2026_11_02)
    pose_a = [1.126, -0.0008, 1.897, 0.99991, -0.0007, -0.0038, 0.0124]
    pose_b = [3.4, 0.6, 0.55, 0.92388, 0.0, 0.0, 0.38268]
    cases = {}
    points, features = _scan(rng, pose_a, 16, 64, beam_and_column=False)
    features.pop("channel")
    cases["spherical"] = (points, features, {2: pose_a}, PointCloudCodecConfig())
    points, features = _scan(rng, pose_a, 16, 64, beam_and_column=False)
    cases["spherical_ring_pose"] = (points, features, {2: pose_a}, PointCloudCodecConfig(kedec_frame="pose"))
    points, features = _scan(rng, pose_b, 16, 64, beam_and_column=True)
    cases["grid"] = (points, features, {5: pose_b}, PointCloudCodecConfig())
    clouds = []
    for sensor_id, pose in ((2, pose_a), (5, pose_b)):
        points, features = _scan(rng, pose, 8, 64, beam_and_column=True)
        # sensor 2 stores no valid column index (-1), so the blob contains a spherical and a grid section
        features["column"] = (
            features["column"].astype(np.int32) if sensor_id == 5 else np.full(len(points), -1, np.int32)
        )
        features["ids"] = np.full(len(points), sensor_id, np.uint8)
        clouds.append((points, features))
    points = np.concatenate([c[0] for c in clouds])
    features = {name: np.concatenate([c[1][name] for c in clouds]) for name in clouds[0][1]}
    features["uid"] = np.arange(len(points))
    features["nan"] = np.where(np.arange(len(points)) % 7 == 0, np.nan, 0.5).astype(np.float32)
    points[11] = np.nan
    cases["merged"] = (points, features, {2: pose_a, 5: pose_b}, PointCloudCodecConfig())
    for name, (points, features, poses, config) in cases.items():
        blob = encode_point_cloud_as_kedec_binary(points, features, config=config, sensor_poses=poses)
        decoded, decoded_features = load_point_cloud_from_kedec_binary(blob)
        (HERE / f"kedec_v1_{name}.kedec").write_bytes(blob)
        np.savez_compressed(HERE / f"kedec_v1_{name}.npz", xyz=decoded, **decoded_features)
        print(name, len(points), len(blob))


if __name__ == "__main__":
    main()
