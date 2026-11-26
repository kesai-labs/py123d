"""Benchmarks of the kedec codecs on converted logs and video samples.

    python -m py123d.script.run_kedec_benchmark lidar LOG_DIR [--every 100] [--json FILE]
    python -m py123d.script.run_kedec_benchmark video NAME=SOURCE [NAME=SOURCE ...] [--frames 50] [--stride 1]
        [--skip 0] [--svt-presets 6 8 10] [--svt-crfs 26 30 34 38 42 46] [--json FILE]

lidar: every N-th scan of every lidar file of a 123D log (any lidar codec) is coded with kedec: the spherical mode
around the sensor position (the default) and in the full sensor frame, and the grid mode where the scan stores a
beam and a column index. Per lidar and mode: bytes per point, share of the stored blobs, bytes per part (range,
angles, time stamps, intensity, ...), largest and mean position error, encoding and decoding time per scan (one
thread).

video: per sample, N images at 10 per second are coded with NVENC (cq 38, the kedec setting) and with SVT-AV1 at
each preset and CRF. Per encoding: bytes per image, PSNR of the luma against the source planes, milliseconds per
image (wall time of the encoder, SVT with all cores), peak memory. Per SVT preset, the CRF whose file size equals
NVENC's (interpolated in log size between the measured CRFs) and the PSNR difference at that point.

A SOURCE is a camera file of a 123D log with JPEG rows (every second image of a 20 per second camera is taken when
--stride 2 is given), a video file (decoded with FFmpeg; --stride takes every n-th image), or a folder of JPEG files.
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa
from pyarrow import ipc

from py123d.common.io.camera.av1_camera_io import AV1Reader, AV1Writer, nvenc_av1_available
from py123d.common.io.camera.video_codec_config import VideoCodecConfig
from py123d.common.io.camera.yuv_frame import YUV420Frame
from py123d.common.io.lidar.kedec_lidar_io import (
    encode_point_cloud_as_kedec_binary,
    load_point_cloud_from_kedec_binary,
    read_kedec_head,
)
from py123d.common.io.lidar.point_cloud_codec_config import PointCloudCodecConfig
from py123d.common.io.lidar.point_cloud_codec_registry import decode_point_cloud

# ------------------------------------------------------------------------------------------------------------------
# Lidar
# ------------------------------------------------------------------------------------------------------------------


def _schema_pose(path: Path) -> List[float]:
    import msgpack

    metadata = msgpack.unpackb(ipc.open_file(pa.memory_map(str(path))).schema.metadata[b"metadata"], raw=False)
    pose = metadata.get("lidar_to_imu_se3")
    return list(pose) if pose is not None else [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]


def _errors(
    xyz: np.ndarray, features: Optional[Dict[str, np.ndarray]], config: PointCloudCodecConfig, pose
) -> np.ndarray:
    """Position error of every point: the scan is coded once more with an index feature that matches the decoded
    points to the input points (the size and time are measured without it)."""
    indexed = dict(features or {})
    indexed["benchmark_index"] = np.arange(len(xyz), dtype=np.int64)
    decoded, decoded_features = load_point_cloud_from_kedec_binary(
        encode_point_cloud_as_kedec_binary(xyz, indexed, config, sensor_poses={0: pose})
    )
    index = decoded_features["benchmark_index"]
    return np.linalg.norm(decoded.astype(np.float64) - xyz[index].astype(np.float64), axis=1)


def benchmark_lidar(log_dir: Path, every: int) -> Dict:
    results: Dict = {}
    for path in sorted(log_dir.glob("lidar.*.arrow")):
        key = path.stem
        pose = _schema_pose(path)
        data = ipc.open_file(pa.memory_map(str(path))).read_all().column(f"{key}.data")
        configs = {
            "spherical_position": PointCloudCodecConfig(kedec_mode="spherical", kedec_frame="position"),
            "spherical_pose": PointCloudCodecConfig(kedec_mode="spherical", kedec_frame="pose"),
            "auto": PointCloudCodecConfig(kedec_mode="auto"),
        }
        totals = {
            name: {"bytes": 0, "parts": {}, "encode_s": 0.0, "decode_s": 0.0, "max_m": 0.0, "sum_m": 0.0}
            for name in configs
        }
        source_bytes = points = scans = 0
        for row in range(0, len(data), every):
            blob = data[row].as_py()
            if blob is None:
                continue
            xyz, features = decode_point_cloud(blob)
            source_bytes += len(blob)
            points += len(xyz)
            scans += 1
            for name, config in configs.items():
                started = time.perf_counter()
                coded = encode_point_cloud_as_kedec_binary(xyz, features, config, sensor_poses={0: pose})
                encoded = time.perf_counter()
                load_point_cloud_from_kedec_binary(coded)
                total = totals[name]
                total["encode_s"] += encoded - started
                total["decode_s"] += time.perf_counter() - encoded
                total["bytes"] += len(coded)
                head, _ = read_kedec_head(coded)
                for section in head["sections"]:
                    total.setdefault("mode", section["mode"])
                    for part in section["parts"]:
                        total["parts"][part[0]] = total["parts"].get(part[0], 0) + int(part[2])
                errors = _errors(xyz, features, config, pose)
                total["max_m"] = max(total["max_m"], float(errors.max()) if len(errors) else 0.0)
                total["sum_m"] += float(errors.sum())
        results[key] = {
            "scans": scans,
            "points_per_scan": round(points / max(scans, 1)),
            "source_bytes_per_point": round(source_bytes / max(points, 1), 3),
            "modes": {
                name: {
                    "mode": total.get("mode"),
                    "bytes_per_point": round(total["bytes"] / max(points, 1), 3),
                    "percent_of_source": round(100 * total["bytes"] / max(source_bytes, 1), 2),
                    "bytes_per_point_by_part": {
                        k: round(v / max(points, 1), 3) for k, v in sorted(total["parts"].items())
                    },
                    "encode_ms_per_scan": round(1000 * total["encode_s"] / max(scans, 1), 1),
                    "decode_ms_per_scan": round(1000 * total["decode_s"] / max(scans, 1), 1),
                    "largest_error_mm": round(1000 * total["max_m"], 2),
                    "mean_error_mm": round(1000 * total["sum_m"] / max(points, 1), 2),
                }
                for name, total in totals.items()
            },
        }
        print(key, json.dumps(results[key]["modes"]["spherical_position"]), flush=True)
    return results


# ------------------------------------------------------------------------------------------------------------------
# Video
# ------------------------------------------------------------------------------------------------------------------


def load_frames(source: str, count: int, stride: int, skip: int) -> List[YUV420Frame]:
    """YUV frames of a sample: a 123D camera file with JPEG rows, a video file or a folder of JPEG files."""
    path = Path(source)
    frames: List[YUV420Frame] = []
    if path.is_dir():
        for jpg in sorted(path.glob("*.jpg"))[skip::stride][:count]:
            frames.append(YUV420Frame.from_jpeg(jpg.read_bytes()))
    elif path.suffix == ".arrow":
        table = ipc.open_file(pa.memory_map(str(path))).read_all()
        data = table.column(f"{path.stem}.data")
        for row in range(skip, len(data), stride)[:count]:
            frames.append(YUV420Frame.from_jpeg(data[row].as_py()))
    else:
        import av

        with av.open(str(path)) as container:
            for index, frame in enumerate(container.decode(video=0)):
                if index < skip or (index - skip) % stride:
                    continue
                frames.append(YUV420Frame.from_av_frame(frame))
                if len(frames) >= count:
                    break
    if not frames:
        raise ValueError(f"no frames in {source}")
    return frames


def _psnr_y(path: Path, frames: List[YUV420Frame]) -> float:
    reader = AV1Reader(path)
    mse = [
        float(np.mean((reader.get_yuv(i).y.astype(np.float64) - f.y.astype(np.float64)) ** 2))
        for i, f in enumerate(frames)
    ]
    reader.close()
    return float(10 * np.log10(255.0**2 / max(np.mean(mse), 1e-12)))


def _gpu_memory_mb() -> Optional[int]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        return int(out.stdout.split()[0])
    except Exception:  # noqa: BLE001
        return None


def encode(frames: List[YUV420Frame], config: VideoCodecConfig, workdir: Path) -> Dict:
    path = workdir / "sample.av1.mp4"
    path.unlink(missing_ok=True)
    gpu_before = _gpu_memory_mb() if config.backend == "nvenc" else None
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    started = time.perf_counter()
    writer = AV1Writer(path, config)
    gpu_peak = None
    for k, frame in enumerate(frames):
        writer.write_frame(frame)
        if k == 5 and gpu_before is not None:
            gpu_peak = _gpu_memory_mb()
    writer.close()
    seconds = time.perf_counter() - started
    size = path.stat().st_size
    return {
        "bytes_per_image": round(size / len(frames)),
        "psnr_y_db": round(_psnr_y(path, frames), 2),
        "ms_per_image": round(1000 * seconds / len(frames), 1),
        "peak_rss_growth_mb": round((resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - rss_before) / 1024),
        "gpu_memory_mb": None if gpu_peak is None or gpu_before is None else gpu_peak - gpu_before,
    }


def matched_crf(nvenc_bytes: float, svt: Dict[int, Dict]) -> Optional[Tuple[float, float]]:
    """(CRF, PSNR-Y) at which SVT's size equals NVENC's, interpolated in log size; None outside the measured range."""
    crfs = sorted(svt)
    sizes = [np.log(svt[c]["bytes_per_image"]) for c in crfs]
    target = np.log(nvenc_bytes)
    for (c0, s0), (c1, s1) in zip(zip(crfs, sizes), zip(crfs[1:], sizes[1:])):
        if min(s0, s1) <= target <= max(s0, s1) and s0 != s1:
            w = (target - s0) / (s1 - s0)
            return c0 + w * (c1 - c0), svt[c0]["psnr_y_db"] + w * (svt[c1]["psnr_y_db"] - svt[c0]["psnr_y_db"])
    return None


def benchmark_video(
    samples: Dict[str, str], count: int, stride: int, skip: int, presets: List[int], crfs: List[int]
) -> Dict:
    results: Dict = {}
    with tempfile.TemporaryDirectory(dir=".") as tmp:
        workdir = Path(tmp)
        for name, source in samples.items():
            frames = load_frames(source, count, stride, skip)
            entry: Dict = {"images": len(frames), "size": [frames[0].width, frames[0].height]}
            if nvenc_av1_available():
                entry["nvenc_cq38"] = encode(frames, VideoCodecConfig(backend="nvenc"), workdir)
            for preset in presets:
                svt = {
                    crf: encode(frames, VideoCodecConfig(backend="svt", svt_preset=preset, svt_crf=crf), workdir)
                    for crf in crfs
                }
                entry[f"svt_p{preset}"] = svt
                if "nvenc_cq38" in entry:
                    match = matched_crf(entry["nvenc_cq38"]["bytes_per_image"], svt)
                    entry[f"svt_p{preset}_matched"] = (
                        None
                        if match is None
                        else {
                            "crf": round(match[0], 1),
                            "psnr_y_db": round(match[1], 2),
                            "psnr_minus_nvenc_db": round(match[1] - entry["nvenc_cq38"]["psnr_y_db"], 2),
                        }
                    )
            results[name] = entry
            print(
                name,
                json.dumps({k: v for k, v in entry.items() if not k.startswith("svt_p") or k.endswith("matched")}),
                flush=True,
            )
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    lidar = sub.add_parser("lidar")
    lidar.add_argument("log_dir", type=Path)
    lidar.add_argument("--every", type=int, default=100)
    lidar.add_argument("--json", type=Path)
    video = sub.add_parser("video")
    video.add_argument("samples", nargs="+", help="NAME=SOURCE")
    video.add_argument("--frames", type=int, default=50)
    video.add_argument("--stride", type=int, default=1)
    video.add_argument("--skip", type=int, default=0)
    video.add_argument("--svt-presets", type=int, nargs="+", default=[6, 8, 10])
    video.add_argument("--svt-crfs", type=int, nargs="+", default=[26, 30, 34, 38, 42, 46])
    video.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    if args.command == "lidar":
        result = benchmark_lidar(args.log_dir, args.every)
    else:
        samples = dict(sample.split("=", 1) for sample in args.samples)
        result = benchmark_video(samples, args.frames, args.stride, args.skip, args.svt_presets, args.svt_crfs)
    if args.json:
        args.json.write_text(json.dumps(result, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
