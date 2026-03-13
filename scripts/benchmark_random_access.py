"""Benchmark random access latency for different modalities via SceneAPI."""

from __future__ import annotations

import argparse
import random
import statistics
import time
from typing import Callable, List

from py123d.api.scene.arrow.helper import get_filtered_scenes
from py123d.api.scene.scene_api import SceneAPI
from py123d.api.scene.scene_filter import SceneFilter
from py123d.datatypes.sensors.lidar import LidarID


def benchmark_access(
    name: str,
    scene: SceneAPI,
    access_fn: Callable[[int], object],
    iterations: List[int],
    num_samples: int,
) -> dict:
    """Benchmark random access for a single modality."""
    sampled = [random.choice(iterations) for _ in range(num_samples)]

    # Warmup
    access_fn(iterations[0])

    latencies_ms = []
    for it in sampled:
        t0 = time.perf_counter()
        result = access_fn(it)
        t1 = time.perf_counter()
        if result is not None:
            latencies_ms.append((t1 - t0) * 1000)

    if not latencies_ms:
        return {"name": name, "samples": 0}

    return {
        "name": name,
        "samples": len(latencies_ms),
        "mean_ms": statistics.mean(latencies_ms),
        "median_ms": statistics.median(latencies_ms),
        "p95_ms": sorted(latencies_ms)[int(len(latencies_ms) * 0.95)],
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
    }


def benchmark_timestamps(name: str, load_fn: Callable[[], object], num_runs: int = 10) -> dict:
    """Benchmark loading all timestamps for a single modality."""
    # Warmup
    load_fn()

    latencies_ms = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        result = load_fn()
        t1 = time.perf_counter()
        if result is not None:
            latencies_ms.append((t1 - t0) * 1000)

    if not latencies_ms:
        return {"name": name, "samples": 0}

    return {
        "name": name,
        "samples": len(latencies_ms),
        "mean_ms": statistics.mean(latencies_ms),
        "median_ms": statistics.median(latencies_ms),
        "p95_ms": sorted(latencies_ms)[int(len(latencies_ms) * 0.95)],
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
    }


def run_timestamp_benchmark(scene: SceneAPI, num_runs: int = 10) -> List[dict]:
    """Benchmark loading all timestamps for each modality."""
    results = []

    results.append(benchmark_timestamps("ts.iteration", scene.get_all_iteration_timestamps, num_runs))
    results.append(benchmark_timestamps("ts.ego_state_se3", scene.get_all_ego_state_se3_timestamps, num_runs))
    results.append(benchmark_timestamps("ts.box_detections_se3", scene.get_all_box_detections_se3_timestamps, num_runs))
    results.append(
        benchmark_timestamps("ts.traffic_light_detections", scene.get_all_traffic_light_detections_timestamps, num_runs)
    )

    for cam_id in scene.available_pinhole_camera_ids:
        results.append(
            benchmark_timestamps(
                f"ts.pinhole_camera.{cam_id.name}",
                lambda cid=cam_id: scene.get_all_pinhole_camera_timestamps(cid),
                num_runs,
            )
        )

    for cam_id in scene.available_fisheye_mei_camera_ids:
        results.append(
            benchmark_timestamps(
                f"ts.fisheye_mei_camera.{cam_id.name}",
                lambda cid=cam_id: scene.get_all_fisheye_mei_camera_timestamps(cid),
                num_runs,
            )
        )

    for lidar_id in scene.available_lidar_ids + [LidarID.LIDAR_MERGED]:
        results.append(
            benchmark_timestamps(
                f"ts.lidar.{lidar_id.name}",
                lambda lid=lidar_id: scene.get_all_lidar_timestamps(lid),
                num_runs,
            )
        )

    for custom_name in scene.get_all_custom_modality_metadatas():
        results.append(
            benchmark_timestamps(
                f"ts.custom.{custom_name}",
                lambda cn=custom_name: scene.get_all_custom_modality_timestamps(cn),
                num_runs,
            )
        )

    return results


def run_benchmark(scene: SceneAPI, num_samples: int = 100) -> List[dict]:
    """Run benchmarks for all available modalities on a scene."""
    n = scene.number_of_iterations
    iterations = list(range(n))
    results = []

    # Ego state
    results.append(
        benchmark_access(
            "ego_state_se3",
            scene,
            lambda it: scene.get_ego_state_se3_at_iteration(it),
            iterations,
            num_samples,
        )
    )

    # Box detections
    results.append(
        benchmark_access(
            "box_detections_se3",
            scene,
            lambda it: scene.get_box_detections_se3_at_iteration(it),
            iterations,
            num_samples,
        )
    )

    # Traffic light detections
    results.append(
        benchmark_access(
            "traffic_light_detections",
            scene,
            lambda it: scene.get_traffic_light_detections_at_iteration(it),
            iterations,
            num_samples,
        )
    )

    # Pinhole cameras
    for cam_id in scene.available_pinhole_camera_ids:
        results.append(
            benchmark_access(
                f"pinhole_camera.{cam_id.name}",
                scene,
                lambda it, cid=cam_id: scene.get_pinhole_camera_at_iteration(it, cid),
                iterations,
                num_samples,
            )
        )

    # Fisheye MEI cameras
    for cam_id in scene.available_fisheye_mei_camera_ids:
        results.append(
            benchmark_access(
                f"fisheye_mei_camera.{cam_id.name}",
                scene,
                lambda it, cid=cam_id: scene.get_fisheye_mei_camera_at_iteration(it, cid),
                iterations,
                num_samples,
            )
        )

    # Lidars
    for lidar_id in scene.available_lidar_ids + [LidarID.LIDAR_MERGED]:
        results.append(
            benchmark_access(
                f"lidar.{lidar_id.name}",
                scene,
                lambda it, lid=lidar_id: scene.get_lidar_at_iteration(it, lid),
                iterations,
                num_samples,
            )
        )

    # Custom modalities
    for custom_name in scene.get_all_custom_modality_metadatas():
        results.append(
            benchmark_access(
                f"custom.{custom_name}",
                scene,
                lambda it, cn=custom_name: scene.get_custom_modality_at_iteration(it, cn),
                iterations,
                num_samples,
            )
        )

    return results


def print_results(results: List[dict]) -> None:
    header = f"{'Modality':<40} {'Samples':>7} {'Mean':>9} {'Median':>9} {'P95':>9} {'Min':>9} {'Max':>9}"
    print(header)
    print("-" * len(header))
    for r in results:
        if r["samples"] == 0:
            print(f"{r['name']:<40} {'N/A':>7}")
            continue
        print(
            f"{r['name']:<40} {r['samples']:>7} "
            f"{r['mean_ms']:>8.2f}ms {r['median_ms']:>8.2f}ms "
            f"{r['p95_ms']:>8.2f}ms {r['min_ms']:>8.2f}ms {r['max_ms']:>8.2f}ms"
        )


def main():
    parser = argparse.ArgumentParser(description="Benchmark random access latency for scene modalities.")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name filter.")
    parser.add_argument("--split", type=str, default=None, help="Split type filter (e.g. train, val).")
    parser.add_argument("--log-name", type=str, default=None, help="Specific log name.")
    parser.add_argument("--duration", type=float, default=10.0, help="Scene duration in seconds.")
    parser.add_argument("--data-root", type=str, default=None, help="Data root directory.")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of random accesses per modality.")
    parser.add_argument("--num-scenes", type=int, default=1, help="Number of scenes to benchmark.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    random.seed(args.seed)

    scene_filter = SceneFilter(
        datasets=[args.dataset] if args.dataset else None,
        split_types=[args.split] if args.split else None,
        log_names=[args.log_name] if args.log_name else None,
        duration_s=args.duration,
        max_num_scenes=args.num_scenes,
        shuffle=True,
    )

    print(f"Loading scenes with filter: {scene_filter}")
    scenes = get_filtered_scenes(scene_filter, data_root=args.data_root)
    print(f"Found {len(scenes)} scene(s)\n")

    for i, scene in enumerate(scenes[: args.num_scenes]):
        print(f"=== Scene {i}: {scene.log_name} ({scene.number_of_iterations} iterations) ===")
        results = run_benchmark(scene, num_samples=args.num_samples)
        print_results(results)
        print()

        print("--- Timestamp loading (all timestamps per modality) ---")
        ts_results = run_timestamp_benchmark(scene)
        print_results(ts_results)
        print()


if __name__ == "__main__":
    main()
