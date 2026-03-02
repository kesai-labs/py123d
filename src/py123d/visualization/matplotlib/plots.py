from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib import animation
from tqdm import tqdm

from py123d.api.scene.scene_api import SceneAPI
from py123d.visualization.matplotlib.observation import add_scene_on_ax


def plot_scene_at_iteration(scene: SceneAPI, iteration: int = 0, radius: float = 80) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(10, 10))
    add_scene_on_ax(ax, scene, iteration, radius)
    return fig, ax


def render_scene_animation(
    scene: SceneAPI,
    output_path: Union[str, Path],
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    step: int = 10,
    fps: float = 20.0,
    dpi: int = 300,
    format: str = "mp4",
    radius: float = 80,
) -> None:
    assert format in ["mp4", "gif"], "Format must be either 'mp4' or 'gif'."
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if end_idx is None:
        end_idx = scene.number_of_iterations
    end_idx = min(end_idx, scene.number_of_iterations)

    fig, ax = plt.subplots(figsize=(10, 10))

    def update(i):
        ax.clear()
        add_scene_on_ax(ax, scene, i, radius)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        pbar.update(1)

    frames = list(range(start_idx, end_idx, step))
    pbar = tqdm(total=len(frames), desc=f"Rendering {scene.log_name} as {format}")
    ani = animation.FuncAnimation(fig, update, frames=frames, repeat=False)

    ani.save(output_path / f"{scene.log_name}_{str(scene.scene_uuid)}.{format}", writer="ffmpeg", fps=fps, dpi=dpi)
    plt.close(fig)
