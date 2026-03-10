"""Interactive Bokeh web UI for visualizing 123D scenes.

Launch with::

    python -m py123d.visualization.bokeh.bokeh_viewer --log-dir /path/to/log
    # or with a Bokeh server:
    bokeh serve --show src/py123d/visualization/bokeh/bokeh_viewer.py --args --log-dir /path/to/log
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from py123d.common.utils.dependencies import check_dependencies

check_dependencies(["bokeh"], "bokeh-viewer")

from bokeh.document import Document  # noqa: E402
from bokeh.io import curdoc  # noqa: E402
from bokeh.layouts import column, row  # noqa: E402
from bokeh.models import (  # noqa: E402
    ColumnDataSource,
    Div,
    Range1d,
    Select,
    Slider,
    Toggle,
)
from bokeh.plotting import figure  # noqa: E402

from py123d.api.scene.scene_api import SceneAPI  # noqa: E402
from py123d.datatypes.sensors.pinhole_camera import PinholeCameraID  # noqa: E402
from py123d.visualization.bokeh.elements import (  # noqa: E402
    get_camera_image_rgba,
    get_detection_data,
    get_ego_data,
    get_lidar_bev_data,
    get_map_data,
)

logger = logging.getLogger(__name__)

_DEFAULT_RADIUS = 80.0
_MAP_RADIUS = 500.0  # Large radius — map is queried once per scene


class BokehViewer:
    """Interactive Bokeh-based web viewer for 123D scenes.

    Provides a bird's-eye view (BEV) with map, ego vehicle, detections, and
    lidar point cloud, alongside a camera image panel.  The user can scrub
    through iterations with a slider, toggle layers, and switch cameras.
    """

    def __init__(
        self,
        scenes: List[SceneAPI],
        radius: float = _DEFAULT_RADIUS,
        show_lidar: bool = True,
        show_map: bool = True,
    ) -> None:
        assert len(scenes) > 0, "At least one scene is required."
        self._scenes = scenes
        self._radius = radius
        self._show_lidar = show_lidar
        self._show_map = show_map

        self._scene_idx = 0
        self._iteration = 0

        # Persistent data sources. Render order = z-order (last added = on top).
        # Map sources are created dynamically in _load_map (rendered at level='image').
        self._lidar_source = ColumnDataSource(data={"x": [], "y": []})
        self._det_source = ColumnDataSource(data={"xs": [], "ys": [], "color": [], "line_color": []})
        self._ego_source = ColumnDataSource(data={"xs": [], "ys": []})
        self._camera_source = ColumnDataSource(data={"image": [], "dw": [], "dh": []})

        self._map_renderers: list = []
        self._lidar_renderer = None
        self._det_renderer = None
        self._ego_renderer = None

    @property
    def scene(self) -> SceneAPI:
        return self._scenes[self._scene_idx]

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, doc: Optional[Document] = None):
        """Build the full Bokeh document layout and return it.

        :param doc: Bokeh Document to add roots to. If None, uses curdoc().
        """
        if doc is None:
            doc = curdoc()
        doc.title = "123D Scene Viewer"

        # ---- BEV figure ----
        bev = figure(
            title="Bird's-Eye View",
            match_aspect=True,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            active_scroll="wheel_zoom",
            width=700,
            height=700,
            output_backend="webgl",
        )
        bev.xgrid.visible = False
        bev.ygrid.visible = False
        bev.background_fill_color = "#f5f5f0"
        self._bev = bev

        # ---- Camera figure ----
        # Uses normalized 0..1 coordinates; pixel height is adjusted dynamically
        # in _update_camera to match the actual image aspect ratio.
        self._cam_width = 700
        cam_fig = figure(
            title="Camera",
            tools="pan,wheel_zoom,reset,save",
            active_scroll="wheel_zoom",
            width=self._cam_width,
            height=400,
            x_range=Range1d(0, 1, bounds=(0, 1)),
            y_range=Range1d(0, 1, bounds=(0, 1)),
        )
        cam_fig.xaxis.visible = False
        cam_fig.yaxis.visible = False
        cam_fig.xgrid.visible = False
        cam_fig.ygrid.visible = False
        self._cam_fig = cam_fig
        cam_fig.image_rgba(image="image", x=0, y=0, dw=1, dh=1, source=self._camera_source)

        # ---- Renderers in z-order (map at bottom, ego on top) ----
        self._setup_renderers()

        # ---- Widgets ----
        scene_options = [f"{i}: {s.log_name}" for i, s in enumerate(self._scenes)]
        self._scene_select = Select(title="Scene", value=scene_options[0], options=scene_options, width=300)
        self._scene_select.on_change("value", self._on_scene_change)

        self._slider = Slider(start=0, end=max(self.scene.number_of_iterations - 1, 1), value=0, step=1, title="Frame")
        self._slider.on_change("value", self._on_iteration_change)

        camera_names = self._get_camera_options()
        default_cam = camera_names[0] if camera_names else "none"
        self._camera_select = Select(title="Camera", value=default_cam, options=camera_names or ["none"], width=200)
        self._camera_select.on_change("value", self._on_camera_change)

        self._map_toggle = Toggle(label="Map", active=self._show_map, button_type="default", width=80)
        self._map_toggle.on_click(self._on_map_toggle)
        self._lidar_toggle = Toggle(label="Lidar", active=self._show_lidar, button_type="default", width=80)
        self._lidar_toggle.on_click(self._on_lidar_toggle)
        self._det_toggle = Toggle(label="Detections", active=True, button_type="default", width=100)
        self._det_toggle.on_click(self._on_det_toggle)

        self._info_div = Div(text="", width=700)

        # ---- Layout ----
        controls = row(self._scene_select, self._camera_select, self._map_toggle, self._lidar_toggle, self._det_toggle)
        left = column(bev, self._slider, controls)
        right = column(cam_fig, self._info_div)
        doc.add_root(row(left, right))

        # Initial data load
        self._load_map()
        self._update_frame()
        return doc

    # ------------------------------------------------------------------
    # Renderer setup (z-order)
    # ------------------------------------------------------------------

    def _setup_renderers(self):
        """Add persistent renderers in correct z-order (last = on top)."""
        bev = self._bev

        # 1. Lidar (below boxes)
        self._lidar_renderer = bev.scatter(
            x="x",
            y="y",
            source=self._lidar_source,
            size=1.5,
            color="#2b2b2b",
            alpha=0.4,
            level="glyph",
        )
        self._lidar_renderer.visible = self._show_lidar

        # 2. Detections
        self._det_renderer = bev.patches(
            xs="xs",
            ys="ys",
            source=self._det_source,
            fill_color="color",
            fill_alpha=0.9,
            line_color="line_color",
            line_width=1.0,
            level="glyph",
        )

        # 3. Ego (top-most)
        self._ego_renderer = bev.patches(
            xs="xs",
            ys="ys",
            source=self._ego_source,
            fill_color="#DE7061",
            line_color="#000000",
            line_width=1.5,
            fill_alpha=1.0,
            level="glyph",
        )

    # ------------------------------------------------------------------
    # Map (loaded once per scene)
    # ------------------------------------------------------------------

    def _load_map(self):
        """Query the map once with a large radius and add static renderers."""
        for r in self._map_renderers:
            r.visible = False
        self._map_renderers.clear()

        if not self._show_map:
            return

        map_api = self.scene.get_map_api()
        if map_api is None:
            return

        ego = self.scene.get_ego_state_se3_at_iteration(0)
        if ego is None:
            return

        center = ego.bounding_box_se2.center_se2.pose_se2.point_2d
        logger.info("Querying map (radius=%.0f m) centered at (%.0f, %.0f) ...", _MAP_RADIUS, center.x, center.y)
        map_data = get_map_data(map_api, center.x, center.y, _MAP_RADIUS)

        for data in map_data.values():
            src = ColumnDataSource(data={"xs": data["xs"], "ys": data["ys"]})
            if data.get("type") == "lines":
                r = self._bev.multi_line(
                    xs="xs",
                    ys="ys",
                    source=src,
                    line_color=data["color"],
                    line_alpha=data["alpha"],
                    line_width=1,
                    line_dash="dashed",
                    level="image",
                )
            else:
                r = self._bev.patches(
                    xs="xs",
                    ys="ys",
                    source=src,
                    fill_color=data["color"],
                    fill_alpha=data["alpha"],
                    line_color=data["color"],
                    line_width=0.5,
                    line_alpha=0.3,
                    level="image",
                )
            self._map_renderers.append(r)
        logger.info("Map loaded with %d layer groups.", len(self._map_renderers))

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def _update_frame(self):
        """Update dynamic elements for the current iteration (ego, detections, lidar, camera)."""
        scene = self.scene
        iteration = self._iteration

        ego = scene.get_ego_state_se3_at_iteration(iteration)
        if ego is None:
            return

        center = ego.bounding_box_se2.center_se2.pose_se2.point_2d
        cx, cy = center.x, center.y

        # Ego
        ego_data = get_ego_data(ego)
        self._ego_source.data = {"xs": ego_data["xs"], "ys": ego_data["ys"]}

        # Detections — merge all labels into one source with per-patch colors
        detections = scene.get_box_detections_se3_at_iteration(iteration)
        det_data = get_detection_data(detections)
        all_xs: list = []
        all_ys: list = []
        all_colors: list = []
        all_line_colors: list = []
        for data in det_data.values():
            n = len(data["xs"])
            all_xs.extend(data["xs"])
            all_ys.extend(data["ys"])
            all_colors.extend([data["color"]] * n)
            all_line_colors.extend([data.get("line_color", "#000000")] * n)
        self._det_source.data = {"xs": all_xs, "ys": all_ys, "color": all_colors, "line_color": all_line_colors}

        # Lidar
        if self._show_lidar and self._lidar_renderer.visible:
            lidar_data = get_lidar_bev_data(scene, iteration, ego)
            self._lidar_source.data = lidar_data if lidar_data is not None else {"x": [], "y": []}
        else:
            self._lidar_source.data = {"x": [], "y": []}

        # BEV viewport
        self._bev.x_range.start = cx - self._radius
        self._bev.x_range.end = cx + self._radius
        self._bev.y_range.start = cy - self._radius
        self._bev.y_range.end = cy + self._radius

        # Camera + info
        self._update_camera()
        self._info_div.text = self._build_info_html()

    def _update_camera(self):
        """Update the camera image panel."""
        cam_name = self._camera_select.value
        if cam_name == "none":
            self._camera_source.data = {"image": [], "dw": [], "dh": []}
            return

        cam_id = self._camera_name_to_id().get(cam_name)
        if cam_id is None:
            self._camera_source.data = {"image": [], "dw": [], "dh": []}
            return

        rgba = get_camera_image_rgba(self.scene, self._iteration, cam_id)
        if rgba is None:
            self._camera_source.data = {"image": [], "dw": [], "dh": []}
            return

        h, w = rgba.shape[:2]
        img = np.zeros((h, w), dtype=np.uint32)
        img.view(dtype=np.uint8).reshape((h, w, 4))[:] = rgba
        # Use normalized coords (0..1); image fills entire figure.
        self._camera_source.data = {"image": [img], "dw": [1], "dh": [1]}
        # Adjust figure pixel height to match image aspect ratio
        self._cam_fig.height = int(self._cam_width * h / w)
        self._cam_fig.title.text = f"Camera: {cam_name}"

    # ------------------------------------------------------------------
    # Widget callbacks
    # ------------------------------------------------------------------

    def _on_scene_change(self, attr, old, new):
        idx = int(new.split(":")[0])
        self._scene_idx = idx
        self._iteration = 0
        self._slider.end = max(self.scene.number_of_iterations - 1, 1)
        self._slider.value = 0

        camera_names = self._get_camera_options()
        self._camera_select.options = camera_names or ["none"]
        self._camera_select.value = camera_names[0] if camera_names else "none"

        self._load_map()
        self._update_frame()

    def _on_iteration_change(self, attr, old, new):
        self._iteration = int(new)
        self._update_frame()

    def _on_camera_change(self, attr, old, new):
        self._update_camera()

    def _on_map_toggle(self, active):
        self._show_map = active
        for r in self._map_renderers:
            r.visible = active

    def _on_lidar_toggle(self, active):
        self._show_lidar = active
        if self._lidar_renderer is not None:
            self._lidar_renderer.visible = active
        if active:
            ego = self.scene.get_ego_state_se3_at_iteration(self._iteration)
            if ego is not None:
                lidar_data = get_lidar_bev_data(self.scene, self._iteration, ego)
                if lidar_data is not None:
                    self._lidar_source.data = lidar_data
                    return
        if not active:
            self._lidar_source.data = {"x": [], "y": []}

    def _on_det_toggle(self, active):
        if self._det_renderer is not None:
            self._det_renderer.visible = active

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_camera_options(self) -> List[str]:
        names = self.scene.available_pinhole_camera_names
        return names if names else []

    def _camera_name_to_id(self) -> Dict[str, PinholeCameraID]:
        metadatas = self.scene.get_pinhole_camera_metadatas()
        return {meta.camera_name: cam_id for cam_id, meta in metadatas.items()}

    def _build_info_html(self) -> str:
        scene = self.scene
        lines = [
            "<div style='font-family: monospace; font-size: 13px; padding: 10px; "
            "background: #fafafa; border: 1px solid #ddd; border-radius: 6px;'>",
            f"<b>Dataset:</b> {scene.dataset}<br>",
            f"<b>Split:</b> {scene.split}<br>",
            f"<b>Log:</b> {scene.log_name}<br>",
        ]
        if scene.location:
            lines.append(f"<b>Location:</b> {scene.location}<br>")
        lines.append(f"<b>Frames:</b> {scene.number_of_iterations}<br>")
        lines.append(f"<b>Current frame:</b> {self._iteration}<br>")
        ego = scene.get_ego_state_se3_at_iteration(self._iteration)
        if ego is not None:
            p = ego.imu_se3
            lines.append(f"<b>Ego (x, y):</b> ({p.x:.1f}, {p.y:.1f})<br>")
        detections = scene.get_box_detections_se3_at_iteration(self._iteration)
        if detections is not None:
            lines.append(f"<b>Detections:</b> {len(list(detections))}<br>")
        lines.append(f"<b>Cameras:</b> {len(scene.available_pinhole_camera_ids)}<br>")
        lines.append(f"<b>Lidars:</b> {len(scene.available_lidar_ids)}<br>")
        lines.append("</div>")
        return "\n".join(lines)


# ------------------------------------------------------------------
# Factory helpers
# ------------------------------------------------------------------


def create_viewer_from_log_dirs(
    log_dirs: List[str],
    radius: float = _DEFAULT_RADIUS,
    show_lidar: bool = True,
    show_map: bool = True,
) -> BokehViewer:
    """Create a BokehViewer from a list of log directory paths."""
    from py123d.api.scene.arrow.arrow_scene_api import ArrowSceneAPI

    scenes = [ArrowSceneAPI(log_dir=d) for d in log_dirs]
    return BokehViewer(scenes, radius=radius, show_lidar=show_lidar, show_map=show_map)


def create_viewer_from_scenes(
    scenes: List[SceneAPI],
    radius: float = _DEFAULT_RADIUS,
    show_lidar: bool = True,
    show_map: bool = True,
) -> BokehViewer:
    """Create a BokehViewer from a list of pre-loaded SceneAPI objects."""
    return BokehViewer(scenes, radius=radius, show_lidar=show_lidar, show_map=show_map)


# ------------------------------------------------------------------
# Entry points
# ------------------------------------------------------------------


def _bokeh_server_entry():
    """Entry point when served via ``bokeh serve``."""
    import sys

    log_dirs = []
    radius = _DEFAULT_RADIUS
    show_lidar = True
    show_map = True

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--log-dir" and i + 1 < len(args):
            log_dirs.append(args[i + 1])
            i += 2
        elif args[i] == "--radius" and i + 1 < len(args):
            radius = float(args[i + 1])
            i += 2
        elif args[i] == "--no-lidar":
            show_lidar = False
            i += 1
        elif args[i] == "--no-map":
            show_map = False
            i += 1
        else:
            i += 1

    if not log_dirs:
        raise ValueError("At least one --log-dir must be specified.")

    viewer = create_viewer_from_log_dirs(log_dirs, radius=radius, show_lidar=show_lidar, show_map=show_map)
    viewer.build()


def main():
    """Standalone entry point: launches a Bokeh server programmatically."""
    import argparse

    from bokeh.server.server import Server

    parser = argparse.ArgumentParser(description="123D Bokeh Scene Viewer")
    parser.add_argument("--log-dir", type=str, nargs="+", required=True, help="Path(s) to log directories")
    parser.add_argument("--radius", type=float, default=_DEFAULT_RADIUS, help="BEV radius in meters")
    parser.add_argument("--port", type=int, default=5006, help="Server port")
    parser.add_argument("--no-lidar", action="store_true", help="Disable lidar display")
    parser.add_argument("--no-map", action="store_true", help="Disable map display")
    args = parser.parse_args()

    from py123d.api.scene.arrow.arrow_scene_api import ArrowSceneAPI

    scenes = [ArrowSceneAPI(log_dir=d) for d in args.log_dir]

    def make_doc(doc):
        viewer = BokehViewer(
            scenes,
            radius=args.radius,
            show_lidar=not args.no_lidar,
            show_map=not args.no_map,
        )
        viewer.build(doc)

    server = Server({"/": make_doc}, port=args.port)
    server.start()
    print(f"\n  123D Scene Viewer running at: http://localhost:{args.port}/\n")

    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()


# When loaded by ``bokeh serve``, auto-run
if __name__.startswith("bk_script") or __name__ == "__main__":
    if __name__ == "__main__":
        main()
    else:
        _bokeh_server_entry()
