import io
import logging
import time
import zipfile
from typing import Dict, List, Optional, Union

import imageio.v3 as iio
import viser
from tqdm import tqdm
from viser.theme import TitlebarButton, TitlebarConfig, TitlebarImage

from py123d.api.scene.scene_api import SceneAPI
from py123d.datatypes.map_objects.map_layer_types import MapLayer
from py123d.datatypes.sensors.fisheye_mei_camera import FisheyeMEICameraID
from py123d.datatypes.sensors.lidar import LidarID
from py123d.datatypes.sensors.pinhole_camera import PinholeCameraID
from py123d.datatypes.vehicle_state.ego_state import EgoStateSE3
from py123d.visualization.viser.elements import (
    add_box_detections_to_viser_server,
    add_camera_frustums_to_viser_server,
    add_camera_gui_to_viser_server,
    add_lidar_pc_to_viser_server,
    add_map_to_viser_server,
)
from py123d.visualization.viser.elements.render_elements import (
    get_ego_3rd_person_view_position,
    get_ego_bev_view_position,
)
from py123d.visualization.viser.elements.sensor_elements import add_fisheye_frustums_to_viser_server
from py123d.visualization.viser.viser_config import ViserConfig

logger = logging.getLogger(__name__)


def _build_viser_server(viser_config: ViserConfig) -> viser.ViserServer:
    server = viser.ViserServer(
        host=viser_config.server_host,
        port=viser_config.server_port,
        label=viser_config.server_label,
        verbose=viser_config.server_verbose,
    )

    buttons = (
        TitlebarButton(
            text="Getting Started",
            icon=None,
            href="https://autonomousvision.github.io/py123d",
        ),
        TitlebarButton(
            text="Github",
            icon="GitHub",
            href="https://github.com/autonomousvision/py123d",
        ),
        TitlebarButton(
            text="Documentation",
            icon="Description",
            href="https://autonomousvision.github.io/py123d",
        ),
    )
    image = TitlebarImage(
        image_url_light="https://autonomousvision.github.io/py123d/_static/123D_logo_transparent_black.svg",
        image_url_dark="https://autonomousvision.github.io/py123d/_static/123D_logo_transparent_white.svg",
        image_alt="123D",
        href="https://autonomousvision.github.io/py123d/",
    )
    titlebar_theme = TitlebarConfig(buttons=buttons, image=image)

    server.gui.configure_theme(
        titlebar_content=titlebar_theme,
        control_layout=viser_config.theme_control_layout,
        control_width=viser_config.theme_control_width,
        dark_mode=viser_config.theme_dark_mode,
        show_logo=viser_config.theme_show_logo,
        show_share_button=viser_config.theme_show_share_button,
        brand_color=viser_config.theme_brand_color,
    )
    return server


class ViserViewer:
    def __init__(
        self,
        scenes: List[SceneAPI],
        viser_config: ViserConfig = ViserConfig(),
        scene_index: int = 0,
    ) -> None:
        assert len(scenes) > 0, "At least one scene must be provided."

        self._scenes = scenes
        self._viser_config = viser_config
        self._scene_index = scene_index

        self._viser_server = _build_viser_server(self._viser_config)
        self.set_scene(self._scenes[self._scene_index % len(self._scenes)])

    def next(self) -> None:
        self._viser_server.flush()
        self._viser_server.gui.reset()
        self._viser_server.scene.reset()
        self._scene_index = (self._scene_index + 1) % len(self._scenes)
        self.set_scene(self._scenes[self._scene_index])

    def set_scene(self, scene: SceneAPI) -> None:
        num_frames = scene.number_of_iterations
        initial_ego_state = scene.get_ego_state_at_iteration(0)
        assert initial_ego_state is not None and isinstance(initial_ego_state, EgoStateSE3)

        server_playing = True
        server_rendering = False

        with self._viser_server.gui.add_folder("Playback"):
            self._viser_server.gui.add_markdown(content=_get_scene_info_markdown(scene))

            gui_timestep = self._viser_server.gui.add_slider(
                "Timestep",
                min=0,
                max=num_frames - 1,
                step=1,
                initial_value=0,
                disabled=True,
            )
            gui_next_frame = self._viser_server.gui.add_button("Next Frame", disabled=True)
            gui_prev_frame = self._viser_server.gui.add_button("Prev Frame", disabled=True)
            gui_next_scene = self._viser_server.gui.add_button("Next Scene", disabled=False)
            gui_playing = self._viser_server.gui.add_checkbox("Playing", self._viser_config.is_playing)
            gui_speed = self._viser_server.gui.add_slider(
                "Playback speed", min=0.1, max=10.0, step=0.1, initial_value=self._viser_config.playback_speed
            )
            gui_speed_options = self._viser_server.gui.add_button_group(
                "Options.", ("0.5", "1.0", "2.0", "5.0", "10.0")
            )

        with self._viser_server.gui.add_folder("Modalities", expand_by_default=True):
            modalities_map_visible = self._viser_server.gui.add_checkbox("Map", self._viser_config.map_visible)
            modalities_bounding_box_visible = self._viser_server.gui.add_checkbox(
                "Bounding Boxes", self._viser_config.bounding_box_visible
            )
            modalities_camera_frustum_visible = self._viser_server.gui.add_checkbox(
                "Camera Frustums", self._viser_config.camera_frustum_visible
            )
            modalities_lidar_visible = self._viser_server.gui.add_checkbox("Lidar", self._viser_config.lidar_visible)

        with self._viser_server.gui.add_folder("Options", expand_by_default=True):
            option_bounding_box_type = self._viser_server.gui.add_dropdown(
                "Bounding Box Type", ("mesh", "lines"), initial_value=self._viser_config.bounding_box_type
            )
            options_map_radius_slider = self._viser_server.gui.add_slider(
                "Map Radius", min=10.0, max=1000.0, step=1.0, initial_value=self._viser_config.map_radius
            )
            options_map_radius_options = self._viser_server.gui.add_button_group(
                "Map Radius Options.", ("25", "50", "100", "500")
            )
            option_lidar_point_color = self._viser_server.gui.add_dropdown(
                "Lidar Coloring",
                ("none", "distance", "ids", "intensity", "channel", "timestamp", "range", "elongation"),
                initial_value=self._viser_config.lidar_point_color,
            )

            lidar_id_list = [LidarID.LIDAR_MERGED] + scene.available_lidar_ids
            lidar_id_names = tuple(lid.name for lid in lidar_id_list)
            option_lidar_id = self._viser_server.gui.add_dropdown(
                "Lidar ID",
                lidar_id_names,
                initial_value=self._viser_config.lidar_ids[0].name,
            )

        with self._viser_server.gui.add_folder("Render", expand_by_default=False):
            render_format = self._viser_server.gui.add_dropdown("Format", ["gif", "mp4", "png"], initial_value="mp4")
            render_view = self._viser_server.gui.add_dropdown(
                "View", ["3rd Person", "BEV", "Manual"], initial_value="3rd Person"
            )
            render_button = self._viser_server.gui.add_button("Render Scene")

        # Options:
        @modalities_map_visible.on_update
        def _(_) -> None:
            for map_handle in map_handles.values():
                map_handle.visible = modalities_map_visible.value
            self._viser_config.map_visible = modalities_map_visible.value

        @modalities_bounding_box_visible.on_update
        def _(_) -> None:
            if box_detection_handles["lines"] is not None:
                box_detection_handles["lines"].visible = modalities_bounding_box_visible.value
            if box_detection_handles["mesh"] is not None:
                box_detection_handles["mesh"].visible = modalities_bounding_box_visible.value
            self._viser_config.bounding_box_visible = modalities_bounding_box_visible.value

        @modalities_camera_frustum_visible.on_update
        def _(_) -> None:
            for frustum_handle in camera_frustum_handles.values():
                frustum_handle.visible = modalities_camera_frustum_visible.value
            self._viser_config.camera_frustum_visible = modalities_camera_frustum_visible.value

        @modalities_lidar_visible.on_update
        def _(_) -> None:
            for lidar_pc_handle in lidar_pc_handles.values():
                if lidar_pc_handle is not None:
                    lidar_pc_handle.visible = modalities_lidar_visible.value
            self._viser_config.lidar_visible = modalities_lidar_visible.value

        @option_bounding_box_type.on_update
        def _(_) -> None:
            self._viser_config.bounding_box_type = option_bounding_box_type.value

        @options_map_radius_slider.on_update
        def _(_) -> None:
            self._viser_config.map_radius = options_map_radius_slider.value
            self._viser_config._force_map_update = True

        @options_map_radius_options.on_click
        def _(_) -> None:
            options_map_radius_slider.value = float(options_map_radius_options.value)
            self._viser_config._force_map_update = True

        @option_lidar_point_color.on_update
        def _(_) -> None:
            self._viser_config.lidar_point_color = option_lidar_point_color.value
            add_lidar_pc_to_viser_server(
                scene,
                gui_timestep.value,
                initial_ego_state,
                self._viser_server,
                self._viser_config,
                lidar_pc_handles,
            )

        @option_lidar_id.on_update
        def _(_) -> None:
            self._viser_config.lidar_ids = [LidarID[option_lidar_id.value]]
            add_lidar_pc_to_viser_server(
                scene,
                gui_timestep.value,
                initial_ego_state,
                self._viser_server,
                self._viser_config,
                lidar_pc_handles,
            )

        # Frame step buttons.
        @gui_next_frame.on_click
        def _(_) -> None:
            gui_timestep.value = (gui_timestep.value + 1) % num_frames

        @gui_prev_frame.on_click
        def _(_) -> None:
            gui_timestep.value = (gui_timestep.value - 1) % num_frames

        @gui_next_scene.on_click
        def _(_) -> None:
            nonlocal server_playing
            server_playing = False

        # Disable frame controls when we're playing.
        @gui_playing.on_update
        def _(_) -> None:
            gui_timestep.disabled = gui_playing.value
            gui_next_frame.disabled = gui_playing.value
            gui_prev_frame.disabled = gui_playing.value
            self._viser_config.is_playing = gui_playing.value

        # Set the framerate when we click one of the options.
        @gui_speed_options.on_click
        def _(_) -> None:
            gui_speed.value = float(gui_speed_options.value)

        # Toggle frame visibility when the timestep slider changes.
        @gui_timestep.on_update
        def _(_) -> None:
            start = time.perf_counter()
            add_box_detections_to_viser_server(
                scene,
                gui_timestep.value,
                initial_ego_state,
                self._viser_server,
                self._viser_config,
                box_detection_handles,
            )
            add_camera_frustums_to_viser_server(
                scene,
                gui_timestep.value,
                initial_ego_state,
                self._viser_server,
                self._viser_config,
                camera_frustum_handles,
            )
            add_camera_gui_to_viser_server(
                scene,
                gui_timestep.value,
                self._viser_server,
                self._viser_config,
                camera_gui_handles,
            )
            add_fisheye_frustums_to_viser_server(
                scene,
                gui_timestep.value,
                initial_ego_state,
                self._viser_server,
                self._viser_config,
                fisheye_frustum_handles,
            )
            add_lidar_pc_to_viser_server(
                scene,
                gui_timestep.value,
                initial_ego_state,
                self._viser_server,
                self._viser_config,
                lidar_pc_handles,
            )
            add_map_to_viser_server(
                scene,
                gui_timestep.value,
                initial_ego_state,
                self._viser_server,
                self._viser_config,
                map_handles,
            )
            rendering_time = time.perf_counter() - start

            sleep_time = 1.0 / gui_speed.value - rendering_time

            # Calculate sleep time based on speed factor
            base_frame_time = scene.log_metadata.timestep_seconds
            target_frame_time = base_frame_time / gui_speed.value
            sleep_time = target_frame_time - rendering_time

            if sleep_time > 0 and not server_rendering:
                time.sleep(max(sleep_time, 0.0))

        @render_button.on_click
        def _(event: viser.GuiEvent) -> None:
            nonlocal server_rendering
            client = event.client
            assert client is not None

            client.scene.reset()

            server_rendering = True
            images = []

            for i in tqdm(range(scene.number_of_iterations)):
                gui_timestep.value = i
                if render_view.value == "BEV":
                    ego_view = get_ego_bev_view_position(scene, i, initial_ego_state)
                    client.camera.position = ego_view.point_3d.array
                    client.camera.wxyz = ego_view.quaternion.array
                elif render_view.value == "3rd Person":
                    ego_view = get_ego_3rd_person_view_position(scene, i, initial_ego_state)
                    client.camera.position = ego_view.point_3d.array
                    client.camera.wxyz = ego_view.quaternion.array
                images.append(client.get_render(height=1080, width=1920))
            format = render_format.value
            buffer = io.BytesIO()
            if format == "gif":
                iio.imwrite(buffer, images, extension=".gif", loop=False)
            elif format == "mp4":
                iio.imwrite(buffer, images, extension=".mp4", fps=20)
            elif format == "png":
                # Create an in-memory ZIP containing all frames as PNGs
                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for idx, img in enumerate(images):
                        name = f"frame_{idx:05d}.png"
                        if isinstance(img, (bytes, bytearray)):
                            zf.writestr(name, img)
                        else:
                            img_bytes = io.BytesIO()
                            iio.imwrite(img_bytes, img, extension=".png")
                            zf.writestr(name, img_bytes.getvalue())
                zip_buf.seek(0)
                content = zip_buf.getvalue()
                format = "zip"
            scene_name = f"{scene.log_metadata.split}_{scene.scene_uuid}"
            client.send_file_download(f"{scene_name}.{format}", content, save_immediately=True)
            server_rendering = False

        box_detection_handles: Dict[str, Union[viser.GlbHandle, viser.LineSegmentsHandle]] = {
            "mesh": None,
            "lines": None,
        }
        camera_frustum_handles: Dict[PinholeCameraID, viser.CameraFrustumHandle] = {}
        fisheye_frustum_handles: Dict[FisheyeMEICameraID, viser.CameraFrustumHandle] = {}
        camera_gui_handles: Dict[PinholeCameraID, viser.GuiImageHandle] = {}
        lidar_pc_handles: Dict[LidarID, Optional[viser.PointCloudHandle]] = {LidarID.LIDAR_MERGED: None}
        map_handles: Dict[MapLayer, viser.MeshHandle] = {}

        add_box_detections_to_viser_server(
            scene,
            gui_timestep.value,
            initial_ego_state,
            self._viser_server,
            self._viser_config,
            box_detection_handles,
        )
        add_camera_frustums_to_viser_server(
            scene,
            gui_timestep.value,
            initial_ego_state,
            self._viser_server,
            self._viser_config,
            camera_frustum_handles,
        )
        add_camera_gui_to_viser_server(
            scene,
            gui_timestep.value,
            self._viser_server,
            self._viser_config,
            camera_gui_handles,
        )
        add_fisheye_frustums_to_viser_server(
            scene,
            gui_timestep.value,
            initial_ego_state,
            self._viser_server,
            self._viser_config,
            fisheye_frustum_handles,
        )
        add_lidar_pc_to_viser_server(
            scene,
            gui_timestep.value,
            initial_ego_state,
            self._viser_server,
            self._viser_config,
            lidar_pc_handles,
        )
        add_map_to_viser_server(
            scene,
            gui_timestep.value,
            initial_ego_state,
            self._viser_server,
            self._viser_config,
            map_handles,
        )

        # Playback update loop.
        while server_playing:
            if gui_playing.value and not server_rendering:
                gui_timestep.value = (gui_timestep.value + 1) % num_frames
            else:
                time.sleep(0.1)

            # update config
            self._viser_config.playback_speed = gui_speed.value

        self._viser_server.flush()
        self.next()


def _get_scene_info_markdown(scene: SceneAPI) -> str:
    markdown = f"""
    - Dataset: {scene.log_metadata.split}
    - Location: {scene.log_metadata.location if scene.log_metadata.location else "N/A"}
    - Log: {scene.log_metadata.log_name}
    - UUID: {scene.scene_uuid}
    """
    return markdown
