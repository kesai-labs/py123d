from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import trimesh
import viser

from py123d.api.scene.scene_api import SceneAPI
from py123d.conversion.registry.box_detection_label_registry import DefaultBoxDetectionLabel
from py123d.datatypes.vehicle_state.ego_state import EgoStateSE3
from py123d.geometry.geometry_index import BoundingBoxSE3Index, Corners3DIndex, PoseSE3Index
from py123d.geometry.utils.bounding_box_utils import (
    bbse3_array_to_corners_array,
    corners_array_to_3d_mesh,
    corners_array_to_edge_lines,
)
from py123d.visualization.color.default import BOX_DETECTION_CONFIG
from py123d.visualization.viser.viser_config import ViserConfig


def add_box_detections_to_viser_server(
    scene: SceneAPI,
    scene_interation: int,
    initial_ego_state: EgoStateSE3,
    viser_server: viser.ViserServer,
    viser_config: ViserConfig,
    box_detection_handles: Optional[Union[viser.GlbHandle, viser.LineSegmentsHandle]],
) -> None:
    visible_handle_keys = []
    if viser_config.bounding_box_visible:
        if viser_config.bounding_box_type == "mesh":
            mesh = _get_bounding_box_meshes(scene, scene_interation, initial_ego_state)
            box_detection_handles["mesh"] = viser_server.scene.add_mesh_trimesh(
                "box_detections",
                mesh=mesh,
                visible=True,
            )
            visible_handle_keys.append("mesh")
        elif viser_config.bounding_box_type == "lines":
            lines, colors, se3_array = _get_bounding_box_outlines(scene, scene_interation, initial_ego_state)
            box_detection_handles["lines"] = viser_server.scene.add_line_segments(
                "box_detections",
                points=lines,
                colors=colors,
                line_width=viser_config.bounding_box_line_width,
                visible=True,
            )
            # viser_server.scene.add_batched_axes(
            #     "frames",
            #     batched_wxyzs=se3_array[:-1, PoseSE3Index.QUATERNION],
            #     batched_positions=se3_array[:-1, PoseSE3Index.XYZ],
            # )
            # ego_rear_axle_se3 = scene.get_ego_state_at_iteration(scene_interation).rear_axle_se3.array
            # ego_rear_axle_se3[PoseSE3Index.XYZ] -= initial_ego_state.center_se3.array[PoseSE3Index.XYZ]
            # viser_server.scene.add_frame(
            #     "ego_rear_axle",
            #     position=ego_rear_axle_se3[PoseSE3Index.XYZ],
            #     wxyz=ego_rear_axle_se3[PoseSE3Index.QUATERNION],
            # )
            visible_handle_keys.append("lines")

        else:
            raise ValueError(f"Unknown bounding box type: {viser_config.bounding_box_type}")

    for key in box_detection_handles:
        if key not in visible_handle_keys and box_detection_handles[key] is not None:
            box_detection_handles[key].visible = False


def _get_bounding_box_meshes(scene: SceneAPI, iteration: int, initial_ego_state: EgoStateSE3) -> trimesh.Trimesh:
    ego_vehicle_state = scene.get_ego_state_at_iteration(iteration)
    box_detections = scene.get_box_detections_at_iteration(iteration)

    # Load boxes to visualize, including ego vehicle at the last position
    boxes = [bd.bounding_box_se3 for bd in box_detections.box_detections] + [ego_vehicle_state.bounding_box_se3]
    boxes_labels = [bd.metadata.default_label for bd in box_detections.box_detections] + [DefaultBoxDetectionLabel.EGO]

    # create meshes for all boxes
    box_se3_array = np.array([box.array for box in boxes])
    box_se3_array[..., BoundingBoxSE3Index.XYZ] -= initial_ego_state.center_se3.array[PoseSE3Index.XYZ]
    box_corners_array = bbse3_array_to_corners_array(box_se3_array)
    box_vertices, box_faces = corners_array_to_3d_mesh(box_corners_array)

    # Create colors for each box based on detection type
    box_colors = []
    for box_lable in boxes_labels:
        box_colors.append(BOX_DETECTION_CONFIG[box_lable].fill_color.rgba)

    # Convert to numpy array and repeat for each vertex
    box_colors = np.array(box_colors)
    vertex_colors = np.repeat(box_colors, len(Corners3DIndex), axis=0)

    # Create trimesh object
    mesh = trimesh.Trimesh(vertices=box_vertices, faces=box_faces)
    mesh.visual.vertex_colors = vertex_colors

    return mesh


# def _get_bounding_box_outlines(
#     scene: AbstractScene, iteration: int, initial_ego_state: EgoStateSE3
# ) -> npt.NDArray[np.float64]:

#     ego_vehicle_state = scene.get_ego_state_at_iteration(iteration)
#     box_detections = scene.get_box_detections_at_iteration(iteration)

#     # Load boxes to visualize, including ego vehicle at the last position
#     boxes = [bd.bounding_box_se3 for bd in box_detections.box_detections] + [ego_vehicle_state.bounding_box_se3]
#     boxes_type = [bd.metadata.detection_type for bd in box_detections.box_detections] + [DetectionType.EGO]

#     # Create lines for all boxes
#     box_se3_array = np.array([box.array for box in boxes])
#     box_se3_array[..., BoundingBoxSE3Index.XYZ] -= initial_ego_state.center_se3.array[PoseSE3Index.XYZ]
#     box_corners_array = bbse3_array_to_corners_array(box_se3_array)
#     box_outlines = corners_array_to_edge_lines(box_corners_array)

#     # Create colors for all boxes
#     box_colors = np.zeros(box_outlines.shape, dtype=np.float32)
#     for i, box_type in enumerate(boxes_type):
#         box_colors[i, ...] = BOX_DETECTION_CONFIG[box_type].fill_color.rgb_norm

#     box_outlines = box_outlines.reshape(-1, *box_outlines.shape[2:])
#     box_colors = box_colors.reshape(-1, *box_colors.shape[2:])

#     return box_outlines, box_colors


def _get_bounding_box_outlines(
    scene: SceneAPI, iteration: int, initial_ego_state: EgoStateSE3
) -> npt.NDArray[np.float64]:
    ego_vehicle_state = scene.get_ego_state_at_iteration(iteration)
    box_detections = scene.get_box_detections_at_iteration(iteration)

    # Load boxes to visualize, including ego vehicle at the last position
    boxes = [bd.bounding_box_se3 for bd in box_detections.box_detections] + [ego_vehicle_state.bounding_box_se3]
    boxes_labels = [bd.metadata.default_label for bd in box_detections.box_detections] + [DefaultBoxDetectionLabel.EGO]

    # Create lines for all boxes
    box_se3_array = np.array([box.array for box in boxes])
    box_se3_array[..., BoundingBoxSE3Index.XYZ] -= initial_ego_state.center_se3.array[PoseSE3Index.XYZ]
    box_corners_array = bbse3_array_to_corners_array(box_se3_array)
    box_outlines = corners_array_to_edge_lines(box_corners_array)

    # Create colors for all boxes
    box_colors = np.zeros(box_outlines.shape, dtype=np.float32)
    for i, box_label in enumerate(boxes_labels):
        box_colors[i, ...] = BOX_DETECTION_CONFIG[box_label].fill_color.rgb_norm

    box_outlines = box_outlines.reshape(-1, *box_outlines.shape[2:])
    box_colors = box_colors.reshape(-1, *box_colors.shape[2:])

    return box_outlines, box_colors, box_se3_array
