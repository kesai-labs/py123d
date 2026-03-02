from typing import Dict, Optional

import numpy as np
import trimesh
import viser

from py123d.api import SceneAPI
from py123d.datatypes.map_objects.base_map_objects import BaseMapSurfaceObject
from py123d.datatypes.map_objects.map_layer_types import MapLayer
from py123d.datatypes.vehicle_state.ego_state import EgoStateSE3
from py123d.geometry import Point3D, Point3DIndex
from py123d.visualization.color.default import MAP_SURFACE_CONFIG
from py123d.visualization.viser.viser_config import ViserConfig

last_query_position: Optional[Point3D] = None


def add_map_to_viser_server(
    scene: SceneAPI,
    iteration: int,
    initial_ego_state: EgoStateSE3,
    viser_server: viser.ViserServer,
    viser_config: ViserConfig,
    map_handles: Dict[MapLayer, viser.GlbHandle],
) -> None:
    global last_query_position  # noqa: PLW0603

    if viser_config.map_visible:
        map_trimesh_dict: Optional[Dict[MapLayer, trimesh.Trimesh]] = None

        if len(map_handles) == 0 or viser_config._force_map_update:
            current_ego_state = initial_ego_state
            map_trimesh_dict = _get_map_trimesh_dict(
                scene,
                initial_ego_state,
                current_ego_state,
                viser_config,
            )
            last_query_position = current_ego_state.center_se3.point_3d
            viser_config._force_map_update = False

        elif viser_config.map_requery:
            current_ego_state = scene.get_ego_state_at_iteration(iteration)
            current_position = current_ego_state.center_se3.point_3d

            if np.linalg.norm(current_position.array - last_query_position.array) > viser_config.map_radius / 2:
                last_query_position = current_position
                map_trimesh_dict = _get_map_trimesh_dict(
                    scene,
                    initial_ego_state,
                    current_ego_state,
                    viser_config,
                )

        if map_trimesh_dict is not None:
            for map_layer, mesh in map_trimesh_dict.items():
                # if map_layer in map_handles:
                #     map_handles[map_layer].mesh = mesh
                # else:
                map_handles[map_layer] = viser_server.scene.add_mesh_trimesh(
                    f"/map/{map_layer.serialize()}",
                    mesh,
                    visible=True,
                )


def _get_map_trimesh_dict(
    scene: SceneAPI,
    initial_ego_state: EgoStateSE3,
    current_ego_state: Optional[EgoStateSE3],
    viser_config: ViserConfig,
) -> Dict[MapLayer, trimesh.Trimesh]:
    # Dictionary to hold the output trimesh meshes.
    output_trimesh_dict: Dict[MapLayer, trimesh.Trimesh] = {}

    # Unpack scene center for translation of map objects.
    scene_center: Point3D = initial_ego_state.center_se3.point_3d
    scene_center_array = scene_center.array
    scene_query_position = current_ego_state.center_se3.point_3d

    # Load map objects within a certain radius around the scene center.
    map_layers = [
        MapLayer.LANE_GROUP,
        MapLayer.INTERSECTION,
        MapLayer.WALKWAY,
        MapLayer.CROSSWALK,
        MapLayer.CARPARK,
        MapLayer.GENERIC_DRIVABLE,
    ]
    map_api = scene.get_map_api()
    if map_api is not None:
        map_objects_dict = map_api.get_map_objects_in_radius(
            scene_query_position.point_2d,
            radius=viser_config.map_radius,
            layers=map_layers,
        )

        # Create trimesh meshes for each map layer.
        for map_layer in map_objects_dict.keys():
            surface_meshes = []
            for map_surface in map_objects_dict[map_layer]:
                map_surface: BaseMapSurfaceObject

                trimesh_mesh = map_surface.trimesh_mesh
                trimesh_mesh.vertices -= scene_center_array

                # Adjust height of non-road surfaces to avoid z-fighting in the visualization.
                if map_layer in [
                    MapLayer.WALKWAY,
                    MapLayer.CROSSWALK,
                    MapLayer.CARPARK,
                ]:
                    trimesh_mesh.vertices[..., Point3DIndex.Z] += viser_config.map_non_road_z_offset

                # If the map does not have z-values, we place the surfaces on the ground level of the ego vehicle.
                if not scene.log_metadata.map_metadata.map_has_z:
                    trimesh_mesh.vertices[..., Point3DIndex.Z] += (
                        scene_query_position.z - initial_ego_state.vehicle_parameters.height / 2
                    )

                # Color the mesh based on the map layer type.
                trimesh_mesh.visual.face_colors = MAP_SURFACE_CONFIG[map_layer].fill_color.rgba
                surface_meshes.append(trimesh_mesh)

            output_trimesh_dict[map_layer] = trimesh.util.concatenate(surface_meshes)

    return output_trimesh_dict
