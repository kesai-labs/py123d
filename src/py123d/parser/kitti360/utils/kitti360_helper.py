import copy
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy.linalg import polar

from py123d.geometry import BoundingBoxSE3, EulerAngles, Polyline3D, PoseSE3
from py123d.parser.kitti360.utils.kitti360_labels import BBOX_LABELS_TO_DETECTION_NAME_DICT, kittiId2label

KITTI3602NUPLAN_IMU_CALIBRATION = np.array(
    [
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.float64,
)
MAX_N: int = 1000


def local2global(semantic_id: int, instance_id: int) -> int:
    """Convert (semanticId, instanceId) pair to a single global ID."""
    global_id = semantic_id * MAX_N + instance_id
    if isinstance(global_id, np.ndarray):
        return int(global_id.astype(np.int32))
    return int(global_id)


def global2local(global_id: int) -> Tuple[int, int]:
    """Convert a global ID back to (semanticId, instanceId) pair."""
    semantic_id = global_id // MAX_N
    instance_id = global_id % MAX_N
    if isinstance(global_id, np.ndarray):
        assert isinstance(semantic_id, np.ndarray) and isinstance(instance_id, np.ndarray)
        return int(semantic_id.astype(np.int32)), int(instance_id.astype(np.int32))
    return int(semantic_id), int(instance_id)


class KITTI360Bbox3D:
    """3D bounding box parsed from KITTI-360 XML annotations."""

    # Class-level counters for sequence 0004 (reset before each parsing run)
    dynamic_global_id: int = 2000000
    static_global_id: int = 1000000

    def __init__(self) -> None:
        self.semanticId: int = -1
        self.instanceId: int = -1
        self.annotationId: int = -1
        self.globalID: int = -1

        self.start_frame: int = -1
        self.end_frame: int = -1

        # timestamp of the bbox (-1 if static)
        self.timestamp: int = -1

        self.name: str = ""
        self.label: str = ""
        self.is_dynamic: int = 0

        self.valid_frames: Dict[str, Any] = {"global_id": -1, "records": []}

        # Set by parse_vertices / parse_scale_rotation
        self.vertices_template: npt.NDArray[np.float64] = np.empty(0)
        self.vertices: npt.NDArray[np.float64] = np.empty(0)
        self.R: npt.NDArray[np.float64] = np.eye(3)
        self.T: npt.NDArray[np.float64] = np.zeros(3)
        self.Rm: npt.NDArray[np.float64] = np.eye(3)
        self.Sm: npt.NDArray[np.float64] = np.eye(3)
        self.scale: npt.NDArray[np.float64] = np.ones(3)
        self.yaw: float = 0.0
        self.pitch: float = 0.0
        self.roll: float = 0.0
        self.qw: float = 1.0
        self.qx: float = 0.0
        self.qy: float = 0.0
        self.qz: float = 0.0

    def parse_bbox(self, child: ET.Element) -> None:
        """Parse a 3D bounding box from an XML element."""
        self.timestamp = int(child.find("timestamp").text)  # type: ignore[union-attr]
        self.annotationId = int(child.find("index").text) + 1  # type: ignore[union-attr]
        self.label = child.find("label").text  # type: ignore[assignment]

        if child.find("semanticId") is None:
            self.name = BBOX_LABELS_TO_DETECTION_NAME_DICT.get(self.label, "unknown")
            self.is_dynamic = int(child.find("dynamic").text)  # type: ignore[union-attr]
            if self.is_dynamic != 0:
                dynamic_seq = int(child.find("dynamicSeq").text)  # type: ignore[union-attr]
                self.globalID = KITTI360Bbox3D.dynamic_global_id + dynamic_seq
            else:
                self.globalID = KITTI360Bbox3D.static_global_id
                KITTI360Bbox3D.static_global_id += 1
        else:
            self.start_frame = int(child.find("start_frame").text)  # type: ignore[union-attr]
            self.end_frame = int(child.find("end_frame").text)  # type: ignore[union-attr]

            semantic_id_kitti = int(child.find("semanticId").text)  # type: ignore[union-attr]
            self.semanticId = kittiId2label[semantic_id_kitti].id
            self.instanceId = int(child.find("instanceId").text)  # type: ignore[union-attr]
            self.name = kittiId2label[semantic_id_kitti].name

            self.globalID = local2global(self.semanticId, self.instanceId)

        self.valid_frames = {"global_id": self.globalID, "records": []}

        self.parse_vertices(child)
        self.parse_scale_rotation()

    def parse_vertices(self, child: ET.Element) -> None:
        """Parse transform and vertices from XML, applying the rotation and translation."""
        transform = parse_opencv_matrix(child.find("transform"))  # type: ignore[arg-type]
        R = transform[:3, :3]
        T = transform[:3, 3]
        vertices = parse_opencv_matrix(child.find("vertices"))  # type: ignore[arg-type]
        self.vertices_template = copy.deepcopy(vertices)

        vertices = np.matmul(R, vertices.transpose()).transpose() + T
        self.vertices = vertices

        self.R = R
        self.T = T

    def parse_scale_rotation(self) -> None:
        """Decompose rotation via polar decomposition and extract Euler angles / quaternion."""
        Rm, Sm = polar(self.R)
        if np.linalg.det(Rm) < 0:
            Rm[0] = -Rm[0]
        scale = np.diag(Sm)
        euler_angles = EulerAngles.from_rotation_matrix(Rm)
        obj_quaternion = euler_angles.quaternion

        self.Rm = np.array(Rm)
        self.Sm = np.array(Sm)
        self.scale = scale
        self.yaw = euler_angles.yaw
        self.pitch = euler_angles.pitch
        self.roll = euler_angles.roll
        self.qw = obj_quaternion.qw
        self.qx = obj_quaternion.qx
        self.qy = obj_quaternion.qy
        self.qz = obj_quaternion.qz

    def get_state_array(self) -> npt.NDArray[np.float64]:
        """Return the bounding box state as a flat array [cx, cy, cz, qw, qx, qy, qz, sx, sy, sz]."""
        center = PoseSE3(
            x=self.T[0],
            y=self.T[1],
            z=self.T[2],
            qw=self.qw,
            qx=self.qx,
            qy=self.qy,
            qz=self.qz,
        )
        bounding_box_se3 = BoundingBoxSE3(center, self.scale[0], self.scale[1], self.scale[2])
        return bounding_box_se3.array

    def filter_by_radius(self, ego_state_xyz: np.ndarray, valid_timestamp: List[int], radius: float = 50.0) -> None:
        """First stage of detection filtering: filter out detections by radius from ego position."""
        d = np.linalg.norm(ego_state_xyz - self.T[None, :], axis=1)
        idxs = np.where(d <= radius)[0]
        for idx in idxs:
            self.valid_frames["records"].append(
                {
                    "timestamp": valid_timestamp[idx],
                    "points_in_box": None,
                }
            )

    def box_visible_in_point_cloud(self, points: np.ndarray) -> Tuple[bool, int]:
        """Check if the bounding box is visible in the given point cloud (>40 points inside)."""
        box = self.vertices.copy()
        z_offset = 0.1
        box[:, 2] += z_offset
        O, A, B, C = box[0], box[1], box[2], box[5]
        OA = A - O
        OB = B - O
        OC = C - O
        POA = (points @ OA[..., None])[:, 0]
        POB = (points @ OB[..., None])[:, 0]
        POC = (points @ OC[..., None])[:, 0]
        mask = (
            (np.dot(O, OA) < POA)
            & (POA < np.dot(A, OA))
            & (np.dot(O, OB) < POB)
            & (POB < np.dot(B, OB))
            & (np.dot(O, OC) < POC)
            & (POC < np.dot(C, OC))
        )

        points_in_box = int(np.sum(mask))
        visible = points_in_box > 40
        return visible, points_in_box

    def load_detection_preprocess(self, records_dict: Dict[int, Any]) -> None:
        """Load preprocessed detection records for this object from the cache."""
        if self.globalID in records_dict:
            self.valid_frames["records"] = records_dict[self.globalID]["records"]


class KITTI360MapBbox3D:
    """3D bounding box for KITTI-360 map objects (roads, sidewalks, driveways)."""

    def __init__(self) -> None:
        self.id: int = -1
        self.label: str = " "
        self.vertices: Optional[Polyline3D] = None
        self.R: Optional[npt.NDArray[np.float64]] = None
        self.T: Optional[npt.NDArray[np.float64]] = None

    def parse_vertices_plane(self, child: ET.Element) -> None:
        """Parse plane vertices from XML, applying the transform."""
        transform = parse_opencv_matrix(child.find("transform"))  # type: ignore[arg-type]
        R = transform[:3, :3]
        T = transform[:3, 3]
        if child.find("transform_plane").find("rows").text == "0":  # type: ignore[union-attr]
            vertices = parse_opencv_matrix(child.find("vertices"))  # type: ignore[arg-type]
        else:
            vertices = parse_opencv_matrix(child.find("vertices_plane"))  # type: ignore[arg-type]

        vertices = np.matmul(R, vertices.transpose()).transpose() + T
        self.vertices = Polyline3D.from_array(vertices)

        self.R = R
        self.T = T

    def parse_bbox(self, child: ET.Element) -> None:
        """Parse a map bounding box from an XML element."""
        self.id = int(child.find("index").text)  # type: ignore[union-attr]
        self.label = child.find("label").text  # type: ignore[assignment]
        self.parse_vertices_plane(child)


def parse_opencv_matrix(node: ET.Element) -> npt.NDArray[np.float64]:
    """Parse an OpenCV-style matrix from an XML element with rows, cols, and data fields."""
    rows = int(node.find("rows").text)  # type: ignore[union-attr]
    cols = int(node.find("cols").text)  # type: ignore[union-attr]
    data = node.find("data").text.split(" ")  # type: ignore[union-attr]

    mat: List[float] = []
    for d in data:
        d = d.replace("\n", "")
        if len(d) < 1:
            continue
        mat.append(float(d))
    return np.reshape(mat, [rows, cols]).astype(np.float64)


def get_kitti360_lidar_extrinsic(kitti360_calibration_root: Path) -> npt.NDArray[np.float64]:
    """Compute the lidar-to-IMU extrinsic transform from KITTI-360 calibration files."""
    cam2pose_txt = kitti360_calibration_root / "calib_cam_to_pose.txt"
    if not cam2pose_txt.exists():
        raise FileNotFoundError(f"calib_cam_to_pose.txt file not found: {cam2pose_txt}")

    cam2velo_txt = kitti360_calibration_root / "calib_cam_to_velo.txt"
    if not cam2velo_txt.exists():
        raise FileNotFoundError(f"calib_cam_to_velo.txt file not found: {cam2velo_txt}")

    lastrow = np.array([0, 0, 0, 1], dtype=np.float64).reshape(1, 4)

    with open(cam2pose_txt, "r", encoding="utf-8") as f:
        image_00 = next(f)
        values = list(map(float, image_00.strip().split()[1:]))
        matrix = np.array(values).reshape(3, 4)
        cam2pose = np.concatenate((matrix, lastrow))
        cam2pose = KITTI3602NUPLAN_IMU_CALIBRATION @ cam2pose

    cam2velo = np.concatenate((np.loadtxt(cam2velo_txt).reshape(3, 4), lastrow))
    extrinsic = cam2pose @ np.linalg.inv(cam2velo)

    return extrinsic
