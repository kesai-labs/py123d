import pytest

pytest.importorskip("tensorflow")  # extra "waymo"

from py123d.datatypes import CameraID, DefaultBoxDetectionLabel, ModalityType
from py123d.parser.registry import WODPerceptionBoxDetectionLabel
from py123d.parser.wod.wod_perception_parser import _extract_wod_perception_box_detections_2d, _lazy_import_tf_and_pb2


def test_the_camera_labels_of_a_frame_become_2d_boxes_of_the_labeled_images():
    """One set of boxes per camera image with a ``camera_labels`` entry (also an empty one), at the image's time."""
    _, dataset_pb2 = _lazy_import_tf_and_pb2()
    frame = dataset_pb2.Frame()
    for name, pose_timestamp in ((1, 10.0), (2, 10.01), (4, 10.02)):  # FRONT, FRONT_LEFT, SIDE_LEFT
        image = frame.images.add()
        image.name, image.pose_timestamp = name, pose_timestamp
    front = frame.camera_labels.add()
    front.name = 1
    label = front.labels.add()
    label.type, label.id = 2, "track-a"  # TYPE_PEDESTRIAN
    label.box.center_x, label.box.center_y, label.box.length, label.box.width = 100.0, 50.0, 20.0, 60.0
    frame.camera_labels.add().name = 2  # labeled, no boxes; SIDE_LEFT has no entry

    front_boxes, front_left_boxes = _extract_wod_perception_box_detections_2d(frame)
    assert front_boxes.modality_key == "box_detections_2d.pcam_f0"
    assert front_boxes.modality_type == ModalityType.BOX_DETECTIONS_2D
    assert front_boxes.metadata.camera_id is CameraID.PCAM_F0
    assert front_boxes.metadata.box_detection_label_class is WODPerceptionBoxDetectionLabel
    assert front_boxes.timestamp.time_us == 10_000_000
    (box,) = front_boxes
    assert box.box == (90.0, 20.0, 110.0, 80.0)  # x1, y1, x2, y2: length is the extent along x, width along y
    assert box.label is WODPerceptionBoxDetectionLabel.TYPE_PEDESTRIAN
    assert box.default_label is DefaultBoxDetectionLabel.PERSON
    assert box.track_token == "track-a" and front_boxes.origin_name(box) == "annotation"
    assert front_left_boxes.modality_key == "box_detections_2d.pcam_l0" and len(front_left_boxes) == 0
