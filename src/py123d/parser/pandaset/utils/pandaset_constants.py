from typing import Dict, List

from py123d.datatypes.detections.box_detections_metadata import BoxDetectionsSE3Metadata
from py123d.datatypes.sensors.lidar import LidarID, LidarMergedMetadata, LidarMetadata
from py123d.datatypes.sensors.pinhole_camera import CameraID, PinholeDistortion, PinholeIntrinsics
from py123d.datatypes.vehicle_state.ego_state_metadata import EgoStateSE3Metadata
from py123d.geometry import PoseSE3
from py123d.parser.pandaset.utils.pandaset_utils import extrinsic_to_imu
from py123d.parser.registry import PandasetBoxDetectionLabel

PANDASET_SPLITS: List[str] = ["pandaset_train", "pandaset_val", "pandaset_test"]

PANDASET_CAMERA_MAPPING: Dict[str, CameraID] = {
    "front_camera": CameraID.PCAM_F0,
    "back_camera": CameraID.PCAM_B0,
    "front_left_camera": CameraID.PCAM_L0,
    "front_right_camera": CameraID.PCAM_R0,
    "left_camera": CameraID.PCAM_L1,
    "right_camera": CameraID.PCAM_R1,
}

PANDASET_LIDAR_MAPPING: Dict[str, LidarID] = {
    "main_pandar64": LidarID.LIDAR_TOP,
    "front_gt": LidarID.LIDAR_FRONT,
}


PANDASET_BOX_DETECTION_FROM_STR: Dict[str, PandasetBoxDetectionLabel] = {
    "Animals - Bird": PandasetBoxDetectionLabel.ANIMALS_BIRD,
    "Animals - Other": PandasetBoxDetectionLabel.ANIMALS_OTHER,
    "Bicycle": PandasetBoxDetectionLabel.BICYCLE,
    "Bus": PandasetBoxDetectionLabel.BUS,
    "Car": PandasetBoxDetectionLabel.CAR,
    "Cones": PandasetBoxDetectionLabel.CONES,
    "Construction Signs": PandasetBoxDetectionLabel.CONSTRUCTION_SIGNS,
    "Emergency Vehicle": PandasetBoxDetectionLabel.EMERGENCY_VEHICLE,
    "Medium-sized Truck": PandasetBoxDetectionLabel.MEDIUM_SIZED_TRUCK,
    "Motorcycle": PandasetBoxDetectionLabel.MOTORCYCLE,
    "Motorized Scooter": PandasetBoxDetectionLabel.MOTORIZED_SCOOTER,
    "Other Vehicle - Construction Vehicle": PandasetBoxDetectionLabel.OTHER_VEHICLE_CONSTRUCTION_VEHICLE,
    "Other Vehicle - Pedicab": PandasetBoxDetectionLabel.OTHER_VEHICLE_PEDICAB,
    "Other Vehicle - Uncommon": PandasetBoxDetectionLabel.OTHER_VEHICLE_UNCOMMON,
    "Pedestrian": PandasetBoxDetectionLabel.PEDESTRIAN,
    "Pedestrian with Object": PandasetBoxDetectionLabel.PEDESTRIAN_WITH_OBJECT,
    "Personal Mobility Device": PandasetBoxDetectionLabel.PERSONAL_MOBILITY_DEVICE,
    "Pickup Truck": PandasetBoxDetectionLabel.PICKUP_TRUCK,
    "Pylons": PandasetBoxDetectionLabel.PYLONS,
    "Road Barriers": PandasetBoxDetectionLabel.ROAD_BARRIERS,
    "Rolling Containers": PandasetBoxDetectionLabel.ROLLING_CONTAINERS,
    "Semi-truck": PandasetBoxDetectionLabel.SEMI_TRUCK,
    "Signs": PandasetBoxDetectionLabel.SIGNS,
    "Temporary Construction Barriers": PandasetBoxDetectionLabel.TEMPORARY_CONSTRUCTION_BARRIERS,
    "Towed Object": PandasetBoxDetectionLabel.TOWED_OBJECT,
    "Train": PandasetBoxDetectionLabel.TRAIN,
    "Tram / Subway": PandasetBoxDetectionLabel.TRAM_SUBWAY,
}


# https://github.com/scaleapi/pandaset-devkit/blob/master/docs/camera_to_imu_se3_calibration.yaml
PANDASET_LIDAR_EXTRINSICS: Dict[str, PoseSE3] = {
    "front_gt": PoseSE3(
        x=-0.000451117754,
        y=-0.605646431446,
        z=-0.301525235176,
        qw=0.021475754959146356,
        qx=-0.002060907279494794,
        qy=0.01134678181520767,
        qz=0.9997028534282365,
    ),
    "main_pandar64": PoseSE3(x=0.0, y=0.0, z=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0),
}

# https://github.com/scaleapi/pandaset-devkit/blob/master/docs/camera_to_imu_se3_calibration.yaml
PANDASET_CAMERA_EXTRINSICS: Dict[str, PoseSE3] = {
    "back_camera": PoseSE3(
        x=-0.0004217634029916384,
        y=-0.21683144949675118,
        z=-1.0553445472201475,
        qw=0.713789231075861,
        qx=0.7003585531940812,
        qy=-0.001595758695393934,
        qz=-0.0005330311533742299,
    ),
    "front_camera": PoseSE3(
        x=0.0002585796504896516,
        y=-0.03907777167811011,
        z=-0.0440125762408362,
        qw=0.016213200031258722,
        qx=0.0030578899383849464,
        qy=0.7114721800418571,
        qz=-0.7025205466606356,
    ),
    "front_left_camera": PoseSE3(
        x=-0.25842240863267835,
        y=-0.3070654284505582,
        z=-0.9244245686318884,
        qw=0.33540022607039827,
        qx=0.3277491469609924,
        qy=-0.6283486651480494,
        qz=0.6206973014480826,
    ),
    "front_right_camera": PoseSE3(
        x=0.2546935700219631,
        y=-0.24929449717803095,
        z=-0.8686597280810242,
        qw=0.3537633879725252,
        qx=0.34931795852655334,
        qy=0.6120314641083645,
        qz=-0.6150170047424814,
    ),
    "left_camera": PoseSE3(
        x=0.23864835336611942,
        y=-0.2801448284013492,
        z=-0.5376795959387791,
        qw=0.5050391917998245,
        qx=0.49253073152800625,
        qy=-0.4989265501075421,
        qz=0.503409565706149,
    ),
    "right_camera": PoseSE3(
        x=-0.23097163411257893,
        y=-0.30843497058841024,
        z=-0.6850441215571058,
        qw=0.5087448402081216,
        qx=0.4947520981649951,
        qy=0.4977829953071897,
        qz=-0.49860920419297333,
    ),
}

# https://github.com/scaleapi/pandaset-devkit/blob/master/docs/camera_to_imu_se3_calibration.yaml
PANDASET_CAMERA_INTRINSICS: Dict[str, PinholeIntrinsics] = {
    "back_camera": PinholeIntrinsics(fx=933.4667, fy=934.6754, cx=896.4692, cy=507.3557),
    "front_camera": PinholeIntrinsics(fx=1970.0131, fy=1970.0091, cx=970.0002, cy=483.2988),
    "front_left_camera": PinholeIntrinsics(fx=929.8429, fy=930.0592, cx=972.1794, cy=508.0057),
    "front_right_camera": PinholeIntrinsics(fx=930.0407, fy=930.0324, cx=965.0525, cy=463.4161),
    "left_camera": PinholeIntrinsics(fx=930.4514, fy=930.0891, cx=991.6883, cy=541.6057),
    "right_camera": PinholeIntrinsics(fx=922.5465, fy=922.4229, cx=945.057, cy=517.575),
}

# https://github.com/scaleapi/pandaset-devkit/blob/master/docs/camera_to_imu_se3_calibration.yaml
PANDASET_CAMERA_DISTORTIONS: Dict[str, PinholeDistortion] = {
    "back_camera": PinholeDistortion.from_list([-0.1619, 0.0113, -0.00028815, -7.9827e-05, 0.0067]),
    "front_camera": PinholeDistortion.from_list([-0.5894, 0.66, 0.0011, -0.001, -1.0088]),
    "front_left_camera": PinholeDistortion.from_list([-0.165, 0.0099, -0.00075376, 5.3699e-05, 0.01]),
    "front_right_camera": PinholeDistortion.from_list([-0.1614, -0.0027, -0.00029662, -0.00028927, 0.0181]),
    "left_camera": PinholeDistortion.from_list([-0.1582, -0.0266, -0.00015221, 0.00059011, 0.0449]),
    "right_camera": PinholeDistortion.from_list([-0.1648, 0.0191, 0.0027, -8.5282e-07, -9.6983e-05]),
}


PANDASET_LOG_NAMES: List[str] = [
    "001",
    "002",
    "003",
    "004",
    "005",
    "006",
    "008",
    "011",
    "012",
    "013",
    "014",
    "015",
    "016",
    "017",
    "018",
    "019",
    "020",
    "021",
    "023",
    "024",
    "027",
    "028",
    "029",
    "030",
    "032",
    "033",
    "034",
    "035",
    "037",
    "038",
    "039",
    "040",
    "041",
    "042",
    "043",
    "044",
    "045",
    "046",
    "047",
    "048",
    "050",
    "051",
    "052",
    "053",
    "054",
    "055",
    "056",
    "057",
    "058",
    "059",
    "062",
    "063",
    "064",
    "065",
    "066",
    "067",
    "068",
    "069",
    "070",
    "071",
    "072",
    "073",
    "074",
    "077",
    "078",
    "079",
    "080",
    "084",
    "085",
    "086",
    "088",
    "089",
    "090",
    "091",
    "092",
    "093",
    "094",
    "095",
    "097",
    "098",
    "099",
    "100",
    "101",
    "102",
    "103",
    "104",
    "105",
    "106",
    "109",
    "110",
    "112",
    "113",
    "115",
    "116",
    "117",
    "119",
    "120",
    "122",
    "123",
    "124",
    "139",
    "149",
    "158",
]


# NOTE: Some parameters are available in PandaSet [1], others are estimated based on the vehicle model [2].
# [1] https://arxiv.org/pdf/2112.12610 (Figure 3 (a))
# [2] https://en.wikipedia.org/wiki/Chrysler_Pacifica_(minivan)
PANDASET_EGO_STATE_SE3_METADATA = EgoStateSE3Metadata(
    vehicle_name="pandaset_chrysler_pacifica",
    width=2.297,
    length=5.176,
    height=1.777,
    wheel_base=3.089,
    center_to_imu_se3=PoseSE3(x=1.461, y=0.0, z=0.45, qw=1.0, qx=0.0, qy=0.0, qz=0.0),
    rear_axle_to_imu_se3=PoseSE3.identity(),
)

PANDASET_BOX_DETECTIONS_SE3_METADATA = BoxDetectionsSE3Metadata(box_detection_label_class=PandasetBoxDetectionLabel)


def _build_pandaset_lidar_merged_metadata() -> LidarMergedMetadata:
    """Helper to build Pandaset lidar merged metadata."""
    lidar_metadata: Dict[LidarID, LidarMetadata] = {}
    for lidar_name, lidar_type in PANDASET_LIDAR_MAPPING.items():
        lidar_metadata[lidar_type] = LidarMetadata(
            lidar_name=lidar_name,
            lidar_id=lidar_type,
            lidar_to_imu_se3=extrinsic_to_imu(PANDASET_LIDAR_EXTRINSICS[lidar_name]),
        )
    return LidarMergedMetadata(lidar_metadata)


PANDASET_LIDAR_MERGED_METADATA = _build_pandaset_lidar_merged_metadata()
