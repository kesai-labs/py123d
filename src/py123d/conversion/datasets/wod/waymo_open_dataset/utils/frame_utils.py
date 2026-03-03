# Vendored from waymo-open-dataset v1.6.7
# https://github.com/waymo-research/waymo-open-dataset
# Copyright 2019 The Waymo Open Dataset Authors. Apache License 2.0.
# Modifications: import paths rewritten for vendoring.
"""Utils for Frame protos."""
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf

from ..protos import dataset_pb2
from . import range_image_utils
from . import transform_utils

RangeImages = Dict['dataset_pb2.LaserName.Name', List[dataset_pb2.MatrixFloat]]
CameraProjections = Dict['dataset_pb2.LaserName.Name',
                         List[dataset_pb2.MatrixInt32]]
SegmentationLabels = Dict['dataset_pb2.LaserName.Name',
                          List[dataset_pb2.MatrixInt32]]
ParsedFrame = Tuple[RangeImages, CameraProjections, SegmentationLabels,
                    dataset_pb2.MatrixFloat]


def parse_range_image_and_camera_projection(
    frame: dataset_pb2.Frame) -> ParsedFrame:
  """Parse range images and camera projections given a frame.

  Args:
    frame: open dataset frame proto

  Returns:
    range_images: A dict of {laser_name,
      [range_image_first_return, range_image_second_return]}.
    camera_projections: A dict of {laser_name,
      [camera_projection_from_first_return,
      camera_projection_from_second_return]}.
    seg_labels: segmentation labels, a dict of {laser_name,
      [seg_label_first_return, seg_label_second_return]}
    range_image_top_pose: range image pixel pose for top lidar.
  """
  range_images = {}
  camera_projections = {}
  seg_labels = {}
  range_image_top_pose: dataset_pb2.MatrixFloat = dataset_pb2.MatrixFloat()
  for laser in frame.lasers:
    if len(laser.ri_return1.range_image_compressed) > 0:  # pylint: disable=g-explicit-length-test
      range_image_str_tensor = tf.io.decode_compressed(
          laser.ri_return1.range_image_compressed, 'ZLIB')
      ri = dataset_pb2.MatrixFloat()
      ri.ParseFromString(bytearray(range_image_str_tensor.numpy()))
      range_images[laser.name] = [ri]

      if laser.name == dataset_pb2.LaserName.TOP:
        range_image_top_pose_str_tensor = tf.io.decode_compressed(
            laser.ri_return1.range_image_pose_compressed, 'ZLIB')
        range_image_top_pose = dataset_pb2.MatrixFloat()
        range_image_top_pose.ParseFromString(
            bytearray(range_image_top_pose_str_tensor.numpy()))

      camera_projection_str_tensor = tf.io.decode_compressed(
          laser.ri_return1.camera_projection_compressed, 'ZLIB')
      cp = dataset_pb2.MatrixInt32()
      cp.ParseFromString(bytearray(camera_projection_str_tensor.numpy()))
      camera_projections[laser.name] = [cp]

      if len(laser.ri_return1.segmentation_label_compressed) > 0:  # pylint: disable=g-explicit-length-test
        seg_label_str_tensor = tf.io.decode_compressed(
            laser.ri_return1.segmentation_label_compressed, 'ZLIB')
        seg_label = dataset_pb2.MatrixInt32()
        seg_label.ParseFromString(bytearray(seg_label_str_tensor.numpy()))
        seg_labels[laser.name] = [seg_label]
    if len(laser.ri_return2.range_image_compressed) > 0:  # pylint: disable=g-explicit-length-test
      range_image_str_tensor = tf.io.decode_compressed(
          laser.ri_return2.range_image_compressed, 'ZLIB')
      ri = dataset_pb2.MatrixFloat()
      ri.ParseFromString(bytearray(range_image_str_tensor.numpy()))
      range_images[laser.name].append(ri)

      camera_projection_str_tensor = tf.io.decode_compressed(
          laser.ri_return2.camera_projection_compressed, 'ZLIB')
      cp = dataset_pb2.MatrixInt32()
      cp.ParseFromString(bytearray(camera_projection_str_tensor.numpy()))
      camera_projections[laser.name].append(cp)

      if len(laser.ri_return2.segmentation_label_compressed) > 0:  # pylint: disable=g-explicit-length-test
        seg_label_str_tensor = tf.io.decode_compressed(
            laser.ri_return2.segmentation_label_compressed, 'ZLIB')
        seg_label = dataset_pb2.MatrixInt32()
        seg_label.ParseFromString(bytearray(seg_label_str_tensor.numpy()))
        seg_labels[laser.name].append(seg_label)
  return range_images, camera_projections, seg_labels, range_image_top_pose


def convert_range_image_to_cartesian(frame,
                                     range_images,
                                     range_image_top_pose,
                                     ri_index=0,
                                     keep_polar_features=False):
  """Convert range images from polar coordinates to Cartesian coordinates."""
  cartesian_range_images = {}
  frame_pose = tf.convert_to_tensor(
      value=np.reshape(np.array(frame.pose.transform), [4, 4]))

  # [H, W, 6]
  range_image_top_pose_tensor = tf.reshape(
      tf.convert_to_tensor(value=range_image_top_pose.data),
      range_image_top_pose.shape.dims)
  # [H, W, 3, 3]
  range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
      range_image_top_pose_tensor[..., 0], range_image_top_pose_tensor[..., 1],
      range_image_top_pose_tensor[..., 2])
  range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
  range_image_top_pose_tensor = transform_utils.get_transform(
      range_image_top_pose_tensor_rotation,
      range_image_top_pose_tensor_translation)

  for c in frame.context.laser_calibrations:
    range_image = range_images[c.name][ri_index]
    if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
      beam_inclinations = range_image_utils.compute_inclination(
          tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
          height=range_image.shape.dims[0])
    else:
      beam_inclinations = tf.constant(c.beam_inclinations)

    beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
    extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

    range_image_tensor = tf.reshape(
        tf.convert_to_tensor(value=range_image.data), range_image.shape.dims)
    pixel_pose_local = None
    frame_pose_local = None
    if c.name == dataset_pb2.LaserName.TOP:
      pixel_pose_local = range_image_top_pose_tensor
      pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
      frame_pose_local = tf.expand_dims(frame_pose, axis=0)
    range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
        tf.expand_dims(range_image_tensor[..., 0], axis=0),
        tf.expand_dims(extrinsic, axis=0),
        tf.expand_dims(tf.convert_to_tensor(value=beam_inclinations), axis=0),
        pixel_pose=pixel_pose_local,
        frame_pose=frame_pose_local)

    range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)

    if keep_polar_features:
      range_image_cartesian = tf.concat(
          [range_image_tensor[..., 0:3], range_image_cartesian], axis=-1)

    cartesian_range_images[c.name] = range_image_cartesian

  return cartesian_range_images


def convert_range_image_to_point_cloud(frame,
                                       range_images,
                                       camera_projections,
                                       range_image_top_pose,
                                       ri_index=0,
                                       keep_polar_features=False):
  """Convert range images to point cloud."""
  calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
  points = []
  cp_points = []

  cartesian_range_images = convert_range_image_to_cartesian(
      frame, range_images, range_image_top_pose, ri_index, keep_polar_features)

  for c in calibrations:
    range_image = range_images[c.name][ri_index]
    range_image_tensor = tf.reshape(
        tf.convert_to_tensor(value=range_image.data), range_image.shape.dims)
    range_image_mask = range_image_tensor[..., 0] > 0

    range_image_cartesian = cartesian_range_images[c.name]
    points_tensor = tf.gather_nd(range_image_cartesian,
                                 tf.compat.v1.where(range_image_mask))

    cp = camera_projections[c.name][ri_index]
    cp_tensor = tf.reshape(tf.convert_to_tensor(value=cp.data), cp.shape.dims)
    cp_points_tensor = tf.gather_nd(cp_tensor,
                                    tf.compat.v1.where(range_image_mask))
    points.append(points_tensor.numpy())
    cp_points.append(cp_points_tensor.numpy())

  return points, cp_points


def convert_frame_to_dict(frame: dataset_pb2.Frame):
  """Convert the frame proto into a dict of numpy arrays."""
  range_images, camera_projection_protos, _, range_image_top_pose = (
      parse_range_image_and_camera_projection(frame))
  first_return_cartesian_range_images = convert_range_image_to_cartesian(
      frame,
      range_images,
      range_image_top_pose,
      ri_index=0,
      keep_polar_features=True)
  second_return_cartesian_range_images = convert_range_image_to_cartesian(
      frame,
      range_images,
      range_image_top_pose,
      ri_index=1,
      keep_polar_features=True)

  data_dict = {}

  for c in frame.context.laser_calibrations:
    laser_name_str = dataset_pb2.LaserName.Name.Name(c.name)

    beam_inclination_key = f'{laser_name_str}_BEAM_INCLINATION'
    if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
      data_dict[beam_inclination_key] = range_image_utils.compute_inclination(
          tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
          height=range_images[c.name][0].shape.dims[0]).numpy()
    else:
      data_dict[beam_inclination_key] = np.array(c.beam_inclinations,
                                                 np.float32)

    data_dict[f'{laser_name_str}_LIDAR_EXTRINSIC'] = np.reshape(
        np.array(c.extrinsic.transform, np.float32), [4, 4])

    data_dict[f'{laser_name_str}_RANGE_IMAGE_FIRST_RETURN'] = (
        first_return_cartesian_range_images[c.name].numpy())
    data_dict[f'{laser_name_str}_RANGE_IMAGE_SECOND_RETURN'] = (
        second_return_cartesian_range_images[c.name].numpy())

    first_return_cp = camera_projection_protos[c.name][0]
    data_dict[f'{laser_name_str}_CAM_PROJ_FIRST_RETURN'] = np.reshape(
        np.array(first_return_cp.data), first_return_cp.shape.dims)

    second_return_cp = camera_projection_protos[c.name][1]
    data_dict[f'{laser_name_str}_CAM_PROJ_SECOND_RETURN'] = np.reshape(
        np.array(second_return_cp.data), second_return_cp.shape.dims)

  for im in frame.images:
    cam_name_str = dataset_pb2.CameraName.Name.Name(im.name)
    data_dict[f'{cam_name_str}_IMAGE'] = tf.io.decode_jpeg(im.image).numpy()
    data_dict[f'{cam_name_str}_SDC_VELOCITY'] = np.array([
        im.velocity.v_x, im.velocity.v_y, im.velocity.v_z, im.velocity.w_x,
        im.velocity.w_y, im.velocity.w_z
    ], np.float32)
    data_dict[f'{cam_name_str}_POSE'] = np.reshape(
        np.array(im.pose.transform, np.float32), (4, 4))
    data_dict[f'{cam_name_str}_POSE_TIMESTAMP'] = np.array(
        im.pose_timestamp, np.float32)
    data_dict[f'{cam_name_str}_ROLLING_SHUTTER_DURATION'] = np.array(im.shutter)
    data_dict[f'{cam_name_str}_CAMERA_TRIGGER_TIME'] = np.array(
        im.camera_trigger_time)
    data_dict[f'{cam_name_str}_CAMERA_READOUT_DONE_TIME'] = np.array(
        im.camera_readout_done_time)

  for c in frame.context.camera_calibrations:
    cam_name_str = dataset_pb2.CameraName.Name.Name(c.name)
    data_dict[f'{cam_name_str}_INTRINSIC'] = np.array(c.intrinsic, np.float32)
    data_dict[f'{cam_name_str}_EXTRINSIC'] = np.reshape(
        np.array(c.extrinsic.transform, np.float32), [4, 4])
    data_dict[f'{cam_name_str}_WIDTH'] = np.array(c.width)
    data_dict[f'{cam_name_str}_HEIGHT'] = np.array(c.height)
    data_dict[f'{cam_name_str}_ROLLING_SHUTTER_DIRECTION'] = np.array(
        c.rolling_shutter_direction)

  data_dict['TOP_RANGE_IMAGE_POSE'] = np.reshape(
      np.array(range_image_top_pose.data, np.float32),
      range_image_top_pose.shape.dims)

  data_dict['POSE'] = np.reshape(
      np.array(frame.pose.transform, np.float32), (4, 4))
  data_dict['TIMESTAMP'] = np.array(frame.timestamp_micros)

  return data_dict
