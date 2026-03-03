# Vendored from waymo-open-dataset v1.6.7
# https://github.com/waymo-research/waymo-open-dataset
# Copyright 2019 The Waymo Open Dataset Authors. Apache License 2.0.
# Modifications: import paths rewritten for vendoring.
"""Utils to manage range images."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np

import tensorflow as tf

__all__ = [
    'encode_lidar_features', 'decode_lidar_features', 'scatter_nd_with_pool',
    'compute_range_image_polar', 'compute_range_image_cartesian',
    'build_range_image_from_point_cloud', 'build_camera_depth_image',
    'extract_point_cloud_from_range_image', 'crop_range_image',
    'compute_inclination'
]


def _combined_static_and_dynamic_shape(tensor):
  static_tensor_shape = tensor.shape.as_list()
  dynamic_tensor_shape = tf.shape(input=tensor)
  combined_shape = []
  for index, dim in enumerate(static_tensor_shape):
    if dim is not None:
      combined_shape.append(dim)
    else:
      combined_shape.append(dynamic_tensor_shape[index])
  return combined_shape


_RANGE_TO_METERS = 0.00585532144


def _encode_range(r):
  encoded_r = r / _RANGE_TO_METERS
  with tf.control_dependencies([
      tf.compat.v1.assert_non_negative(encoded_r),
      tf.compat.v1.assert_less_equal(encoded_r, math.pow(2, 16) - 1.)
  ]):
    return tf.cast(encoded_r, dtype=tf.uint16)


def _decode_range(r):
  return tf.cast(r, dtype=tf.float32) * _RANGE_TO_METERS


def _encode_intensity(intensity):
  if intensity.dtype != tf.float32:
    raise TypeError('intensity must be of type float32')
  intensity_uint32 = tf.bitcast(intensity, tf.uint32)
  intensity_uint32_shifted = tf.bitwise.right_shift(intensity_uint32, 16)
  return tf.cast(intensity_uint32_shifted, dtype=tf.uint16)


def _decode_intensity(intensity):
  if intensity.dtype != tf.uint16:
    raise TypeError('intensity must be of type uint16')
  intensity_uint32 = tf.cast(intensity, dtype=tf.uint32)
  intensity_uint32_shifted = tf.bitwise.left_shift(intensity_uint32, 16)
  return tf.bitcast(intensity_uint32_shifted, tf.float32)


def _encode_elongation(elongation):
  encoded_elongation = elongation / _RANGE_TO_METERS
  with tf.control_dependencies([
      tf.compat.v1.assert_non_negative(encoded_elongation),
      tf.compat.v1.assert_less_equal(encoded_elongation, math.pow(2, 8) - 1.)
  ]):
    return tf.cast(encoded_elongation, dtype=tf.uint8)


def _decode_elongation(elongation):
  return tf.cast(elongation, dtype=tf.float32) * _RANGE_TO_METERS


def encode_lidar_features(lidar_point_feature):
  if lidar_point_feature.dtype != tf.float32:
    raise TypeError('lidar_point_feature must be of type float32.')
  r, intensity, elongation = tf.unstack(lidar_point_feature, axis=-1)
  encoded_r = tf.cast(_encode_range(r), dtype=tf.uint32)
  encoded_intensity = tf.cast(_encode_intensity(intensity), dtype=tf.uint32)
  encoded_elongation = tf.cast(_encode_elongation(elongation), dtype=tf.uint32)
  encoded_r_shifted = tf.bitwise.left_shift(encoded_r, 16)
  encoded_intensity = tf.cast(
      tf.bitwise.bitwise_or(encoded_r_shifted, encoded_intensity),
      dtype=tf.int64)
  encoded_elongation = tf.cast(
      tf.bitwise.bitwise_or(encoded_r_shifted, encoded_elongation),
      dtype=tf.int64)
  encoded_r = tf.cast(encoded_r, dtype=tf.int64)
  return tf.stack([encoded_r, encoded_intensity, encoded_elongation], axis=-1)


def decode_lidar_features(lidar_point_feature):
  r, intensity, elongation = tf.unstack(lidar_point_feature, axis=-1)
  decoded_r = _decode_range(r)
  intensity = tf.bitwise.bitwise_and(intensity, int(0xFFFF))
  decoded_intensity = _decode_intensity(tf.cast(intensity, dtype=tf.uint16))
  elongation = tf.bitwise.bitwise_and(elongation, int(0xFF))
  decoded_elongation = _decode_elongation(tf.cast(elongation, dtype=tf.uint8))
  return tf.stack([decoded_r, decoded_intensity, decoded_elongation], axis=-1)


def scatter_nd_with_pool(index,
                         value,
                         shape,
                         pool_method=tf.math.unsorted_segment_max):
  if len(shape) != 2:
    raise ValueError('shape must be of size 2')
  height = shape[0]
  width = shape[1]
  index_encoded, idx = tf.unique(index[:, 0] * width + index[:, 1])
  value_pooled = pool_method(value, idx, tf.size(input=index_encoded))
  index_unique = tf.stack(
      [index_encoded // width,
       tf.math.mod(index_encoded, width)], axis=-1)
  shape = [height, width]
  value_shape = _combined_static_and_dynamic_shape(value)
  if len(value_shape) > 1:
    shape = shape + value_shape[1:]
  image = tf.scatter_nd(index_unique, value_pooled, shape)
  return image


def compute_range_image_polar(range_image,
                              extrinsic,
                              inclination,
                              dtype=tf.float32,
                              scope=None):
  _, height, width = _combined_static_and_dynamic_shape(range_image)
  range_image_dtype = range_image.dtype
  range_image = tf.cast(range_image, dtype=dtype)
  extrinsic = tf.cast(extrinsic, dtype=dtype)
  inclination = tf.cast(inclination, dtype=dtype)

  with tf.compat.v1.name_scope(scope, 'ComputeRangeImagePolar',
                               [range_image, extrinsic, inclination]):
    with tf.compat.v1.name_scope('Azimuth'):
      az_correction = tf.atan2(extrinsic[..., 1, 0], extrinsic[..., 0, 0])
      ratios = (tf.cast(tf.range(width, 0, -1), dtype=dtype) - .5) / tf.cast(
          width, dtype=dtype)
      azimuth = (ratios * 2. - 1.) * np.pi - tf.expand_dims(az_correction, -1)

    azimuth_tile = tf.tile(azimuth[:, tf.newaxis, :], [1, height, 1])
    inclination_tile = tf.tile(inclination[:, :, tf.newaxis], [1, 1, width])
    range_image_polar = tf.stack([azimuth_tile, inclination_tile, range_image],
                                 axis=-1)
    return tf.cast(range_image_polar, dtype=range_image_dtype)


def compute_range_image_cartesian(range_image_polar,
                                  extrinsic,
                                  pixel_pose=None,
                                  frame_pose=None,
                                  dtype=tf.float32,
                                  scope=None):
  range_image_polar_dtype = range_image_polar.dtype
  range_image_polar = tf.cast(range_image_polar, dtype=dtype)
  extrinsic = tf.cast(extrinsic, dtype=dtype)
  if pixel_pose is not None:
    pixel_pose = tf.cast(pixel_pose, dtype=dtype)
  if frame_pose is not None:
    frame_pose = tf.cast(frame_pose, dtype=dtype)

  with tf.compat.v1.name_scope(
      scope, 'ComputeRangeImageCartesian',
      [range_image_polar, extrinsic, pixel_pose, frame_pose]):
    azimuth, inclination, range_image_range = tf.unstack(
        range_image_polar, axis=-1)

    cos_azimuth = tf.cos(azimuth)
    sin_azimuth = tf.sin(azimuth)
    cos_incl = tf.cos(inclination)
    sin_incl = tf.sin(inclination)

    x = cos_azimuth * cos_incl * range_image_range
    y = sin_azimuth * cos_incl * range_image_range
    z = sin_incl * range_image_range

    range_image_points = tf.stack([x, y, z], -1)
    rotation = extrinsic[..., 0:3, 0:3]
    translation = tf.expand_dims(tf.expand_dims(extrinsic[..., 0:3, 3], 1), 1)

    range_image_points = tf.einsum('bkr,bijr->bijk', rotation,
                                   range_image_points) + translation
    if pixel_pose is not None:
      pixel_pose_rotation = pixel_pose[..., 0:3, 0:3]
      pixel_pose_translation = pixel_pose[..., 0:3, 3]
      range_image_points = tf.einsum(
          'bhwij,bhwj->bhwi', pixel_pose_rotation,
          range_image_points) + pixel_pose_translation
      if frame_pose is None:
        raise ValueError('frame_pose must be set when pixel_pose is set.')
      world_to_vehicle = tf.linalg.inv(frame_pose)
      world_to_vehicle_rotation = world_to_vehicle[:, 0:3, 0:3]
      world_to_vehicle_translation = world_to_vehicle[:, 0:3, 3]
      range_image_points = tf.einsum(
          'bij,bhwj->bhwi', world_to_vehicle_rotation,
          range_image_points) + world_to_vehicle_translation[:, tf.newaxis,
                                                             tf.newaxis, :]

    range_image_points = tf.cast(
        range_image_points, dtype=range_image_polar_dtype)
    return range_image_points


def extract_point_cloud_from_range_image(range_image,
                                         extrinsic,
                                         inclination,
                                         pixel_pose=None,
                                         frame_pose=None,
                                         dtype=tf.float32,
                                         scope=None):
  with tf.compat.v1.name_scope(
      scope, 'ExtractPointCloudFromRangeImage',
      [range_image, extrinsic, inclination, pixel_pose, frame_pose]):
    range_image_polar = compute_range_image_polar(
        range_image, extrinsic, inclination, dtype=dtype)
    range_image_cartesian = compute_range_image_cartesian(
        range_image_polar,
        extrinsic,
        pixel_pose=pixel_pose,
        frame_pose=frame_pose,
        dtype=dtype)
    return range_image_cartesian


def compute_inclination(inclination_range, height, scope=None):
  with tf.compat.v1.name_scope(scope, 'ComputeInclination',
                               [inclination_range]):
    diff = inclination_range[..., 1] - inclination_range[..., 0]
    inclination = (
        (.5 + tf.cast(tf.range(0, height), dtype=inclination_range.dtype)) /
        tf.cast(height, dtype=inclination_range.dtype) *
        tf.expand_dims(diff, axis=-1) + inclination_range[..., 0:1])
    return inclination
