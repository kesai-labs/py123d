# Vendored from waymo-open-dataset v1.6.7
# https://github.com/waymo-research/waymo-open-dataset
# Copyright 2019 The Waymo Open Dataset Authors. Apache License 2.0.
# Modifications: import paths rewritten for vendoring.
"""Utils to manage geometry transforms."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

__all__ = [
    'get_yaw_rotation', 'get_yaw_rotation_2d', 'get_rotation_matrix',
    'get_transform'
]


def get_yaw_rotation(yaw, name=None):
  with tf.compat.v1.name_scope(name, 'GetYawRotation', [yaw]):
    cos_yaw = tf.cos(yaw)
    sin_yaw = tf.sin(yaw)
    ones = tf.ones_like(yaw)
    zeros = tf.zeros_like(yaw)
    return tf.stack([
        tf.stack([cos_yaw, -1.0 * sin_yaw, zeros], axis=-1),
        tf.stack([sin_yaw, cos_yaw, zeros], axis=-1),
        tf.stack([zeros, zeros, ones], axis=-1),
    ], axis=-2)


def get_yaw_rotation_2d(yaw):
  with tf.name_scope('GetYawRotation2D'):
    cos_yaw = tf.cos(yaw)
    sin_yaw = tf.sin(yaw)
    return tf.stack([
        tf.stack([cos_yaw, -1.0 * sin_yaw], axis=-1),
        tf.stack([sin_yaw, cos_yaw], axis=-1),
    ], axis=-2)


def get_rotation_matrix(roll, pitch, yaw, name=None):
  with tf.compat.v1.name_scope(name, 'GetRotationMatrix', [yaw, pitch, roll]):
    cos_roll = tf.cos(roll)
    sin_roll = tf.sin(roll)
    cos_yaw = tf.cos(yaw)
    sin_yaw = tf.sin(yaw)
    cos_pitch = tf.cos(pitch)
    sin_pitch = tf.sin(pitch)

    ones = tf.ones_like(yaw)
    zeros = tf.zeros_like(yaw)

    r_roll = tf.stack([
        tf.stack([ones, zeros, zeros], axis=-1),
        tf.stack([zeros, cos_roll, -1.0 * sin_roll], axis=-1),
        tf.stack([zeros, sin_roll, cos_roll], axis=-1),
    ], axis=-2)
    r_pitch = tf.stack([
        tf.stack([cos_pitch, zeros, sin_pitch], axis=-1),
        tf.stack([zeros, ones, zeros], axis=-1),
        tf.stack([-1.0 * sin_pitch, zeros, cos_pitch], axis=-1),
    ], axis=-2)
    r_yaw = tf.stack([
        tf.stack([cos_yaw, -1.0 * sin_yaw, zeros], axis=-1),
        tf.stack([sin_yaw, cos_yaw, zeros], axis=-1),
        tf.stack([zeros, zeros, ones], axis=-1),
    ], axis=-2)

    return tf.matmul(r_yaw, tf.matmul(r_pitch, r_roll))


def get_transform(rotation, translation):
  with tf.name_scope('GetTransform'):
    translation_n_1 = translation[..., tf.newaxis]
    transform = tf.concat([rotation, translation_n_1], axis=-1)
    last_row = tf.zeros_like(translation)
    last_row = tf.concat([last_row, tf.ones_like(last_row[..., 0:1])], axis=-1)
    transform = tf.concat([transform, last_row[..., tf.newaxis, :]], axis=-2)
    return transform
