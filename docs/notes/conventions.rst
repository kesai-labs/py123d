Conventions
===========

This page documents the coordinate systems, data representations, and naming
conventions used throughout py123d. All source datasets are converted to these
unified conventions during the conversion step so that downstream code can
work with a single, consistent representation.


Coordinate System
-----------------

py123d uses a **right-handed coordinate system** following the
`ISO 8855 <https://www.iso.org/standard/51180.html>`_ ground vehicle standard:

.. code-block:: text

         Z (up)
         |
         |
         |________ Y (left)
        /
       /
      X (forward)

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Axis
     - Direction
   * - **X**
     - Forward (longitudinal)
   * - **Y**
     - Left (lateral)
   * - **Z**
     - Up (vertical)

This convention applies to **all** coordinate frames in py123d: the global
frame, the ego vehicle frame, sensor frames, and the body frames of bounding
boxes and detections.


Units
-----

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Quantity
     - Unit
   * - Distances & positions
     - Meters (m)
   * - Angles
     - Radians (rad)
   * - Velocities
     - Meters per second (m/s)
   * - Accelerations
     - Meters per second squared (m/s²)
   * - Angular velocities
     - Radians per second (rad/s)
   * - Timestamps
     - Microseconds (µs) since epoch, stored as ``int64``
   * - Camera intrinsics
     - Pixels (px)


Rotation Conventions
--------------------

Euler Angles
^^^^^^^^^^^^

Euler angles follow the **Tait-Bryan ZYX intrinsic** convention
(yaw |rarr| pitch |rarr| roll):

.. |rarr| unicode:: U+2192 .. right arrow

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Component
     - Index
     - Description
   * - Roll
     - ``EulerAnglesIndex.ROLL`` (0)
     - Rotation around the X axis
   * - Pitch
     - ``EulerAnglesIndex.PITCH`` (1)
     - Rotation around the Y axis
   * - Yaw
     - ``EulerAnglesIndex.YAW`` (2)
     - Rotation around the Z axis

The combined rotation matrix is computed as :math:`R = R_z(\text{yaw}) \cdot R_y(\text{pitch}) \cdot R_x(\text{roll})`.

Angles are **always in radians** and normalized to :math:`[-\pi, \pi]` where applicable.

Quaternions
^^^^^^^^^^^

Quaternions use the **scalar-first** (Hamilton) convention:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Component
     - Index
     - Description
   * - :math:`q_w`
     - ``QuaternionIndex.QW`` (0)
     - Scalar (real) part
   * - :math:`q_x`
     - ``QuaternionIndex.QX`` (1)
     - Imaginary i component
   * - :math:`q_y`
     - ``QuaternionIndex.QY`` (2)
     - Imaginary j component
   * - :math:`q_z`
     - ``QuaternionIndex.QZ`` (3)
     - Imaginary k component

Quaternions are always **unit quaternions** (normalized to length 1). The
identity rotation is represented as :math:`(1, 0, 0, 0)`.


Poses: SE(2) and SE(3)
-----------------------

SE(2) — 2D Pose
^^^^^^^^^^^^^^^^

A pose on the 2D plane, stored as a flat array of length 3:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Component
     - Index
     - Description
   * - x
     - ``PoseSE2Index.X`` (0)
     - Position along the forward axis
   * - y
     - ``PoseSE2Index.Y`` (1)
     - Position along the left axis
   * - yaw
     - ``PoseSE2Index.YAW`` (2)
     - Heading angle (rotation around Z)

The corresponding homogeneous matrix is a :math:`3 \times 3` transformation matrix:

.. math::

   T_{SE(2)} = \begin{bmatrix} \cos\theta & -\sin\theta & t_x \\ \sin\theta & \cos\theta & t_y \\ 0 & 0 & 1 \end{bmatrix}

SE(3) — 3D Pose
^^^^^^^^^^^^^^^^

A rigid-body pose in 3D, stored as a flat array of length 7:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Component
     - Index
     - Description
   * - x
     - ``PoseSE3Index.X`` (0)
     - Position along the forward axis
   * - y
     - ``PoseSE3Index.Y`` (1)
     - Position along the left axis
   * - z
     - ``PoseSE3Index.Z`` (2)
     - Position along the up axis
   * - :math:`q_w`
     - ``PoseSE3Index.QW`` (3)
     - Quaternion scalar part
   * - :math:`q_x`
     - ``PoseSE3Index.QX`` (4)
     - Quaternion i component
   * - :math:`q_y`
     - ``PoseSE3Index.QY`` (5)
     - Quaternion j component
   * - :math:`q_z`
     - ``PoseSE3Index.QZ`` (6)
     - Quaternion k component

The corresponding homogeneous matrix is a :math:`4 \times 4` transformation matrix:

.. math::

   T_{SE(3)} = \begin{bmatrix} R_{3 \times 3} & \mathbf{t} \\ \mathbf{0}^T & 1 \end{bmatrix}


Bounding Boxes
--------------

SE(2) Bounding Box
^^^^^^^^^^^^^^^^^^

A 2D oriented bounding box, stored as a flat array of length 5:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Component
     - Index
     - Description
   * - x
     - ``BoundingBoxSE2Index.X`` (0)
     - Center x position
   * - y
     - ``BoundingBoxSE2Index.Y`` (1)
     - Center y position
   * - yaw
     - ``BoundingBoxSE2Index.YAW`` (2)
     - Heading angle
   * - length
     - ``BoundingBoxSE2Index.LENGTH`` (3)
     - Extent along X (forward)
   * - width
     - ``BoundingBoxSE2Index.WIDTH`` (4)
     - Extent along Y (left)

SE(3) Bounding Box
^^^^^^^^^^^^^^^^^^

A 3D oriented bounding box, stored as a flat array of length 10:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Component
     - Index
     - Description
   * - x, y, z
     - ``BoundingBoxSE3Index.X`` -- ``Z`` (0--2)
     - Center position
   * - :math:`q_w, q_x, q_y, q_z`
     - ``BoundingBoxSE3Index.QW`` -- ``QZ`` (3--6)
     - Orientation quaternion
   * - length
     - ``BoundingBoxSE3Index.LENGTH`` (7)
     - Extent along X (forward)
   * - width
     - ``BoundingBoxSE3Index.WIDTH`` (8)
     - Extent along Y (left)
   * - height
     - ``BoundingBoxSE3Index.HEIGHT`` (9)
     - Extent along Z (up)

Corner Ordering
^^^^^^^^^^^^^^^

Corners are computed by applying half-extent offsets in the body frame and
follow the ISO 8855 convention. The 2D corner order (``Corners2DIndex``) is:

0. Front-left
1. Front-right
2. Back-right
3. Back-left

The 3D corner order (``Corners3DIndex``) extends this with bottom then top:

.. code-block:: text

          4------5
          |\     |\
          | \    | \
          0--\---1  \
          \  \   \  \
   length  \  7------6   height
    (x)     \ |    \ |    (z)
              \|     \|
               3------2
                width
                 (y)

0--3: bottom face (front-left, front-right, back-right, back-left)

4--7: top face (front-left, front-right, back-right, back-left)


Ego Vehicle
-----------

The ego vehicle state (``EgoStateSE3``, ``EgoStateSE2``) is defined at the
**rear axle** position. This is the standard reference point for vehicle
kinematics and the bicycle model.

Key reference frames on the vehicle are related through extrinsic transforms
stored in ``EgoMetadata``:

- ``center_to_imu_se3``: maps coordinates from the vehicle center frame to the
  IMU frame.
- ``rear_axle_to_imu_se3``: maps coordinates from the rear axle frame to the
  IMU frame. For most datasets the IMU is co-located with the rear axle, so
  this is the identity transform.

The vehicle center pose can be derived from the rear axle pose by translating
along the body-frame X axis by the ``rear_axle_to_center_longitudinal`` offset
and along the Z axis by the ``rear_axle_to_center_vertical`` offset.

Dynamic State
^^^^^^^^^^^^^

Vehicle dynamics (``DynamicStateSE3``, ``DynamicStateSE2``) are expressed in
the **body frame** of the vehicle:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Component
     - Axes
   * - Linear velocity
     - X (forward), Y (left), Z (up)
   * - Linear acceleration
     - X (forward), Y (left), Z (up)
   * - Angular velocity
     - X (roll rate), Y (pitch rate), Z (yaw rate)


Sensor Conventions
------------------

Extrinsic Calibration
^^^^^^^^^^^^^^^^^^^^^

Sensor extrinsics are stored as ``PoseSE3`` transforms that map from the
**sensor frame** to the **IMU/vehicle frame**:

- ``PinholeCameraMetadata.camera_to_imu_se3``: camera-to-IMU transform.
- ``LidarMetadata.extrinsic``: LiDAR-to-vehicle transform.

This convention means that to transform a point from sensor coordinates to the
global frame, you first apply the sensor-to-IMU extrinsic, then the ego
vehicle's global pose.

Camera
^^^^^^

Camera intrinsics follow the standard pinhole model with parameters:

- ``fx``, ``fy``: focal lengths (in pixels)
- ``cx``, ``cy``: principal point (in pixels)
- ``skew``: skew coefficient (typically 0)

The intrinsic matrix is:

.. math::

   K = \begin{bmatrix} f_x & s & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}

Distortion follows the OpenCV convention with five parameters:
:math:`k_1, k_2, p_1, p_2, k_3` (three radial, two tangential).

LiDAR
^^^^^^

LiDAR point clouds are stored as ``(N, 3)`` arrays in the **vehicle frame**
(not the sensor frame). The three columns correspond to
:class:`~py123d.geometry.Point3DIndex` ``(X, Y, Z)``.

Optional per-point features (intensity, ring/channel, timestamp, range,
elongation) are stored in a separate dictionary alongside the point cloud.


Timestamps
----------

Timestamps are represented by the ``Timestamp`` class and internally stored as
**integer microseconds** since epoch (``int64``). Conversion methods are
provided for nanoseconds, milliseconds, and seconds:

.. code-block:: python

   ts = Timestamp.from_us(1625000000000000)  # from microseconds
   ts = Timestamp.from_s(1625000000.0)       # from seconds
   ts.time_us   # -> int   (microseconds)
   ts.time_s    # -> float (seconds)


Array Representation
--------------------

Most geometric and state types in py123d are backed by flat NumPy arrays and
follow a consistent pattern:

1. An ``IntEnum`` index class (e.g. ``PoseSE3Index``) defines named offsets
   into the array.
2. Types inherit from ``ArrayMixin``, which provides ``array``, ``tolist()``,
   and ``from_array()`` / ``from_list()`` factory methods.
3. Index enums provide ``@classproperty`` slices for convenient sub-array access
   (e.g. ``PoseSE3Index.XYZ``, ``PoseSE3Index.QUATERNION``).
4. All types are **immutable** -- the underlying arrays are not modified after
   construction.

This design enables efficient batch operations via NumPy broadcasting while
keeping the API ergonomic through named accessors.

.. code-block:: python

   from py123d.geometry import PoseSE3, PoseSE3Index

   pose = PoseSE3(x=1.0, y=2.0, z=3.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)
   pose.array[PoseSE3Index.XYZ]         # np.array([1.0, 2.0, 3.0])
   pose.array[PoseSE3Index.QUATERNION]   # np.array([1.0, 0.0, 0.0, 0.0])


Transform Naming
----------------

Coordinate frame transformations follow a consistent naming pattern:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Function
     - Description
   * - ``abs_to_rel_*(origin, data)``
     - Global frame |rarr| local frame of ``origin``
   * - ``rel_to_abs_*(origin, data)``
     - Local frame of ``origin`` |rarr| global frame
   * - ``reframe_*(from_origin, to_origin, data)``
     - Local frame of ``from_origin`` |rarr| local frame of ``to_origin``

These are available for both points (``*_points_3d_array``, ``*_point_3d``) and
poses (``*_se3_array``, ``*_se3``), as well as the 2D variants (``*_se2``).

Extrinsic transforms are named ``<source>_to_<target>_se3``, indicating the
direction in which coordinates are mapped. For example, ``camera_to_imu_se3``
transforms a point from the camera frame into the IMU frame.
