Conventions
===========

This page documents the coordinate systems, data representations, and naming
conventions used throughout 123D. All source datasets are converted to these
unified conventions during the conversion step so that downstream code can
work with a single, consistent representation.


Coordinate System
-----------------

123D uses a **right-handed coordinate system** following the
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

This convention applies to coordinates of the ego vehicle, and other agents or objects.


Rotation Conventions
--------------------

Quaternions
^^^^^^^^^^^

We use quaternions to store or regularly express rotations. Quaternions use the **scalar-first** convention:

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

Euler Angles
^^^^^^^^^^^^

We have some utilities for working with Euler angles. Please be careful of rotations orders. We follow the **Tait-Bryan ZYX intrinsic** convention
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



Poses: SE(2) and SE(3)
-----------------------

SE(2): 2D Pose
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

SE(3): 3D Pose
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
