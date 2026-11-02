kedec Sensor Codec
==================

kedec is a compact storage format for lidar scans in 123D logs. It stores point positions with a fixed largest
error and every other per-point feature (time stamps, intensity, ring, ...) without loss. For a scan of the KE:SAI
top lidar (RoboSense Ruby Plus, 780,641 points), kedec stores 0.80 bytes per point, 7.9 % of the Arrow IPC blob
with LZ4 compression of the same scan.


Enabling kedec
--------------

kedec is a lidar codec of the log writer. Set it in the ``log_writer_config`` of a conversion:

.. code-block:: yaml

  log_writer_config:
    lidar_store_option: "binary"
    lidar_codec: "kedec"
    encode_threads: 4          # threads per lidar writer that encode scans in parallel
    lidar_codec_config:        # optional; the values shown are the defaults
      _target_: py123d.common.io.lidar.point_cloud_codec_config.PointCloudCodecConfig
      kedec_mode: "auto"       # "auto", "spherical" or "grid"
      kedec_range_step_m: 0.02
      kedec_angle_step_deg: 0.008
      kedec_side_step_m: 0.02
      kedec_zstd_level: 19

Readers detect the codec of every blob from its first bytes, so logs written with other codecs stay readable and
no reader setting is needed. Each lidar file written with kedec carries the schema-metadata key ``codec``, a JSON
object with the codec name, the format version, the writing py123d version and the settings.

The functions can also be called directly:

.. code-block:: python

  from py123d.common.io.lidar.point_cloud_codec_registry import decode_point_cloud, encode_point_cloud

  blob = encode_point_cloud("kedec", xyz, features, sensor_poses={lidar_id: pose_list})
  xyz, features = decode_point_cloud(blob)


Modes
-----

**Spherical** (every lidar). The points are moved to the sensor position. Per point, the range is rounded to
2 cm and the azimuth and elevation to 0.008 degrees. The points are stored in beam order: sorted by the ring index
(feature ``channel``) when the scan stores one, else by rows of 0.1 degrees elevation, then by azimuth.

**Grid** (lidars that report a beam index ``channel`` and a column index ``column``, for example the PhysicalAI
``model_element``). The points are transformed into the sensor frame and sorted by beam, then by column. Per scan,
three tables are stored: one elevation per beam, one azimuth per column and one azimuth offset per beam. Per point,
the range is rounded to 2 cm, and the deviations of both angles from the tables are stored as sideways distances
in steps of 2 cm.

``kedec_mode: "auto"`` uses the grid mode only when both indices are stored, as non-negative integers.


Error bounds
------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Mode
     - Largest position error of a point at range :math:`r` (default steps)
   * - Spherical
     - :math:`\sqrt{(1\,\mathrm{cm})^2 + 2\,(r \cdot 0.004^\circ)^2}`: 2 cm up to 175 m, 5.04 cm at 500 m
   * - Grid
     - :math:`\sqrt{3} \cdot 1\,\mathrm{cm} = 1.73` cm (for points more than a few degrees from the poles)

Integer features are stored exactly, including the absolute point times in microseconds. Features of other dtypes
are stored as their bytes. Points with non-finite coordinates are not stored; the blob head counts them.


Point order and sensors
-----------------------

The decoded points come in the stored order (beam order, see above), not in the input order. Features are reordered
with their points. A merged cloud with sensor ids (feature ``ids``) is coded sensor by sensor, each around its own
pose from the lidar metadata (``lidar_to_imu_se3``), and decodes sensor by sensor in ascending id. The poses are
stored in the blob, so decoding requires no metadata.


Blob format (version 1)
-----------------------

.. code-block:: text

  "KEDC" | version (uint8) | reserved (uint8) | head length (uint32, little-endian) | msgpack head | parts

The head lists the number of points, the features (name, dtype, shape), and per section (sensor) the sensor id,
pose, mode, mode parameters and the parts with their sizes, followed by a CRC32 of all parts. Each integer channel
(quantized range and angles, integer features) is stored as values minus their minimum or as zigzag differences of
neighboring values, whichever compresses better, in byte planes of the smallest width, compressed with Zstandard.
The decoder raises an error for an unknown version, a truncated blob or a checksum mismatch.
