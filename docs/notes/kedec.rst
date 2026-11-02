kedec Sensor Codec
==================

kedec is a compact storage format for the cameras and lidars of 123D logs:

* Cameras: AV1 video, one file per camera (camera store option ``"av1"``), encoded on an NVIDIA GPU (NVENC) or on
  the CPU (SVT-AV1).
* Lidars: the lidar codec ``"kedec"``, which stores point positions with a fixed largest error and every other
  per-point feature (time stamps, intensity, ring, ...) without loss. For a scan of the KE:SAI top lidar
  (RoboSense Ruby Plus, 780,641 points), it stores 0.80 bytes per point, 7.9 % of the Arrow IPC blob with LZ4
  compression of the same scan.


Converting and transcoding
--------------------------

The preset ``codec=kedec`` switches a conversion to the kedec codecs (AV1 cameras with NVENC at cq 38, kedec
lidars, cameras referenced by their nearest image in the sync table); without it, every dataset keeps its own
settings:

.. code-block:: bash

  py123d-conversion dataset=nuplan-mini codec=kedec

``py123d-transcode`` converts existing 123D logs into the kedec codecs without the raw dataset: every file except
the camera and lidar files is copied byte for byte, the cameras and lidars are written again from their stored
rows, and only the camera columns of ``sync.arrow`` are recomputed (UUIDs, time stamps and all other columns stay
identical). A log is written into ``.incomplete`` and moved into place when complete; existing logs are skipped
unless ``force=true``. ``camera_selection`` chooses the kept images (all by default, or one per grid time with
:class:`~py123d.api.scene.arrow.transcode.GridSelection`):

.. code-block:: bash

  py123d-transcode source_logs_root=/data/123D/logs target_logs_root=/data/123D_kedec/logs splits=[nuplan_mini]

With the Ray executor, each task of a conversion or transcoding that encodes with NVENC requests a share of a GPU
(``gpus_per_task: auto``: the GPUs of a node divided by its parallel tasks), so the tasks spread over the GPUs. When
``RAY_TMPDIR`` is set, a local Ray instance keeps its session files and object store there (for cluster nodes
whose ``/tmp`` and ``/dev/shm`` must stay free).


Cameras: AV1 video
------------------

.. code-block:: yaml

  log_writer_config:
    camera_store_option: "av1"
    video_codec_config:        # optional; the values shown are the defaults
      _target_: py123d.common.io.camera.video_codec_config.VideoCodecConfig
      backend: "nvenc"         # "nvenc" (GPU), "svt" (CPU) or "auto"
      frame_rate: 10
      gop_size: 10             # one key frame every 10 images
      nvenc_cq: 38
      svt_crf: 38
      svt_preset: 6

The images of a camera are written to ``{modality_key}.av1.mp4`` next to its Arrow file; the column
``{modality_key}.data`` stores the number of each image in the video, and the time stamps, poses and exposure stay
in the Arrow file. The camera's Arrow file carries the schema-metadata key ``codec`` with the encoder and its settings.

* NVENC settings: preset p7, tuning hq, variable rate with constant quality 38, rate cap 200 Mbit/s (buffer
  400 Mbit), up to 3 B-frames. NVENC encodes AV1 on NVIDIA GPUs from the RTX 40 series on (Ada Lovelace,
  Blackwell); ``backend: "auto"`` falls back to SVT-AV1 when no such encoder opens.
* JPEG sources keep their decoded YUV 4:2:0 planes, so the encoder receives exactly the JPEG's pixels (full range,
  BT.601 matrix, as in JFIF). RGB images are converted with the JFIF matrix; video sources keep their planes and
  color tags. Width and height must be even.
* Reading (``AV1Reader``, used by the camera reader): sequential reads decode one image; other reads seek to the
  previous key frame and decode at most ``gop_size - 1`` further images. The decoder is dav1d on the CPU; the
  NVIDIA decoder (``decoder="av1_cuvid"``) returns the same planes. The conversion to RGB uses the range and matrix
  of the file.
* Logs written with the older ``"mp4"`` option (``{modality_key}.mp4``) stay readable.


Camera rate and the sync table
------------------------------

A camera that captures faster than it is stored (20 or 30 images per second, stored at 10) keeps one image per
reference time, selected with :func:`py123d.common.frame_selection.select_nearest_frames`: the image nearest to the
reference time, or, with hysteresis (0.5 ms in kedec), the image one period after the previously kept image unless
the nearest image is nearer by more than the hysteresis. This keeps the intervals of the kept images regular when a
camera clock drifts against the reference. The reference times come from the dataset parser (for example the times at
which the lidar measures the camera direction), or from a uniform grid
(:func:`~py123d.common.frame_selection.uniform_reference_grid`).

The deferred sync table can reference the cameras by the nearest image instead of the first image after the
reference time, with ``direction_overrides`` of the
:class:`~py123d.api.scene.arrow.arrow_log_writer.SyncConfig`:

.. code-block:: yaml

  sync_config:
    reference_column: lidar.lidar_top.timestamp_us
    direction: "forward"
    direction_overrides:
      camera: "nearest"   # the image nearest to each row, strictly within half the row period

With one kept image per row, ``"nearest"`` also references an image captured shortly before the row's reference
time, which ``"forward"`` would assign to no row.


Lidars: enabling kedec
----------------------

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
