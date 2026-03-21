.. _physical_ai_av:

Physical AI AV
--------------

.. warning::

  **Experimental Dataset Support**

  The Physical AI AV dataset integration is currently **under active development** and should be considered experimental.
  Features may be incomplete, APIs may change, and unexpected bugs are possible.

  If you encounter any issues, please report them on our
  `GitHub Issues <https://github.com/py123d/py123d/issues>`_ page. Your feedback helps us improve!

The Physical AI AV dataset provides autonomous driving sensor data collected using the NVIDIA Hyperion 8 platform.
It includes 7 f-theta (fisheye) cameras at ~30 fps, a 360-degree LiDAR at ~10 Hz, auto-labeled 3D bounding box detections,
and high-rate egomotion data (67-100 Hz). The dataset features Draco-compressed LiDAR point clouds with per-point
timestamps and dual egomotion sources (real-time and offline-smoothed).


.. dropdown:: Overview
  :open:

  .. list-table::
    :header-rows: 0
    :widths: 20 60

    * -
      -
    * - :octicon:`law` License
      - Please refer to the dataset's official license terms.
    * - :octicon:`database` Available splits
      - ``physical-ai-av_train``, ``physical-ai-av_val``, ``physical-ai-av_test``


Available Modalities
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 5 70

   * - **Name**
     - **Available**
     - **Description**
   * - Ego Vehicle
     - ✓
     - State of the ego vehicle including poses, velocity, and acceleration at 67-100 Hz. Two egomotion sources are available: real-time and offline-smoothed. See :class:`~py123d.datatypes.vehicle_state.EgoStateSE3`.
   * - Map
     - X
     - Not available for this dataset.
   * - Bounding Boxes
     - ✓
     - Auto-labeled 3D bounding box detections with 10 semantic classes and track tokens. See :class:`~py123d.parser.registry.PhysicalAIAVBoxDetectionLabel` and :class:`~py123d.datatypes.detections.BoxDetectionsSE3`.
   * - Traffic Lights
     - X
     - Not available for this dataset.
   * - Cameras
     - ✓
     -
      Includes 7 f-theta (fisheye) cameras at ~30 fps, see :class:`~py123d.datatypes.sensors.Camera`:

      - :class:`~py123d.datatypes.sensors.CameraID.FTCAM_F0` (front wide, 120 fov)
      - :class:`~py123d.datatypes.sensors.CameraID.FTCAM_TELE_F0` (front tele, 30 fov)
      - :class:`~py123d.datatypes.sensors.CameraID.FTCAM_R0` (cross right, 120 fov)
      - :class:`~py123d.datatypes.sensors.CameraID.FTCAM_L0` (cross left, 120 fov)
      - :class:`~py123d.datatypes.sensors.CameraID.FTCAM_R1` (rear right, 70 fov)
      - :class:`~py123d.datatypes.sensors.CameraID.FTCAM_L1` (rear left, 70 fov)
      - :class:`~py123d.datatypes.sensors.CameraID.FTCAM_TELE_B0` (rear tele, 30 fov)

   * - Lidars
     - ✓
     -
      Includes 1 top-mounted 360-degree LiDAR, see :class:`~py123d.datatypes.sensors.Lidar`:

      - :class:`~py123d.datatypes.sensors.LidarID.LIDAR_TOP` (top 360 fov)


.. dropdown:: Dataset Specific

  .. autoclass:: py123d.parser.registry.PhysicalAIAVBoxDetectionLabel
    :members:
    :no-index:
    :no-inherited-members:


Download
~~~~~~~~

Please refer to the official Physical AI AV dataset distribution channels for download instructions.

The downloaded dataset should have the following structure:

.. code-block:: none

  $PHYSICAL_AI_AV_DATA_ROOT
  ├── clip_index.parquet
  ├── calibration/
  │   ├── camera_intrinsics/
  │   │   └── camera_intrinsics.chunk_XXXX.parquet
  │   └── sensor_extrinsics/
  │       └── sensor_extrinsics.chunk_XXXX.parquet
  ├── labels/
  │   ├── egomotion/
  │   │   └── {clip_id}.egomotion.parquet
  │   ├── egomotion.offline/
  │   │   └── {clip_id}.egomotion.offline.parquet
  │   └── obstacle.offline/
  │       └── {clip_id}.obstacle.offline.parquet
  ├── lidar/
  │   └── lidar_top_360fov/
  │       └── {clip_id}.lidar_top_360fov.parquet
  └── camera/
      ├── camera_front_wide_120fov/
      ├── camera_front_tele_30fov/
      ├── camera_cross_left_120fov/
      ├── camera_cross_right_120fov/
      ├── camera_rear_left_70fov/
      ├── camera_rear_right_70fov/
      └── camera_rear_tele_30fov/
          ├── {clip_id}.{cam_name}.mp4
          └── {clip_id}.{cam_name}.timestamps.parquet


Installation
~~~~~~~~~~~~

No additional installation steps are required beyond the standard ``py123d`` installation.


Conversion
~~~~~~~~~~

To run the conversion, you need to set the environment variable ``$PHYSICAL_AI_AV_DATA_ROOT``.
You can also override the file path directly:

.. code-block:: bash

  py123d-conversion datasets=["physical-ai-av"] \
  dataset_paths.physical_ai_av_data_root=$PHYSICAL_AI_AV_DATA_ROOT # optional if env variable is set


.. note::
  By default, the conversion stores camera data as JPEG binary and LiDAR data as IPC with LZ4 compression.
  You can adjust these options in the ``physical-ai-av.yaml`` converter configuration.

.. note::
  By default, ``max_clips`` is set to 3. Set it to ``null`` for converting all available clips:

  .. code-block:: bash

    py123d-conversion datasets=["physical-ai-av"] \
    dataset.parser.max_clips=null


Dataset Issues
~~~~~~~~~~~~~~

- **Auto-labeled detections:** Bounding box labels are auto-generated and may be noisier than manually annotated datasets.
- **No map data:** This dataset does not include HD-Map information.


Citation
~~~~~~~~

n/a
