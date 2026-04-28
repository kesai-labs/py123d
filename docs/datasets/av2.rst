.. _av2_sensor:

Argoverse 2 - Sensor
--------------------

Argoverse 2 (AV2) is a collection of three datasets.
The *Sensor Dataset* includes 1000 logs of ~20 second duration, including multi-view cameras, Lidar point clouds, maps, ego-vehicle data, and bounding boxes.
This dataset is intended to train 3D perception models for autonomous vehicles.

.. dropdown:: Overview
  :open:

  .. list-table::
    :header-rows: 0
    :widths: 20 60

    * -
      -
    * - :octicon:`file` Paper
      -
        `Argoverse 2: Next Generation Datasets for Self-Driving Perception and Forecasting <https://arxiv.org/abs/2301.00493>`_

    * - :octicon:`download` Download
      - `argoverse.org <https://www.argoverse.org/>`_
    * - :octicon:`mark-github` Code
      - `argoverse/av2-api <https://github.com/argoverse/av2-api>`_
    * - :octicon:`law` License
      -
        `CC BY-NC-SA 4.0 <https://creativecommons.org/licenses/by-nc-sa/4.0/>`_

        `Argoverse Terms of Use <https://www.argoverse.org/about.html#terms-of-use>`_

        MIT License
    * - :octicon:`database` Available splits
      - ``av2-sensor_train``, ``av2-sensor_val``, ``av2-sensor_test``


Available Modalities
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 5 65

   * - **Name**
     - **Available**
     - **Description**
   * - Ego Vehicle
     - ✓
     - State of the ego vehicle, including poses, and vehicle parameters, see :class:`~py123d.datatypes.vehicle_state.EgoStateSE3`.
   * - Map
     - (✓)
     - The HD-Maps are in 3D, but may have artifacts due to polyline to polygon conversion (see below). For more information, see :class:`~py123d.api.MapAPI`.
   * - Bounding Boxes
     - ✓
     - The bounding boxes are available with the :class:`~py123d.parser.registry.AV2SensorBoxDetectionLabel`. For more information, :class:`~py123d.datatypes.detections.BoxDetectionsSE3`.
   * - Traffic Lights
     - X
     - n/a
   * - Cameras
     - ✓
     -
      Includes 9 cameras, see :class:`~py123d.datatypes.sensors.Camera`:

      - :class:`~py123d.datatypes.sensors.CameraID.PCAM_F0` (ring_front_center)
      - :class:`~py123d.datatypes.sensors.CameraID.PCAM_R0` (ring_front_right)
      - :class:`~py123d.datatypes.sensors.CameraID.PCAM_R1` (ring_side_right)
      - :class:`~py123d.datatypes.sensors.CameraID.PCAM_R2` (ring_rear_right)
      - :class:`~py123d.datatypes.sensors.CameraID.PCAM_L0` (ring_front_left)
      - :class:`~py123d.datatypes.sensors.CameraID.PCAM_L1` (ring_side_left)
      - :class:`~py123d.datatypes.sensors.CameraID.PCAM_L2` (ring_rear_left)
      - :class:`~py123d.datatypes.sensors.CameraID.PCAM_STEREO_R` (stereo_front_right)
      - :class:`~py123d.datatypes.sensors.CameraID.PCAM_STEREO_L` (stereo_front_left)

   * - Lidars
     - ✓
     -
      Includes 2 Lidars, see :class:`~py123d.datatypes.sensors.Lidar`:

      - :class:`~py123d.datatypes.sensors.LidarID.LIDAR_TOP` (top up)
      - :class:`~py123d.datatypes.sensors.LidarID.LIDAR_DOWN` (top down)


.. dropdown:: Dataset Specific

  .. autoclass:: py123d.parser.registry.AV2SensorBoxDetectionLabel
    :members:
    :no-index:
    :no-inherited-members:


Installation
~~~~~~~~~~~~

The AV2 downloader uses ``boto3`` to pull from the public Argoverse S3 bucket. Install
the extra:

.. tab-set::

  .. tab-item:: PyPI

    .. code-block:: bash

      pip install py123d[av2]

  .. tab-item:: Source

    .. code-block:: bash

      pip install -e .[av2]

``boto3`` is only required to *download* the dataset. Parsing a locally-downloaded
dataset needs no extra dependencies beyond the standard ``py123d`` install.


Download
~~~~~~~~

The AV2 Sensor dataset lives on a publicly-readable AWS S3 bucket
(``s3://argoverse/datasets/av2/sensor/``). No AWS credentials are required.
Downloads run through the unified ``py123d-download`` CLI:

.. code-block:: bash

  export AV2_DATA_ROOT=/path/to/argoverse

  # Download a 5-log subset of the validation split (~1.25 GB) to $AV2_DATA_ROOT
  py123d-download dataset=av2-sensor \
      'dataset.downloader.splits=[av2-sensor_val]' \
      dataset.downloader.num_logs=5

  # Or the full dataset (~250 GB across 1000 logs)
  py123d-download dataset=av2-sensor

  # Preview the plan without downloading
  py123d-download dataset=av2-sensor \
      dataset.downloader.num_logs=3 \
      dataset.downloader.dry_run=true

The downloaded dataset has the following per-log structure:

.. code-block:: none

  $AV2_DATA_ROOT
  └── sensor/
      ├── train/
      │   └── 00a6ffc1-6ce9-3bc3-a060-6006e9893a1a/
      │       ├── annotations.feather
      │       ├── calibration/
      │       │   ├── egovehicle_SE3_sensor.feather
      │       │   └── intrinsics.feather
      │       ├── city_SE3_egovehicle.feather
      │       ├── map/
      │       │   └── ...
      │       └── sensors/
      │           ├── cameras/...
      │           └── lidar/...
      ├── val/
      └── test/


Conversion
~~~~~~~~~~

**Local mode** — data already downloaded to ``$AV2_DATA_ROOT``:

.. code-block:: bash

  py123d-conversion dataset=av2-sensor

.. note::
  The conversion of AV2 by default does not store sensor data in the logs, but only
  relative file paths. To change this behavior, adapt the ``av2-sensor.yaml``
  converter configuration.

**Streaming mode** — ``dataset=av2-sensor-stream`` attaches an ``Av2Downloader`` to
the parser; it fetches selected logs from S3 into a temp directory at parser
construction time and cleans up on parser GC. Useful when the full ~250 GB dataset
is too large for local disk and you only need a handful of logs for iteration:

.. code-block:: bash

  # Stream the first log of the validation split only:
  py123d-conversion dataset=av2-sensor-stream \
      dataset.parser.downloader.num_logs=1 \
      'dataset.parser.splits=[av2-sensor_val]'

  # Stream specific log UUIDs:
  py123d-conversion dataset=av2-sensor-stream \
      'dataset.parser.downloader.log_ids={av2-sensor_val: [00a6ffc1-6ce9-3bc3-a060-6006e9893a1a]}'

  # Persist downloads under a dedicated cache dir instead of a tempdir:
  py123d-conversion dataset=av2-sensor-stream \
      dataset.parser.downloader.num_logs=1 \
      dataset.parser.downloader.output_dir=/mnt/scratch/av2_sensor_cache

.. warning::
  Each AV2 Sensor log is ~250 MB (~1000 objects: annotations + calibration + map +
  per-camera JPEGs + per-lidar feathers). Even small ``num_logs`` values imply
  multi-hundred-MB of download traffic.

.. note::
  The streaming variant overrides the default ``log_writer_config`` to force
  self-contained sensor payloads (``camera_store_option: jpeg_binary``,
  ``lidar_store_option: binary``) since the source temp directory is deleted when
  the parser is garbage collected.

Dataset Issues
~~~~~~~~~~~~~~

- **Ego Vehicle:** The vehicle parameters are partially estimated and may be subject to inaccuracies.


Citation
~~~~~~~~

If you use this dataset in your research, please cite:

.. code-block:: bibtex

  @article{Wilson2021NEURIPS,
    author = {Benjamin Wilson and William Qi and Tanmay Agarwal and John Lambert and Jagjeet Singh and Siddhesh Khandelwal and Bowen Pan and Ratnesh Kumar and Andrew Hartnett and Jhony Kaesemodel Pontes and Deva Ramanan and Peter Carr and James Hays},
    title = {Argoverse 2: Next Generation Datasets for Self-Driving Perception and Forecasting},
    booktitle = {Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks (NeurIPS Datasets and Benchmarks 2021)},
    year = {2021}
  }
