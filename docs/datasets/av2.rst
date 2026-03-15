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


Download
~~~~~~~~

You can download the Argoverse 2 Sensor dataset from the `Argoverse website <https://www.argoverse.org/>`_.
You can also use directly the dataset from AWS. For that, you first need to install `s5cmd <https://github.com/peak/s5cmd>`_:

.. code-block:: bash

  pip install s5cmd


Next, you can run the following bash script to download the dataset:

.. code-block:: bash

  DATASET_NAME="sensor" # "sensor" "lidar" "motion-forecasting" "tbv"
  AV2_SENSOR_ROOT="/path/to/argoverse/sensor"

  mkdir -p "$AV2_SENSOR_ROOT"
  s5cmd --no-sign-request cp "s3://argoverse/datasets/av2/$DATASET_NAME/*" "$AV2_SENSOR_ROOT"
  # or: s5cmd --no-sign-request sync "s3://argoverse/datasets/av2/$DATASET_NAME/*" "$AV2_SENSOR_ROOT"


The downloaded dataset should have the following structure:

.. code-block:: none

  $AV2_SENSOR_ROOT
  ├── train
  │   ├── 00a6ffc1-6ce9-3bc3-a060-6006e9893a1a
  │   │   ├── annotations.feather
  │   │   ├── calibration
  │   │   │   ├── egovehicle_SE3_sensor.feather
  │   │   │   └── intrinsics.feather
  │   │   ├── city_SE3_egovehicle.feather
  │   │   ├── map
  │   │   │   ├── 00a6ffc1-6ce9-3bc3-a060-6006e9893a1a_ground_height_surface____PIT.npy
  │   │   │   ├── 00a6ffc1-6ce9-3bc3-a060-6006e9893a1a___img_Sim2_city.json
  │   │   │   └── log_map_archive_00a6ffc1-6ce9-3bc3-a060-6006e9893a1a____PIT_city_31785.json
  │   │   └── sensors
  │   │       ├── cameras
  │   │       │   └──...
  │   │       └── lidar
  │   │           └──...
  │   └── ...
  ├── test
  │   └── ...
  └── val
      └── ...


Installation
~~~~~~~~~~~~

No additional installation steps are required beyond the standard `py123d`` installation.


Conversion
~~~~~~~~~~

To run the conversion, you either need to set the environment variable ``$AV2_DATA_ROOT`` or ``$AV2_SENSOR_ROOT``.
You can also override the file path and run:

.. code-block:: bash

  py123d-conversion datasets=["av2-sensor"] \
  dataset_paths.av2_data_root=$AV2_DATA_ROOT # optional if env variable is set


.. note::
  The conversion of AV2 by default does not store sensor data in the logs, but only relative file paths.
  To change this behavior, you need to adapt the ``av2-sensor.yaml`` converter configuration.

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
