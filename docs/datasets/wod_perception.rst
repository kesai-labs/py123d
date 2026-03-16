Waymo Open Dataset - Perception
-------------------------------

The Waymo Open Dataset (WOD) is a collective term for publicly available datasets from Waymo.
The *Perception Dataset*, abbreviated as WOD-Perception, is a high-quality dataset targeted for perceptions tasks.
With 1150 logs each spanning 20 seconds, the dataset includes about 6.4 hours

.. dropdown:: Overview
  :open:

  .. list-table::
    :header-rows: 0
    :widths: 20 60

    * -
      -
    * - :octicon:`file` Paper
      - `Scalability in Perception for Autonomous Driving: Waymo Open Dataset <https://arxiv.org/abs/1912.04838>`_
    * - :octicon:`download` Download
      - `waymo.com/open <https://waymo.com/open/>`_
    * - :octicon:`mark-github` Code
      - `waymo-open-dataset <https://github.com/waymo-research/waymo-open-dataset>`_
    * - :octicon:`law` License
      -
        `Waymo Dataset License Agreement for Non-Commercial Use <https://waymo.com/open/terms/>`_

        Apache License 2.0 + `Code Specific Licenses <https://github.com/waymo-research/waymo-open-dataset/blob/master/LICENSE>`_

    * - :octicon:`database` Available splits
      - ``wod-perception_train``, ``wod-perception_val``, ``wod-perception_test``


Available Modalities
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 5 75

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
     - The bounding boxes are available with the :class:`~py123d.parser.registry.WODBoxDetectionLabel`. For more information, :class:`~py123d.datatypes.detections.BoxDetectionsSE3`.
   * - Traffic Lights
     - X
     - n/a
   * - Cameras
     - ✓
     -
      Includes 5 cameras, see :class:`~py123d.datatypes.sensors.Camera`:

      - :class:`~py123d.datatypes.sensors.CameraID.PCAM_F0` (front_camera)
      - :class:`~py123d.datatypes.sensors.CameraID.PCAM_L0` (front_left_camera)
      - :class:`~py123d.datatypes.sensors.CameraID.PCAM_R0` (front_right_camera)
      - :class:`~py123d.datatypes.sensors.CameraID.PCAM_L1` (left_camera)
      - :class:`~py123d.datatypes.sensors.CameraID.PCAM_R1` (right_camera)

   * - Lidars
     - ✓
     -
      Includes 5 Lidars, see :class:`~py123d.datatypes.sensors.Lidar`:

      - :class:`~py123d.datatypes.sensors.LidarID.LIDAR_TOP` (top)
      - :class:`~py123d.datatypes.sensors.LidarID.LIDAR_FRONT` (front)
      - :class:`~py123d.datatypes.sensors.LidarID.LIDAR_SIDE_LEFT` (side_left)
      - :class:`~py123d.datatypes.sensors.LidarID.LIDAR_SIDE_RIGHT` (side_right)
      - :class:`~py123d.datatypes.sensors.LidarID.LIDAR_BACK` (rear)

.. dropdown:: Dataset Specific


  .. autoclass:: py123d.parser.registry.WODPerceptionBoxDetectionLabel
    :members:
    :no-inherited-members:



Download
~~~~~~~~

To download the Waymo Open Dataset for Perception, please visit the `official website <https://waymo.com/open/>`_ and follow the instructions provided there.
You will need to register and download the Perception Dataset ``V1.4.3``.
(We currently do not support ``V2.0.1`` due to the missing maps.)
The expected directory structure after downloading and extracting the dataset is as follows:

.. code-block:: text

  $WOD_PERCEPTION_DATA_ROOT
    ├── testing/
    |   ├── segment-10084636266401282188_1120_000_1140_000_with_camera_labels.tfrecord
    |   ├── ...
    |   └── segment-9806821842001738961_4460_000_4480_000_with_camera_labels.tfrecord
    ├── training/
    |   ├── segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord
    |   ├── ...
    |   └── segment-9985243312780923024_3049_720_3069_720_with_camera_labels.tfrecord
    └── validation/
        ├── segment-10203656353524179475_7625_000_7645_000_with_camera_labels.tfrecord
        ├── ...
        └── segment-967082162553397800_5102_900_5122_900_with_camera_labels.tfrecord

You can add the dataset root directory to the environment variable ``WOD_PERCEPTION_DATA_ROOT`` for easier access.

.. code-block:: bash

   export WOD_PERCEPTION_DATA_ROOT=/path/to/wod_perception_dataset_root

Optionally, you can adjust the ``py123d/script/config/common/default_dataset_paths.yaml`` accordingly.

Installation
~~~~~~~~~~~~

The Waymo Open Dataset requires additional dependencies that are included as optional dependencies in ``py123d``. You can install them via:

.. tab-set::

  .. tab-item:: PyPI

    .. code-block:: bash

      pip install py123d[waymo]

  .. tab-item:: Source

    .. code-block:: bash

      pip install -e .[waymo]

The optional dependencies (``tensorflow-cpu`` and ``protobuf``) are only needed to convert the dataset or read from the raw TFRecord files.
After conversion, you may use any other ``py123d`` installation without these dependencies.

Conversion
~~~~~~~~~~~~

You can convert the Waymo Open Dataset for Perception by running:

.. code-block:: bash

  py123d-conversion datasets=["wod-perception"]

.. note::
  The conversion of WOD-Perception by default stores the camera images as jpegs and the Lidar point clouds as binary files in the logs.
  Thus, the logs need fairly large disk space. Reading from the raw TFRecord files is also supported, but requires the Waymo Open Dataset specific dependencies (see above) and might be slower.
  To change the default behavior, you need to adapt the ``wod-perception.yaml`` converter configuration.

Dataset Specific Issues
~~~~~~~~~~~~~~~~~~~~~~~


* **Map:** The HD-Map in Waymo has bugs ...

Citation
~~~~~~~~

If you use this dataset in your research, please cite:

.. code-block:: bibtex

  @inproceedings{Sun2020CVPR,
    title={Scalability in perception for autonomous driving: Waymo open dataset},
    author={Sun, Pei and Kretzschmar, Henrik and Dotiwalla, Xerxes and Chouard, Aurelien and Patnaik, Vijaysai and Tsui, Paul and Guo, James and Zhou, Yin and Chai, Yuning and Caine, Benjamin and others},
    booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
    pages={2446--2454},
    year={2020}
  }
