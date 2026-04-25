.. _nuplan:

nuPlan
------

nuPlan is a planning simulator that comes with a large-scale dataset for autonomous vehicle research.
This dataset contains ~1282 hours of driving logs, including ego-vehicle data, HD maps, and auto-labeled bounding boxes, spanning 4 cities.
About 120 hours of nuPlan include sensor data from 8 cameras and 5 Lidars.

.. dropdown:: Overview
  :open:

  .. list-table::
    :header-rows: 0
    :widths: 20 60

    * -
      -
    * - :octicon:`file` Papers
      -
        `Towards learning-based planning: The nuplan benchmark for real-world autonomous driving <https://arxiv.org/abs/2403.04133>`_

        `nuplan: A closed-loop ml-based planning benchmark for autonomous vehicles <https://arxiv.org/abs/2106.11810>`_
    * - :octicon:`download` Download
      - `nuplan.org <https://www.nuplan.org/download>`_
    * - :octicon:`mark-github` Code
      - `nuplan-devkit <https://github.com/motional/nuplan-devkit>`_
    * - :octicon:`law` License
      -
        `CC BY-NC-SA 4.0 <https://creativecommons.org/licenses/by-nc-sa/4.0/>`_

        `nuPlan Terms of Use <https://www.nuscenes.org/terms-of-use>`_

        Apache License 2.0
    * - :octicon:`database` Available splits
      - ``nuplan_train``, ``nuplan_val``, ``nuplan_test``, ``nuplan-mini_train``, ``nuplan-mini_val``, ``nuplan-mini_test``


Available Modalities
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 5 70

   * - **Name**
     - **Available**
     - **Description**
   * - Ego Vehicle
     - ✓
     - State of the ego vehicle, including poses, dynamic state, and vehicle parameters, see :class:`~py123d.datatypes.vehicle_state.EgoStateSE3`.
   * - Map
     - (✓)
     - The HD-Maps are in 2D vector format and defined per-location. For more information, see :class:`~py123d.api.MapAPI`.
   * - Bounding Boxes
     - ✓
     - The bounding boxes are available, see :class:`~py123d.datatypes.detections.BoxDetectionsSE3`.
   * - Traffic Lights
     - ✓
     - Traffic lights include the status and the lane id they are associated with, see :class:`~py123d.datatypes.detections.TrafficLightDetections`.
   * - Cameras
     - (✓)
     -
      Subset of nuPlan includes 8x :class:`~py123d.datatypes.sensors.Camera`:

      - :class:`~py123d.datatypes.sensors.CameraID.PCAM_F0`: Front camera
      - :class:`~py123d.datatypes.sensors.CameraID.PCAM_R0`: Right front camera
      - :class:`~py123d.datatypes.sensors.CameraID.PCAM_R1`: Right middle camera
      - :class:`~py123d.datatypes.sensors.CameraID.PCAM_R2`: Right rear camera
      - :class:`~py123d.datatypes.sensors.CameraID.PCAM_L0`: Left front camera
      - :class:`~py123d.datatypes.sensors.CameraID.PCAM_L1`: Left middle camera
      - :class:`~py123d.datatypes.sensors.CameraID.PCAM_L2`: Left rear camera
      - :class:`~py123d.datatypes.sensors.CameraID.PCAM_B0`: Back camera

   * - Lidars
     - (✓)
     -
      Subset of nuPlan includes 5x :class:`~py123d.datatypes.sensors.Lidar`:

      - :class:`~py123d.datatypes.sensors.LidarID.LIDAR_TOP`: Top
      - :class:`~py123d.datatypes.sensors.LidarID.LIDAR_FRONT`: Front
      - :class:`~py123d.datatypes.sensors.LidarID.LIDAR_SIDE_LEFT`: Side left
      - :class:`~py123d.datatypes.sensors.LidarID.LIDAR_SIDE_RIGHT`: Side right
      - :class:`~py123d.datatypes.sensors.LidarID.LIDAR_BACK`: Rear

.. dropdown:: Dataset Specific

  .. autoclass:: py123d.parser.registry.NuPlanBoxDetectionLabel
    :members:
    :no-index:
    :no-inherited-members:


Download
~~~~~~~~

py123d ships an automated downloader that fetches the nuPlan archives from the
public Motional AWS bucket and extracts them into the canonical on-disk layout.
No credentials are required — the bucket is anonymously readable. Users should
still review the upstream license at
``https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/LICENSE`` before use.

.. code-block:: bash

  # Regular set — logs + maps for nuplan_{train,val,test} (~135 GB).
  py123d-download dataset=nuplan

  # Mini set — logs + maps for nuplan-mini_{train,val,test} (~11 GB).
  py123d-download dataset=nuplan \
      'dataset.downloader.splits=[nuplan-mini_train, nuplan-mini_val, nuplan-mini_test]'

  # Also fetch 8-camera shards and merged-point-cloud shards (hundreds of GB per split).
  py123d-download dataset=nuplan \
      dataset.downloader.include_cameras=true \
      dataset.downloader.include_lidar=true

  # Preview the plan without hitting the network.
  py123d-download dataset=nuplan dataset.downloader.dry_run=true

Content selection is controlled by three flags on the downloader:

* ``include_maps`` (default ``true``) — fetches ``nuplan-maps-v1.1.zip``.
  Required for HD-map conversion.
* ``include_cameras`` (default ``false``) — fetches the 8-camera ``.jpg`` shards for
  every requested split. Hundreds of GB per split — opt in explicitly.
* ``include_lidar`` (default ``false``) — fetches the merged-point-cloud shards.

Logs (``.db`` files) are always fetched for the requested splits.

.. dropdown:: Manual download (bash)

  If you prefer to click-through or run raw ``wget`` calls, the same archives are
  served at ``https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/``:

  **License**:

  .. code-block:: bash

      # NOTE: Please check the LICENSE file when downloading the nuPlan dataset
      wget https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/LICENSE

  **Maps** (required for ``nuplan`` and ``nuplan-mini``):

  .. code-block:: bash

      wget https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-maps-v1.1.zip


  **Logs**:

  .. code-block:: bash

    # 1. nuplan_train, nuplan_val (both derived from splits/trainval/)
    wget https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-v1.1_train_boston.zip
    wget https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-v1.1_train_pittsburgh.zip
    wget https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-v1.1_train_singapore.zip
    for split in {1..6}; do
        wget https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-v1.1_train_vegas_${split}.zip
    done

    # 2. nuplan_test (→ splits/test/)
    wget https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-v1.1_test.zip

    # 3. nuplan-mini_train, nuplan-mini_val, nuplan-mini_test (→ splits/mini/)
    wget https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-v1.1_mini.zip


  **Sensors**:

  .. code-block:: bash

    # 1. nuplan_train
    for split in {0..42}; do
        wget https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/sensor_blobs/train_set/nuplan-v1.1_train_camera_${split}.zip
        wget https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/sensor_blobs/train_set/nuplan-v1.1_train_lidar_${split}.zip
    done

    # 2. nuplan_val
    for split in {0..11}; do
        wget https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/sensor_blobs/val_set/nuplan-v1.1_val_camera_${split}.zip
        wget https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/sensor_blobs/val_set/nuplan-v1.1_val_lidar_${split}.zip
    done

    # 3. nuplan_test
    for split in {0..11}; do
        wget https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/sensor_blobs/test_set/nuplan-v1.1_test_camera_${split}.zip
        wget https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/sensor_blobs/test_set/nuplan-v1.1_test_lidar_${split}.zip
    done

    # 4. nuplan_mini_train, nuplan_mini_val, nuplan_mini_test
    for split in {0..8}; do
        wget https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/sensor_blobs/mini_set/nuplan-v1.1_mini_camera_${split}.zip
        wget https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/sensor_blobs/mini_set/nuplan-v1.1_mini_lidar_${split}.zip
    done

The 123D conversion expects the following directory structure:

.. code-block:: none

  $NUPLAN_DATA_ROOT
    ├── maps (or $NUPLAN_MAPS_ROOT)
    │   ├── nuplan-maps-v1.0.json
    │   ├── sg-one-north
    │   │   └── 9.17.1964
    │   │       └── map.gpkg
    │   ├── us-ma-boston
    │   │   └── 9.12.1817
    │   │       └── map.gpkg
    │   ├── us-nv-las-vegas-strip
    │   │    └── 9.15.1915
    │   │        └── map.gpkg
    │   └── us-pa-pittsburgh-hazelwood
    │       └── 9.17.1937
    │           └── map.gpkg
    └── nuplan-v1.1
        ├── splits
        │     ├── mini
        │     │    ├── 2021.05.12.22.00.38_veh-35_01008_01518.db
        │     │    ├── 2021.06.09.17.23.18_veh-38_00773_01140.db
        │     │    ├── ...
        │     │    └── 2021.10.11.08.31.07_veh-50_01750_01948.db
        │     ├── test
        │     │    └── ...
        │     └── trainval
        │          ├── 2021.05.12.22.00.38_veh-35_01008_01518.db
        │          ├── 2021.06.09.17.23.18_veh-38_00773_01140.db
        │          ├── ...
        │          └── 2021.10.11.08.31.07_veh-50_01750_01948.db
        └── sensor_blobs (or $NUPLAN_SENSOR_ROOT)
              ├── 2021.05.12.22.00.38_veh-35_01008_01518
              │    ├── CAM_F0
              │    │     ├── c082c104b7ac5a71.jpg
              │    │     ├── af380db4b4ca5d63.jpg
              │    │     ├── ...
              │    │     └── 2270fccfb44858b3.jpg
              │    ├── CAM_B0
              │    ├── CAM_L0
              │    ├── CAM_L1
              │    ├── CAM_L2
              │    ├── CAM_R0
              │    ├── CAM_R1
              │    ├── CAM_R2
              │    └──MergedPointCloud
              │         ├── 03fafcf2c0865668.pcd
              │         ├── 5aee37ce29665f1b.pcd
              │         ├── ...
              │         └── 5fe65ef6a97f5caf.pcd
              │
              ├── 2021.06.09.17.23.18_veh-38_00773_01140
              ├── ...
              └── 2021.10.11.08.31.07_veh-50_01750_01948


Lastly, you need to add the following environment variables to your ``~/.bashrc`` according to your installation paths:

.. code-block:: bash

  export NUPLAN_DATA_ROOT=/path/to/nuplan/data/root
  export NUPLAN_MAPS_ROOT=/path/to/nuplan/data/root/maps
  export NUPLAN_SENSOR_ROOT=/path/to/nuplan/data/root/nuplan-v1.1/sensor_blobs

Or configure the config ``py123d/script/config/common/default_dataset_paths.yaml`` accordingly.

Installation
~~~~~~~~~~~~

For nuPlan, you need to additionally install the `nuplan-devkit <https://github.com/motional/nuplan-devkit>`_ and optional dependencies in ``py123d``.
You can install both either from PyPI or from source:

.. tab-set::

  .. tab-item:: PyPI

    .. code-block:: bash

      pip install py123d[nuplan]
      pip install "nuplan-devkit @ git+https://github.com/motional/nuplan-devkit/@nuplan-devkit-v1.2"

  .. tab-item:: Source

    .. code-block:: bash

      pip install -e .[nuplan]
      pip install "nuplan-devkit @ git+https://github.com/motional/nuplan-devkit/@nuplan-devkit-v1.2"


Conversion
~~~~~~~~~~~~

**Local mode** — data already extracted under ``$NUPLAN_DATA_ROOT`` /
``$NUPLAN_MAPS_ROOT`` / ``$NUPLAN_SENSOR_ROOT`` (see the `Download`_ section above):

.. code-block:: bash

  py123d-conversion datasets=["nuplan"]
  # or
  py123d-conversion datasets=["nuplan-mini"]

.. note::
  The conversion of nuPlan by default does not store sensor data in the logs, but only relative file paths.
  To change this behavior, you need to adapt the ``nuplan.yaml`` or ``nuplan-mini.yaml`` converter configuration.

**Streaming mode** — materialize the configured archive subset from the public nuPlan
AWS bucket into a session-scoped temp directory at parser construction time, convert
from it, and delete the temp directory on parser destruction. The ``maps/`` and
``sensor_blobs/`` subdirectories are auto-detected (the three ``nuplan_*_root``
paths do not need to be set). No credentials required.

.. code-block:: bash

  # Smoketest (~11 GB download): mini logs + HD maps.
  py123d-conversion dataset=nuplan-mini-stream

  # Regular set (~135 GB download): full train/val/test logs + HD maps.
  py123d-conversion dataset=nuplan-stream
  py123d-conversion dataset=nuplan-stream 'dataset.parser.splits=[nuplan_val]'

  # Opt into sensor data (hundreds of GB per split).
  py123d-conversion dataset=nuplan-mini-stream \
      dataset.parser.downloader.include_cameras=true \
      dataset.parser.downloader.include_lidar=true

.. warning::
  Streaming downloads can be large even for a "small" slice — the smallest regular
  configuration is ~135 GB on the wire. Use ``dataset=nuplan-mini-stream`` (~11 GB)
  when smoke-testing the pipeline.

.. note::
  Streaming mode forces ``camera_store_option: "jpeg_binary"`` and
  ``lidar_store_option: "binary"`` — the temp directory is deleted immediately after
  the parser is garbage-collected, so any ``"path"`` references would point at
  vanished sources.


Dataset Issues
~~~~~~~~~~~~~~

* **Map:** The HD-Maps are only available in 2D.
* **Camera & Lidar:** There are synchronization issues between the sensors and the ego vehicle state.
* **Bounding Boxes:** Due to the auto-labeling process of nuPlan, some bounding boxes may be noisy.
* **Traffic Lights:** The status of the traffic lights are inferred from the vehicle movements. As such, there may be incorrect labels.

Citation
~~~~~~~~

If you use nuPlan in your research, please cite:

.. code-block:: bibtex

  @article{Karnchanachari2024ICRA,
    title={Towards learning-based planning: The nuplan benchmark for real-world autonomous driving},
    author={Karnchanachari, Napat and Geromichalos, Dimitris and Tan, Kok Seang and Li, Nanxiang and Eriksen, Christopher and Yaghoubi, Shakiba and Mehdipour, Noushin and Bernasconi, Gianmarco and Fong, Whye Kit and Guo, Yiluan and others},
    booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)},
    year={2024},
  }
  @article{Caesar2021CVPRW,
    title={nuplan: A closed-loop ml-based planning benchmark for autonomous vehicles},
    author={Caesar, Holger and Kabzan, Juraj and Tan, Kok Seang and Fong, Whye Kit and Wolff, Eric and Lang, Alex and Fletcher, Luke and Beijbom, Oscar and Omari, Sammy},
    booktitle={Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR) Workshops},
    year={2021}
  }
