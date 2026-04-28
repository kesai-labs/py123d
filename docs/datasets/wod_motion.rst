Waymo Open Dataset - Motion
---------------------------

The Waymo Open Dataset (WOD) is a collective term for publicly available datasets from Waymo.

.. warning::

   The WOD-Motion dataset is not fully supported yet. You might encounter issues. This documentation is incomplete.

.. dropdown:: Overview
  :open:

  .. list-table::
    :header-rows: 0
    :widths: 20 60

    * -
      -
    * - :octicon:`file` Paper
      - `Large scale interactive motion forecasting for autonomous driving: The waymo open motion dataset <https://arxiv.org/abs/2104.10133>`_
    * - :octicon:`download` Download
      - `waymo.com/open <https://waymo.com/open/>`_
    * - :octicon:`mark-github` Code
      - `waymo-open-dataset <https://github.com/waymo-research/waymo-open-dataset>`_
    * - :octicon:`law` License
      -
        `Waymo Dataset License Agreement for Non-Commercial Use <https://waymo.com/open/terms/>`_

        Apache License 2.0 + `Code Specific Licenses <https://github.com/waymo-research/waymo-open-dataset/blob/master/LICENSE>`_

    * - :octicon:`database` Available splits
      - ``wod-motion_train``, ``wod-motion_val``, ``wod-motion_test``


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
     - The bounding boxes are available with the :class:`~py123d.parser.registry.WODMotionBoxDetectionLabel`. For more information, :class:`~py123d.datatypes.detections.BoxDetectionsSE3`.
   * - Traffic Lights
     - ✓
     - Traffic lights include the status and the lane id they are associated with, see :class:`~py123d.datatypes.detections.TrafficLightDetections`.
   * - Cameras
     - X
     - n/a
   * - Lidars
     - X
     - n/a

.. dropdown:: Dataset Specific


  .. autoclass:: py123d.parser.registry.WODMotionBoxDetectionLabel
    :members:
    :no-inherited-members:


Download
~~~~~~~~

To download the Waymo Open Dataset - Motion, please visit the `official website <https://waymo.com/open/>`_.


.. code-block:: text

  $WOD_MOTION_DATA_ROOT
    ├── testing/
    |   ├── ...
    |   ├── ...
    |   └── ...
    ├── training/
    |   ├── ...
    |   ├── ...
    |   └── ...
    └── validation/
        ├── ...
        ├── ...
        └── ...

You can add the dataset root directory to the environment variable ``WOD_MOTION_DATA_ROOT`` for easier access.

.. code-block:: bash

   export WOD_MOTION_DATA_ROOT=/path/to/wod_motion_data_root

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
~~~~~~~~~~

**Local mode** — data already downloaded to ``$WOD_MOTION_DATA_ROOT``:

.. code-block:: bash

  py123d-conversion dataset=wod-motion

**Streaming mode** — ``dataset=wod-motion-stream`` attaches a ``WODMotionDownloader`` to
the parser; it fetches selected shards from GCS into a temp directory at parser
construction time and cleans up when the parser is garbage-collected:

.. code-block:: bash

  # Stream the first shard of each default split:
  py123d-conversion dataset=wod-motion-stream \
      dataset.parser.downloader.num_shards=1

  # Stream specific shard indices (keyed by 123D split name):
  py123d-conversion dataset=wod-motion-stream \
      'dataset.parser.downloader.shard_indices={wod-motion_train: [0, 1, 2], wod-motion_val: [0]}'

  # Persist downloads under a dedicated cache dir instead of a tempdir:
  py123d-conversion dataset=wod-motion-stream \
      dataset.parser.downloader.num_shards=1 \
      dataset.parser.downloader.output_dir=/mnt/scratch/wod_motion_cache

The motion bucket is anonymously readable after license acceptance, so ADC is not strictly
required; if available it will be used automatically.

To pre-stage data outside the conversion pipeline, use ``py123d-download``:

.. code-block:: bash

  # Preview which shards would be downloaded (first 3 training shards):
  py123d-download dataset=wod-motion \
      'dataset.downloader.splits=[wod-motion_train]' \
      dataset.downloader.num_shards=3 \
      dataset.downloader.dry_run=true

  # Download the first training shard to $WOD_MOTION_DATA_ROOT:
  py123d-download dataset=wod-motion \
      'dataset.downloader.splits=[wod-motion_train]' \
      dataset.downloader.num_shards=1


Dataset Specific Issues
~~~~~~~~~~~~~~~~~~~~~~~


* **Map:** The HD-Map in Waymo has bugs ...

Citation
~~~~~~~~

If you use this dataset in your research, please cite:

.. code-block:: bibtex

  @inproceedings{Ettinger2021ICCV,
    title={Large scale interactive motion forecasting for autonomous driving: The waymo open motion dataset},
    author={Ettinger, Scott and Cheng, Shuyang and Caine, Benjamin and Liu, Chenxi and Zhao, Hang and Pradhan, Sabeek and Chai, Yuning and Sapp, Ben and Qi, Charles R and Zhou, Yin and others},
    booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
    pages={9710--9719},
    year={2021}
  }
