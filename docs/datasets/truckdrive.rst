.. _truckdrive:

TruckDrive
----------

.. warning::

  **Experimental Dataset Support**

  TruckDrive support is currently experimental and may still change.
  If you run into issues, please open a bug report on
  `GitHub Issues <https://github.com/kesai-labs/py123d/issues>`_.

TruckDrive is a long-range autonomous highway driving dataset designed for heavy-truck safety,
perception, prediction, and planning research. It targets high-speed highway operation, where reliable
scene understanding hundreds of meters ahead is required for anticipatory planning and safe braking.

The py123d integration supports multi-camera and multi-lidar scene conversion, per-frame ego states,
3D bounding boxes (where available), and per-log lane-map objects.

For extensive details about the dataset contents, sensor setup, and companion tools,
refer to the official TruckDrive repository:
`torc-ai/TruckDrive <https://github.com/torc-ai/TruckDrive>`_.


.. dropdown:: Overview
  :open:

  .. list-table::
    :header-rows: 0
    :widths: 20 60

    * -
      -
    * - :octicon:`download` Download
      - `Hugging Face <https://huggingface.co/datasets/Torc-Robotics/TruckDrive>`_ (gated)
    * - :octicon:`mark-github` Code
      - ``py123d.parser.truckdrive`` (parser/downloader integration in this repository)
    * - :octicon:`law` License
      - Please refer to the dataset's official license terms.
    * - :octicon:`database` Available splits
      - ``truckdrive_train``, ``truckdrive_val``, ``truckdrive_test``


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
     - Vehicle pose and inferred dynamics from trajectory and calibration data. See :class:`~py123d.datatypes.EgoStateSE3`.
   * - Map
     - ✓
     - Per-log map objects parsed from lane-line and lane-segment annotations (including lane topology). See :class:`~py123d.datatypes.Lane` and :class:`~py123d.datatypes.RoadLine`.
   * - Bounding Boxes
     - ✓
     - 3D box detections with TruckDrive label mapping for train/val scenes. See :class:`~py123d.datatypes.BoxDetectionsSE3`.
   * - Traffic Lights
     - X
     - Not currently exposed as a dedicated detection modality.
   * - Cameras
     - ✓
     - Multi-camera rig (Leopard cameras) with calibrated pinhole intrinsics/extrinsics. See :class:`~py123d.datatypes.Camera`.
   * - Lidars
     - ✓
     - One merged AEVA stream plus three Ouster lidars. See :class:`~py123d.datatypes.Lidar`.


Download
~~~~~~~~

TruckDrive is distributed as a gated Hugging Face dataset.
After requesting access, export a token and download selected scenes with:

.. code-block:: bash

  pip install py123d[hf]
  export HF_TOKEN=hf_...

  py123d-download dataset=truckdrive \
      'dataset.downloader.scenes=[scene_28_1]'

By default, the downloader fetches camera, lidar, poses, calibrations, and annotation archives
and extracts them into the expected on-disk layout.


Installation
~~~~~~~~~~~~

The parser itself is included in ``py123d``. Install the ``hf`` extra if you want to use
the built-in Hugging Face downloader:

.. code-block:: bash

  pip install py123d[hf]


Conversion
~~~~~~~~~~

**Local mode** (already downloaded scenes):

.. code-block:: bash

  export TRUCKDRIVE_DATA_ROOT=/path/to/TruckDrive
  py123d-conversion dataset=truckdrive

  # Convert a custom scene list:
  py123d-conversion dataset=truckdrive \
      'dataset.parser.scene_names=[scene_28_1,scene_35_1]'


**Streaming mode** (download + convert in one run):

.. code-block:: bash

  export HF_TOKEN=hf_...
  py123d-conversion dataset=truckdrive-stream \
      'dataset.parser.scene_names=[scene_28_1]'

In streaming mode, scenes are downloaded to ``$PY123D_DATA_ROOT/.download/truckdrive``,
converted, then deleted.


Dataset Issues
~~~~~~~~~~~~~~

- ``truckdrive_test`` scenes currently lack the ground-truth trajectory/annotations
  needed for full log conversion, so the current parser skips test logs in
  ``get_log_parsers()``.
- Sensor frequencies and annotation completeness can vary by scene; verify assumptions for downstream training/evaluation.


Citation
~~~~~~~~

If you use TruckDrive, please cite the original dataset publication and follow the official citation instructions
from the dataset maintainers.
