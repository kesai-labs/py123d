.. _nureasoning:

nuReasoning
-----------

nuReasoning is a reasoning-centric autonomous driving dataset and benchmark focused on
long-tail scenarios.
It contains 20,000 clips of 20 seconds each (~105 hours), collected from multiple cities,
combining synchronized multi-view camera images, LiDAR point clouds, ego state, HD maps,
and traffic-signal context with human-verified reasoning annotations.
The reasoning annotations span three categories — spatial reasoning, driving decisions,
and counterfactual reasoning — and support both a Reasoning VQA benchmark and a planning
benchmark.

.. note::
  The authors release the dataset in stages. The current release holds 11,890 clips (27 TB):
  9,890 train, 1,000 validation and 1,000 test clips. The remaining clips are announced for
  after the nuReasoning Challenge 2026.


.. dropdown:: Overview
  :open:

  .. list-table::
    :header-rows: 0
    :widths: 20 60

    * -
      -
    * - :octicon:`file` Papers
      -
        `nuReasoning: A Reasoning-Centric Dataset and Benchmark for Long-Tail Autonomous Driving <https://arxiv.org/abs/2605.31572>`_

        `Project page <https://nureasoning.github.io/>`_
    * - :octicon:`download` Download
      - `Hugging Face <https://huggingface.co/datasets/nureasoning/nuReasoning>`_ (gated)
    * - :octicon:`mark-github` Code
      - `nureasoning-devkit <https://github.com/nureasoning/nureasoning-devkit>`_
    * - :octicon:`law` License
      - `nuScenes Terms of Use <https://www.nuscenes.org/terms-of-use>`_ for non-commercial use.
        Commercial use requires a separate license, see the dataset page.
    * - :octicon:`database` Available splits
      - ``nureasoning_train``, ``nureasoning_val``, ``nureasoning_test``, ``nureasoning-mini_train``
        (parts 1-3 of train, which include the former mini release)


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
     - State of the ego vehicle, including poses, dynamic state, and vehicle parameters, see :class:`~py123d.datatypes.EgoStateSE3`.
   * - Map
     - (✓)
     - The HD-Maps are in 2D vector format and stored per-log (one map per clip). Not available in the test split. For more information, see :class:`~py123d.api.MapAPI`.
   * - Bounding Boxes
     - (✓)
     - The bounding boxes are available with the :class:`~py123d.parser.registry.NureasoningBoxDetectionLabel`. Not available in the test split. For more information, see :class:`~py123d.datatypes.BoxDetectionsSE3`.
   * - Traffic Lights
     - (✓)
     - Traffic-signal states are provided per frame. Not available in the test split.
       The lane they refer to can lie outside the clip's map.
   * - Cameras
     - ✓
     -
      nuReasoning includes 8x :class:`~py123d.datatypes.Camera`:

      - :class:`~py123d.datatypes.CameraID.PCAM_F0`: front
      - :class:`~py123d.datatypes.CameraID.PCAM_B0`: back
      - :class:`~py123d.datatypes.CameraID.PCAM_L0`: front_left
      - :class:`~py123d.datatypes.CameraID.PCAM_L1`: left
      - :class:`~py123d.datatypes.CameraID.PCAM_L2`: back_left
      - :class:`~py123d.datatypes.CameraID.PCAM_R0`: front_right
      - :class:`~py123d.datatypes.CameraID.PCAM_R1`: right
      - :class:`~py123d.datatypes.CameraID.PCAM_R2`: back_right

   * - Lidars
     - (✓)
     -
      A single merged :class:`~py123d.datatypes.Lidar` point cloud fusing five sensors
      (top, front, side-left, side-right, back). Only some clips include lidar, and none
      of the test split.
      Point clouds are LZF-compressed PCD files. Their ``intensity``, ``ring`` and
      ``lidar_info`` (source sensor) channels are converted. The ``azimuth``, ``range``,
      ``is_second_return`` and ``lidar_confidence`` channels are not.
   * - Reasoning
     - (✓)
     - Human-verified spatial, decision, and counterfactual reasoning annotations, stored
       as the custom modality ``reasoning``. They are passed through as the raw nuReasoning
       reasoning JSON (no dedicated py123d datatype yet). About one frame per second
       carries them, and not all of those hold all three categories. Not available in the
       test split.
   * - Scenario
     - ✓
     - The custom modality ``scenario`` holds the route command and route path of each
       frame, the clip's scenario type, and the frame's upstream ``frame_index`` and
       token. Its ``is_key_frame`` flag marks the frame that the clip is named after
       (the ``key_frame_index`` of a test clip). The custom modality ``ego_trajectory``
       holds the ego history and future that the dataset provides per frame.

.. dropdown:: Dataset Specific

  .. autoclass:: py123d.parser.registry.NureasoningBoxDetectionLabel
    :members:
    :no-index:
    :no-inherited-members:

  .. note::
    nuReasoning does not publish an object taxonomy. The label set above holds the categories
    seen in the released annotations and the ones the devkit lists. A category outside of it is
    converted to ``OTHER_OTHER`` with a warning.

  **Test split.** Test clips come without ground truth. They cover the 10 s up to the key
  frame with cameras, ego states and the per-frame route, and have no map, boxes, traffic
  lights, lidar or reasoning annotations. The challenge questions of a clip
  (``reasoning_questions.json``) are stored as the custom modality ``reasoning_questions``
  on its key frame.

  **Frame indices.** Most clips list their key frame twice. The conversion stores it once,
  so a frame's position in the converted log can be one lower than its upstream
  ``frame_index``. The reasoning annotations and the challenge use the upstream index,
  which the ``scenario`` modality provides for every frame.

Download
~~~~~~~~

nuReasoning is a gated dataset on Hugging Face at
`nureasoning/nuReasoning <https://huggingface.co/datasets/nureasoning/nuReasoning>`_.
Request access on the dataset page, then provide a token of that account with
``export HF_TOKEN=hf_...`` or ``hf auth login``. py123d ships an automated downloader that
fetches and extracts the per-clip archives for you.

.. code-block:: bash

  export HF_TOKEN=hf_...

  # Preview the selection and its size
  py123d-download dataset=nureasoning dataset.downloader.dry_run=true

  # Download the validation split into $NUREASONING_DATA_ROOT
  py123d-download dataset=nureasoning 'dataset.downloader.splits=[nureasoning_val]'

A clip takes about 2.3 GB (1 GB in the test split), and the whole release 27 TB.

The downloader exposes several selection knobs (see
``py123d/script/config/download/dataset/nureasoning.yaml``):

* ``splits`` — e.g. ``[nureasoning_train]``, ``[nureasoning_val, nureasoning_test]`` (``null`` selects train, val and test)
* ``parts`` — e.g. ``[part_1, part_2]`` (drops the test split, which has no parts)
* ``log_names`` — explicit clip names ``<log>_<token>`` (mutually exclusive with ``num_logs``)
* ``num_logs`` — the first N clips of the selection (or N random with ``sample_random=true`` and ``seed``)
* ``max_workers`` — parallel clip download/extract workers (default ``8``)
* ``keep_archive`` — keep each ``.tar`` next to its extracted directory (default: extract then discard)

Each selected clip is downloaded as a single ``.tar`` and extracted to
``<output_dir>/<split>/[<part>/]<clip>/``. This is the layout of the devkit's
``dataset/data`` folder, so data downloaded with the devkit can be converted as well.
The 123D conversion expects the following directory structure:

.. code-block:: none

  $NUREASONING_DATA_ROOT
    ├── train/
    │   ├── part_1/
    │   │   ├── <log_name>_<keyframe_token>/
    │   │   │   ├── metadata.json
    │   │   │   ├── map.pkl
    │   │   │   ├── ego_state/
    │   │   │   │   └── <timestamp_us>.pkl
    │   │   │   ├── annotations/
    │   │   │   │   └── <timestamp_us>.pkl
    │   │   │   ├── reasoning/
    │   │   │   │   └── <timestamp_us>.json
    │   │   │   ├── cameras/
    │   │   │   │   ├── CAM_M_F/
    │   │   │   │   │   └── CAM_M_F_<timestamp_us>.jpg
    │   │   │   │   └── ...
    │   │   │   └── lidar/                     (only in some clips)
    │   │   │       └── <timestamp_us>.pcd
    │   │   └── ...
    │   ├── ...
    │   └── part_10/
    ├── validation/
    │   └── part_1/
    └── test/
        └── <log_name>_<keyframe_token>/
            ├── metadata.json
            ├── reasoning_questions.json
            ├── ego_state/
            └── cameras/

Lastly, you need to add the following environment variable to your ``~/.bashrc`` according
to your installation path:

.. code-block:: bash

  export NUREASONING_DATA_ROOT=/path/to/nureasoning/data/root

Or configure the config ``py123d/script/config/common/default_dataset_paths.yaml`` accordingly.

Installation
~~~~~~~~~~~~

Downloading and streaming nuReasoning requires the HuggingFace Hub client, included as an
optional dependency in ``py123d``. You can install it via:

.. tab-set::

  .. tab-item:: PyPI

    .. code-block:: bash

      pip install py123d[hf]

  .. tab-item:: Source

    .. code-block:: bash

      pip install -e .[hf]

Conversion
~~~~~~~~~~~~

**Local mode** — data already extracted to ``$NUREASONING_DATA_ROOT`` (see the `Download`_
section above):

.. code-block:: bash

  # The train, val and test splits; a split without a folder is skipped
  py123d-conversion dataset=nureasoning

  # Parts 1-3 of the train split
  py123d-conversion dataset=nureasoning-mini

.. note::
  The local conversion of nuReasoning by default does not store sensor data in the logs,
  but only relative file paths (``camera_store_option: "path"`` and
  ``lidar_store_option: "path"``), which are resolved against the nuReasoning sensor root
  at read time. To change this behavior, adapt the ``nureasoning.yaml`` converter
  configuration.

**Streaming mode** — download the selected clips from Hugging Face into a temp directory
at parser construction time, convert from it, and delete the temp directory afterwards.

.. code-block:: bash

  export HF_TOKEN=hf_...

  # The first five clips of the validation split
  py123d-conversion dataset=nureasoning-stream 'dataset.parser.splits=[nureasoning_val]' \
    dataset.parser.downloader.num_logs=5

  # Named clips, from any of the three splits
  py123d-conversion dataset=nureasoning-stream \
    'dataset.parser.downloader.log_names=[<log>_<token>,<log>_<token>]'

All selected clips are downloaded before the conversion starts, so the temp directory
(``$TMPDIR``) has to hold them. Narrow the selection down with ``num_logs``, ``log_names``
or ``parts``: the full release does not fit. ``dataset=nureasoning-mini-stream`` is the
same for the mini split.

.. note::
  Streaming mode forces ``camera_store_option: "jpeg_binary"`` and
  ``lidar_store_option: "binary"`` (with the ``laz`` codec) — the temp directory is
  deleted immediately after conversion, so any ``"path"`` references would point at
  vanished sources.


Citation
~~~~~~~~

If you use nuReasoning in your research, please cite:

.. code-block:: bibtex

  @misc{huang2026nureasoning,
    title={nuReasoning: A Reasoning-Centric Dataset and Benchmark for Long-Tail Autonomous Driving},
    author={Huang, Zhiyu and Liu, Johnson and Song, Rui and others},
    year={2026},
    eprint={2605.31572},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
  }
