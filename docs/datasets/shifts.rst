Shifts
------

The Shifts vehicle motion prediction dataset is the self-driving-car task of the Shifts benchmark,
recorded by the Yandex Self-Driving Group fleet in six cities (Moscow, Skolkovo, Innopolis, Ann Arbor,
Modiin, Tel Aviv). It is a motion-prediction dataset: there are no sensors, each scene holds the
tracked agents, the ego vehicle, a per-scene vector map, traffic light states and scene tags.

.. dropdown:: Overview
  :open:

  .. list-table::
    :header-rows: 0
    :widths: 20 60

    * -
      -
    * - :octicon:`file` Paper
      - `Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks <https://arxiv.org/abs/2107.07455>`_
    * - :octicon:`download` Download
      - `Yandex Object Storage <https://github.com/Shifts-Project/shifts#motion-prediction>`_ (public, no account)
    * - :octicon:`mark-github` Code
      - `Shifts-Project/shifts <https://github.com/Shifts-Project/shifts/tree/main/sdc>`_
    * - :octicon:`law` License
      - `CC BY-NC-SA 4.0 <https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode>`_
    * - :octicon:`database` Available splits
      - ``shifts_train``, ``shifts_val``, ``shifts_test``


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
     - Pose and velocity of the ego vehicle at 5 Hz, see :class:`~py123d.datatypes.EgoStateSE3`. Dynamics are inferred by the writer (see below).
   * - Map
     - (✓)
     - One vector map per scene (log): lane centerlines with speed limits, crosswalks and road polygons. Lane boundaries are synthesized (see below). For more information, see :class:`~py123d.api.MapAPI`.
   * - Bounding Boxes
     - ✓
     - Vehicles and pedestrians with the :class:`~py123d.parser.registry.ShiftsBoxDetectionLabel`, with global-frame velocities. For more information, see :class:`~py123d.datatypes.BoxDetectionsSE3`.
   * - Traffic Lights
     - (✓)
     - Per-lane status for the 25 past frames of each scene, see :class:`~py123d.datatypes.TrafficLightDetections`. The future frames carry no traffic light states.
   * - Custom
     - ✓
     - Scene tags and prediction requests are available through the ``aux`` custom modality.
   * - Cameras
     - X
     - n/a
   * - Lidars
     - X
     - n/a

.. dropdown:: Dataset Specific


  .. autoclass:: py123d.parser.registry.ShiftsBoxDetectionLabel
    :members:
    :no-inherited-members:


Splits and timing
~~~~~~~~~~~~~~~~~

py123d converts the *canonical partition* of Shifts, the one used in the Shifts Challenge 2021:

.. list-table::
   :header-rows: 1
   :widths: 20 20 15 45

   * - **py123d split**
     - **Shifts archive**
     - **Scenes**
     - **Content**
   * - ``shifts_train``
     - ``train``
     - 388,406
     - Moscow, no precipitation
   * - ``shifts_val``
     - ``development``
     - 36,605
     - 27,036 in-distribution (Moscow, dry) and 9,569 shifted scenes (Skolkovo, Innopolis, Modiin, any weather)
   * - ``shifts_test``
     - ``evaluation``
     - 36,804
     - 26,865 in-distribution and 9,939 shifted scenes (Ann Arbor, Tel Aviv, any weather)

Shifts calls its validation set *development* and its test set *evaluation*; both carry the full
future for every agent, nothing is masked. In-distribution and shifted scenes share an archive and are
told apart by the scene tags in the ``aux`` metadata, or by :attr:`~py123d.datatypes.LogMetadata.location`,
which holds the city.

Every scene becomes one log of 50 frames at 5 Hz: 25 past frames followed by 25 future frames. The raw
data has no absolute timestamps, so the log timestamps start at 0 s and the prediction time *T0* is frame
24 (4.8 s); the ``aux`` metadata records this as ``t0_frame_index``. The log name is the Shifts scene id.


Custom Metadata
~~~~~~~~~~~~~~~

The ``aux`` custom modality carries the scene-level information that has no slot in the 123D schema:

.. code-block:: python

  aux_metadata = scene.get_all_custom_modality_metadatas()["aux"].metadata

* ``scene_id``
* ``scene_tags``: ``day_time``, ``season``, ``track`` (city), ``sun_phase``, ``precipitation``
* ``prediction_requests``: track token of each vehicle to predict, with its trajectory tags (``kMoveLeft``, ``kStopping``, ...)
* ``num_history_frames``, ``num_future_frames``, ``t0_frame_index``
* ``lanes_give_way``: ids of the lanes flagged as yielding to another lane

Per frame, the modality holds ``frame_index`` and ``phase`` (``history``, ``current`` or ``future``).


Download
~~~~~~~~

The canonical partition is hosted on Yandex Object Storage as two plain tar files
(``canonical-trn-dev-data.tar``, 129 GB, and ``canonical-eval-data.tar``, 11 GB), each wrapping one
gzipped tar per split plus pre-rendered feature maps that py123d does not need. ``py123d-download``
fetches only the scene archives, by HTTP range requests into the outer tar files:

.. code-block:: bash

  # Validation and test scenes (5.6 GB + 6.0 GB compressed):
  py123d-download dataset=shifts 'dataset.downloader.splits=[shifts_val,shifts_test]'

  # Everything, including the 61 GB training archive:
  py123d-download dataset=shifts

  # Only the first 100 scenes of each split:
  py123d-download dataset=shifts dataset.downloader.num_scenes=100

The result, or a manual extraction of the three ``.tar.gz`` members, has the layout

.. code-block:: text

  $SHIFTS_DATA_ROOT
    ├── train/
    |   ├── 000/
    |   |   ├── 000000.pb
    |   |   └── ...
    |   └── ...
    ├── development/
    |   └── ...
    └── evaluation/
        └── ...

You can add the dataset root directory to the environment variable ``SHIFTS_DATA_ROOT`` for easier access.

.. code-block:: bash

   export SHIFTS_DATA_ROOT=/path/to/shifts_data_root

Optionally, you can adjust the ``py123d/script/config/common/default_dataset_paths.yaml`` accordingly.

Installation
~~~~~~~~~~~~

Reading the raw scenes needs ``protobuf``, which is included as an optional dependency:

.. tab-set::

  .. tab-item:: PyPI

    .. code-block:: bash

      pip install py123d[shifts]

  .. tab-item:: Source

    .. code-block:: bash

      pip install -e .[shifts]

After conversion, any other ``py123d`` installation can read the logs.


Conversion
~~~~~~~~~~

**Local mode** — data already downloaded to ``$SHIFTS_DATA_ROOT``:

.. code-block:: bash

  py123d-conversion dataset=shifts

**Streaming mode** — ``dataset=shifts-stream`` attaches a ``ShiftsDownloader`` to the parser; it
streams the selected scene archives into ``$PY123D_DATA_ROOT/.download/shifts`` at parser
construction time and deletes them when the conversion ends. Because the archives are sequential
gzip streams of one file per scene, ``num_scenes`` stops the stream after the first scenes of a split:

.. code-block:: bash

  # The first 5 validation scenes (about 1 MB of download):
  py123d-conversion dataset=shifts-stream \
      'dataset.parser.splits=[shifts_val]' \
      dataset.parser.downloader.num_scenes=5

  # Whole validation and test splits:
  py123d-conversion dataset=shifts-stream 'dataset.parser.splits=[shifts_val,shifts_test]'

  # Keep the downloaded scenes instead of deleting them (a later run reuses them):
  py123d-conversion dataset=shifts-stream 'dataset.parser.splits=[shifts_val]' keep_downloads=true


Dataset Specific Issues
~~~~~~~~~~~~~~~~~~~~~~~

* **Ego vehicle parameters:** The scenes report a constant 4.62 x 1.77 x 1.50 m ego box and nothing
  else about the vehicle. The wheelbase (2.7 m) and the axle position (rear axle 1.35 m behind the box
  centre, IMU frame at the box centre) in :class:`~py123d.datatypes.EgoStateSE3Metadata` are placeholders
  for a mid-size sedan. The ego position is reported at ground level and is lifted by half the height to
  the box centre, like the agent boxes.
* **Dynamics:** Shifts provides ego and agent velocities but no ego acceleration, so the ego dynamics are
  inferred from the poses (``infer_ego_dynamics: true``). Agent velocities are stored as given; the
  per-agent acceleration of the raw data is dropped, since :class:`~py123d.datatypes.BoxDetectionSE3` has
  no slot for it.
* **Road surface:** The road polygons of a scene are overlapping tiles. py123d merges them into one
  drivable surface per connected road area (holes are traffic islands) and stores the outlines of the
  merged surfaces as road edges.
* **Lane boundaries:** Shifts lanes are centerlines without boundaries, neighbours or connectivity. The
  boundaries are synthesized with the same ray casting as for WOD, using the merged road edges: a lane
  ends at the road surface or halfway to a neighbouring centerline, with a default width of 3.7 m. Lanes
  therefore have no predecessors, successors or lane groups.
* **Traffic lights:** States are given per traffic light *section*, which the lanes reference. py123d
  reports one status per regulated lane for the 25 past frames, following the lane availability logic of
  the official API: arrow sections govern their lane unless the main light is off. Traffic lights have no
  position in Shifts.
* **Pedestrian heading:** Pedestrian tracks carry no orientation. Like the official API, their heading
  is taken from the velocity vector (0 when standing still).
* **Coordinates:** Positions are in a metric frame per city, not geo-referenced; the ``location`` of a log
  is the city name from the scene tags.
* **Full-unpartitioned release:** Shifts also publishes all 600,000 scenes as ``full-unpartitioned-data.tar``.
  In that release the future part of a scene holds only the prediction-request vehicles, without the ego
  and the pedestrians, so py123d supports the canonical partition only.

Citation
~~~~~~~~

If you use this dataset in your research, please cite:

.. code-block:: bibtex

  @inproceedings{Malinin2021Shifts,
    title={Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks},
    author={Malinin, Andrey and Band, Neil and Chesnokov, German and Gal, Yarin and Gales, Mark J. F. and Noskov, Alexey and Ploskonosov, Andrey and Prokhorenkova, Liudmila and Provilkov, Ivan and Raina, Vatsal and Raina, Vyas and Shmatova, Mariya and Tigas, Panos and Yangel, Boris},
    booktitle={Advances in Neural Information Processing Systems, Datasets and Benchmarks Track},
    year={2021}
  }
