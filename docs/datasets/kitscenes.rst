.. _kitscenes:

KITScenes Multimodal
--------------------

.. warning::

  **Experimental Dataset Support**

  KITScenes Multimodal support is currently experimental and may still change. The dataset itself is an early
  preview release. If you run into issues, please open a bug report on
  `GitHub Issues <https://github.com/kesai-labs/py123d/issues>`_.

KITScenes Multimodal is an urban driving dataset recorded by the Karlsruhe Institute of Technology (KIT) in
Karlsruhe, Frankfurt and Sindelfingen. It contains more than 1,000 scenes of ~20 s, recorded with nine
global-shutter cameras, seven lidars and three 4D imaging radars, together with Lanelet2 HD maps that include
3D-localized traffic lights and signs.

The py123d integration converts the ego trajectory, all cameras, lidars and radars, and the per-scene Lanelet2 maps.
The dataset provides no 3D bounding boxes.

For details about the sensor setup and the official devkit, refer to the
`dataset website <https://kitscenes.com/multimodal/>`_ and the
`KIT-MRT/kitscenes <https://github.com/KIT-MRT/kitscenes>`_ repository.


.. dropdown:: Overview
  :open:

  .. list-table::
    :header-rows: 0
    :widths: 20 60

    * -
      -
    * - :octicon:`file` Paper
      - `The Road Ahead in Autonomous Driving: The KITScenes Multimodal Dataset <https://arxiv.org/abs/2606.02956>`_
    * - :octicon:`download` Download
      - `Hugging Face <https://huggingface.co/datasets/KIT-MRT/KITScenes-Multimodal>`_ (gated)
    * - :octicon:`mark-github` Code
      - `KIT-MRT/kitscenes <https://github.com/KIT-MRT/kitscenes>`_
    * - :octicon:`law` License
      - `CC BY-NC 4.0 <https://creativecommons.org/licenses/by-nc/4.0/>`_ and the KITScenes terms (non-commercial).
    * - :octicon:`database` Available splits
      - ``kitscenes_train``, ``kitscenes_val``, ``kitscenes_test``, ``kitscenes_test_e2e``,
        ``kitscenes_overlap_train_val``


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
     - Georeferenced 6-DoF poses at 10 Hz. Dynamics are inferred from the trajectory. See :class:`~py123d.datatypes.EgoStateSE3`.
   * - Map
     - ✓
     - Per-log Lanelet2 maps: lanes with topology, lane groups, intersections, crosswalks, stop zones, road edges and road lines. See :class:`~py123d.datatypes.Lane`.
   * - Bounding Boxes
     - X
     - Not provided by the dataset.
   * - Traffic Lights
     - X
     - Traffic lights are part of the map, but py123d has no map layer for them yet. Their stop lines are converted to :class:`~py123d.datatypes.StopZone` objects.
   * - Cameras
     - ✓
     - Nine rectified pinhole cameras at 10 Hz: a six-camera surround ring, a tilted stereo pair and a high-resolution front camera. See :class:`~py123d.datatypes.Camera`.
   * - Lidars
     - ✓
     - Seven lidars at 10 Hz: a 128-channel roof lidar, four long-range lidars and two tilted corner lidars. See :class:`~py123d.datatypes.Lidar`.
   * - Radars
     - ✓
     - Three Continental ARS548 4D imaging radars at 10 Hz, with RCS and Doppler velocity. See :class:`~py123d.datatypes.Radar`.


.. dropdown:: Sensor Mapping

  .. list-table::
    :header-rows: 1
    :widths: 40 40

    * - **KITScenes sensor**
      - **py123d ID**
    * - ``camera_ring_front`` / ``_front_left`` / ``_rear_left``
      - ``PCAM_F0`` / ``PCAM_L0`` / ``PCAM_L1``
    * - ``camera_ring_rear`` / ``_front_right`` / ``_rear_right``
      - ``PCAM_B0`` / ``PCAM_R0`` / ``PCAM_R1``
    * - ``camera_base_front_left_rect`` / ``_right_rect``
      - ``PCAM_STEREO_L`` / ``PCAM_STEREO_R``
    * - ``camera_base_front_center`` (high resolution)
      - ``PCAM_F1``
    * - ``lidar_top`` / ``lidar_front`` / ``lidar_rear``
      - ``LIDAR_TOP`` / ``LIDAR_FRONT`` / ``LIDAR_BACK``
    * - ``lidar_left`` / ``lidar_right``
      - ``LIDAR_SIDE_LEFT`` / ``LIDAR_SIDE_RIGHT``
    * - ``lidar_corner_left`` / ``lidar_corner_right``
      - ``LIDAR_FRONT_LEFT`` / ``LIDAR_FRONT_RIGHT``
    * - ``radar_front`` / ``radar_left`` / ``radar_right``
      - ``RADAR_FRONT`` / ``RADAR_BACK_LEFT`` / ``RADAR_BACK_RIGHT``


Download
~~~~~~~~

KITScenes Multimodal is a gated Hugging Face dataset. Accept the terms on the
`dataset page <https://huggingface.co/datasets/KIT-MRT/KITScenes-Multimodal>`_, then log in and download scenes.
Every scene is a single archive of 2-5 GB, which the downloader extracts into the layout the parser expects.

.. code-block:: bash

  pip install py123d[hf]
  hf auth login

  export KITSCENES_DATA_ROOT=/path/to/kitscenes
  # Show the plan first, then download one validation scene:
  py123d-download dataset=kitscenes 'dataset.downloader.splits=[val]' \
      dataset.downloader.max_num_scenes=1 dataset.downloader.dry_run=true
  py123d-download dataset=kitscenes 'dataset.downloader.splits=[val]' dataset.downloader.max_num_scenes=1

Data downloaded with the official ``hf download`` command and extracted with ``tar`` works as well.
The expected layout is:

.. code-block:: text

  $KITSCENES_DATA_ROOT
  └── data/
      ├── sequence_archives.csv
      ├── train/
      │   └── <scene_uuid>/
      └── val/
          └── <scene_uuid>/
              ├── calibration/calib.json
              ├── camera_*/0000000000.jpg ...
              ├── lidar_*/0000000000.parquet ...
              ├── radar_*/0000000000.parquet ...
              ├── maps/map.osm
              ├── maps/origin.json
              ├── poses.txt
              └── timestamp.reference.txt


Installation
~~~~~~~~~~~~

The parser is included in ``py123d`` and does not require the KITScenes devkit or the ``lanelet2`` library.
Install the ``hf`` extra to use the built-in downloader:

.. code-block:: bash

  pip install py123d[hf]


Conversion
~~~~~~~~~~

**Local mode** (already downloaded scenes):

.. code-block:: bash

  export KITSCENES_DATA_ROOT=/path/to/kitscenes
  py123d-conversion dataset=kitscenes

  # Convert selected splits or scenes:
  py123d-conversion dataset=kitscenes 'dataset.parser.splits=[val]'
  py123d-conversion dataset=kitscenes 'dataset.parser.scene_ids=[c34c778f-ad8c-0aa9-7e1a-c86a73f887c7]'

Sensor data is stored as references to the original files, so the converted logs need ``$KITSCENES_DATA_ROOT``.

**Streaming mode** (download + convert in one run):

.. code-block:: bash

  py123d-conversion dataset=kitscenes-stream \
      'dataset.parser.scene_ids=[c34c778f-ad8c-0aa9-7e1a-c86a73f887c7]'

In streaming mode, scenes are downloaded to a managed temporary directory, converted with all sensor data stored in
the logs, and cleaned up afterwards.


Conventions
~~~~~~~~~~~

- **Ego frame:** The KITScenes reference frame (``base_frame``) is the frame of the roof lidar ``lidar_top``
  (x forward, y left, z up). py123d uses it as the ego (IMU) frame.
- **Global frame:** Ego poses and maps share a local metric frame: UTM coordinates minus the UTM coordinates of the
  scene's map origin (``maps/origin.json``), as in Lanelet2's ``UtmProjector``. Heights are absolute.
- **Timestamps:** All timestamps are relative to the start of the scene; the dataset does not publish absolute time.
- **Location:** The city is inferred from the map origin.
- **Lidar:** Points are transformed from the sensor to the ego frame, and invalid returns are removed. Intensity is
  the reflectivity clipped to ``[0, 1]``. Each sweep keeps its per-point timestamps and its own time window.
- **Radar:** Detections are transformed to the ego frame. The raw radial velocity is kept and also projected onto the
  ego-frame ray direction.
- **Map:** Lanelet2 maps are read without the ``lanelet2`` library. Lanelet2 stores no explicit topology, so
  successors and neighbors are derived from shared nodes and linestrings. Two-way lanelets are converted to one lane
  per direction.


Dataset Issues
~~~~~~~~~~~~~~

- **Vehicle dimensions:** The recording vehicle "Joy" is a BMW 7 Series with a roof sensor rack, but its generation
  and exact dimensions are not published. The ego metadata assumes the long-wheelbase BMW 7 Series G12: the wheelbase of
  3.21 m is taken from the `KITScenes LongTail paper <https://arxiv.org/abs/2603.23607>`_, while length (5.238 m),
  width (1.902 m) and height (1.479 m, without the ~0.3 m sensor rack) are stock G12 values. The roof lidar is
  ~2.0 m above the ground (measured from the point cloud) and assumed above the vehicle center. These values are
  unconfirmed by the dataset authors.
- **Ego-motion compensation:** Lidar points and radar velocities are not ego-motion compensated. At urban speeds, a
  lidar sweep spans ~1 m of travel.
- **Ego vehicle points:** The corner lidars see parts of the ego vehicle. These points are not removed.
- **Intersections:** Lanelet2 has no intersection primitive. Intersections are derived from crossing lanes, which
  can produce small false positives where lanes merge or split. They are marked as traffic-light controlled if a lane
  with a traffic light leads into them.
- **Not converted:** Traffic lights, traffic signs and poles of the maps, speed limits from traffic signs (only the
  lanelet ``speed_limit`` tags are used), GNSS/INS measurements and the precomputed ground segmentation.


Citation
~~~~~~~~

If you use KITScenes Multimodal in your research, please cite the
`KITScenes Multimodal paper <https://arxiv.org/abs/2606.02956>`_ as given in the
`official repository <https://github.com/KIT-MRT/kitscenes#citation>`_.
