.. _nurec:

NuRec (PhysicalAI-AV NuRec)
---------------------------

.. warning::

  **Experimental Dataset Support**

  The NuRec dataset integration is currently **under active development** and should be considered experimental.
  Features may be incomplete, APIs may change, and unexpected bugs are possible.

  If you encounter any issues, please report them on our
  `GitHub Issues <https://github.com/kesai-labs/py123d/issues>`_ page. Your feedback helps us improve!

NuRec is NVIDIA's ``PhysicalAI-Autonomous-Vehicles-NuRec`` dataset: neural-reconstruction
assets for closed-loop simulation. Each scene is one ``.usdz`` archive covering about
20 s: rig-to-world ego poses, auto-labeled 3D cuboid tracks, an HD map, and the 3D
Gaussian reconstruction used for rendering. The parser converts the driving log and the
map. The reconstruction assets are left untouched.

Scenes carry the HD map in two forms: the MADS ``clipgt/*.parquet`` layers and a
USDZ-internal OpenDRIVE map (``map.xodr``). The parser reads the clipgt layers, which
NVIDIA's own simulator also prefers, and falls back to the OpenDRIVE map for a scene
without them. All 1607 scenes of the ``26.04`` release carry clipgt. In ``26.01``,
184 of 916 scenes ship only ``map.xodr`` (see Dataset Issues).


.. dropdown:: Overview
  :open:

  .. list-table::
    :header-rows: 0
    :widths: 20 60

    * -
      -
    * - :octicon:`download` Download
      - `Hugging Face <https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec>`_ (gated)
    * - :octicon:`law` License
      - Please refer to the dataset's official license terms.
    * - :octicon:`database` Available splits
      - ``nurec-curated_train``, ``nurec-curated_val``, ``nurec-2601_train``, ``nurec-2604_train`` (see Conversion)


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
     - Rig-to-world poses, resampled to a uniform 10 Hz. NuRec stores poses only; ``infer_ego_dynamics: true`` derives velocity/acceleration during conversion. Vehicle dimensions and the rig-to-box-centre offset come from the rig bounding box, and the wheel base from the rig calibration's axle positions. The release spans several vehicle platforms, with wheel bases from 2.73 m to 3.22 m. See :class:`~py123d.datatypes.EgoStateSE3`.
   * - Map
     - ✓
     - Lanes with connectivity, neighbours, lane groups and speed limits, road edges, crosswalks, stop zones (typed by the light or sign controlling their lane, and linked to it), painted road lines, intersection areas typed by their control, generic drivable areas (gore areas), and walkways (traffic islands). See :class:`~py123d.datatypes.Lane`.
   * - Bounding Boxes
     - ✓
     - Auto-labeled 3D cuboid tracks, interpolated onto the same 10 Hz grid as the ego poses. NuRec shares the Physical AI AV taxonomy (:class:`~py123d.parser.registry.PhysicalAIAVBoxDetectionLabel`). See :class:`~py123d.datatypes.BoxDetectionsSE3`.
   * - Traffic Lights
     - X
     - No per-timestep light states are recorded. Light-controlled stopping points are converted as :class:`~py123d.datatypes.StopZone` instead.
   * - Cameras
     - X
     - The dataset ships no recorded camera frames. Views are rendered from the reconstruction.
   * - Lidars
     - X
     - Not converted.


Download
~~~~~~~~

The dataset is gated on Hugging Face. You need (1) an HF account that has accepted the
NVIDIA AV dataset license and (2) an HF token exported as ``HF_TOKEN``. Scenes are
plain ``.usdz`` files (about 2 GB each), fetched with the ``py123d-download`` CLI:

.. code-block:: bash

  export HF_TOKEN=hf_...
  export NUREC_DATA_ROOT=/path/to/nurec

  # The scenes of nurec-curated_train and nurec-curated_val, drawn from both releases
  py123d-download dataset=nurec-curated

  # Every scene of one release
  py123d-download dataset=nurec-2601
  py123d-download dataset=nurec-2604

  # A few scenes per split, or the plan without downloading
  py123d-download dataset=nurec-curated dataset.downloader.num_sequences=5
  py123d-download dataset=nurec-curated dataset.downloader.dry_run=true

  # Scenes of release 26.04 by name
  py123d-download dataset=nurec-2604 \
      'dataset.downloader.sequence_ids=[{scene_uuid},{scene_uuid}]'

Downloads keep the layout of the Hugging Face repository, which is what the parser
expects. Any other Hugging Face client that preserves it works as well:

.. code-block:: none

  $NUREC_DATA_ROOT
  └── sample_set/
      ├── 26.01_release/
      │   └── {scene_uuid}/
      │       └── {scene_uuid}.usdz
      └── 26.04_release/
          └── {scene_uuid}/
              └── {scene_uuid}.usdz

The release directory is part of a scene's identity. Some scene UUIDs exist in both
releases, as separate reconstructions of the same drive.


Installation
~~~~~~~~~~~~

NuRec conversion requires the ``nurec`` extras group (``csaps`` for the cubic smoothing
spline used by the AlpaSim-parity profile, and ``huggingface_hub`` for downloads):

.. code-block:: bash

  pip install py123d[nurec]


Conversion
~~~~~~~~~~

.. code-block:: bash

  export NUREC_DATA_ROOT=/path/to/nurec
  export PY123D_DATA_ROOT=/path/to/py123d_data

  py123d-conversion dataset=nurec-curated

NuRec ships as releases without official splits. ``dataset=nurec-curated`` converts the
train and validation scenes curated for the `AlpaSim <https://github.com/NVlabs/alpasim>`_
E2E challenge, which draw from both releases. The other two configs convert one release each:

.. list-table::
   :header-rows: 1
   :widths: 25 50 25

   * - **Split**
     - **Scenes**
     - **Config**
   * - ``nurec-curated_train``
     - 1761 named scenes (702 of ``26.01``, 1059 of ``26.04``)
     - ``dataset=nurec-curated``
   * - ``nurec-curated_val``
     - 441 named scenes (177 of ``26.01``, 264 of ``26.04``)
     - ``dataset=nurec-curated``
   * - ``nurec-2601_train``
     - Every scene of release ``26.01``
     - ``dataset=nurec-2601``
   * - ``nurec-2604_train``
     - Every scene of release ``26.04``
     - ``dataset=nurec-2604``

The scene names of ``nurec-curated_train`` and ``nurec-curated_val`` are listed in
``parser/nurec/nurec_curated_splits.yaml``. A listed scene that is not on
disk is skipped with a warning, so a partial download converts partially.
``dataset.parser.splits`` selects any combination of the four splits.

Each config has a ``-stream`` variant, which downloads the scenes into a temporary
directory instead of reading ``NUREC_DATA_ROOT`` and deletes them when the conversion
ends. All selected scenes are downloaded before the conversion starts, so limit the
selection unless the temporary directory can hold it:

.. code-block:: bash

  py123d-conversion dataset=nurec-curated-stream dataset.parser.num_sequences=3

Each config has a scene filter of the same name for reading the converted logs, for
example ``py123d-viser scene_filter=nurec-curated`` or ``scene_filter=nurec-2601``.

Frames are placed on a uniform 10 Hz grid, with ego poses and cuboid tracks interpolated
onto it, since the recorded timestamps are only nominally uniform and tracks run on
their own clock (see Dataset Issues).

Two options also apply the transforms NVIDIA's simulator performs at replay time.
They smooth track positions with a cubic smoothing spline and drop tracks shorter
than 3 s within the scene window:

.. code-block:: bash

  py123d-conversion dataset=nurec-curated \
      dataset.parser.smooth_track_positions=true \
      dataset.parser.min_traffic_duration_us=3000000

The map source is selected with ``dataset.parser.map_source``:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - **Value**
     - **Map is read from**
   * - ``clip_gt_or_xodr`` (default)
     - The clipgt layers, or ``map.xodr`` for a scene that has none.
   * - ``xodr_or_clip_gt``
     - ``map.xodr``, or the clipgt layers for a scene that has none.
   * - ``clip_gt``
     - The clipgt layers only.
   * - ``xodr``
     - ``map.xodr`` only.

A scene that carries none of the requested sources is converted without a map. Its log
then has no map, and a scene filter with ``has_map`` treats it accordingly. A map written
by an earlier conversion is not removed, so convert into a fresh ``PY123D_DATA_ROOT`` when
changing ``map_source`` for scenes that were already converted.


Not Converted
~~~~~~~~~~~~~

NuRec labels more of the road than the 123D map schema can represent. Some of the
layers below are never opened. The rest are fields on rows the parser reads and
ignores. They are listed here in case the schema later covers them:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - **Source**
     - **Content**
   * - ``road_marking``
     - Arrows, text and symbols painted on the road (polygons). ``ROAD_LINE`` covers painted *lines* only.
   * - ``pole``
     - Sign and signal poles (polylines).
   * - ``traffic_light`` / ``traffic_sign`` geometry
     - 3D boxes with position, dimensions, orientation and sign category (``..._R1_STOP``, ``..._R2_SPEED_LIMIT``, ...). The map schema has no layer for a roadside device, so only their effect is converted, as the type of the :class:`~py123d.datatypes.StopZone` and :class:`~py123d.datatypes.Intersection` they control.
   * - ``lane.lane_direction``
     - Whether a lane goes straight, turns, or both. :class:`~py123d.datatypes.Lane` has no turn-direction field.
   * - ``lane.left_edge_styles`` / ``colors``
     - Paint style and colour of each lane's own boundary, per point.
   * - ``lane.map_end``
     - Marks lanes truncated by the clip boundary rather than by the road.
   * - ``intersection_area.category``
     - Intersection shape (``FOUR_WAY``, ``T_JUNCTION``, ...). :class:`~py123d.datatypes.IntersectionType` describes control rather than shape.
   * - ``road_boundary.category``
     - What an edge physically is: ``tall_curb``, ``barrier``, ``fence``, ``wall`` or a plain ``road_boundary``.
   * - ``road_boundary`` driving directions
     - Which side of an edge is drivable and in which direction, per point. Boundaries are oriented with the drivable side on the left, so every edge converts as ``ROAD_EDGE_BOUNDARY``; NuRec does not mark medians, which appear as two opposing boundaries.
   * - Further ``association`` kinds
     - Opposite-direction and overlapping lane neighbours, branch/merge siblings, lane-to-boundary-line links, and crosswalk/marking-to-lane links.
   * - Sensor calibrations and frame poses
     - Intrinsics and rig extrinsics for 6 cameras and 1 lidar, with per-frame poses and timestamps (~600 camera frames, ~200 lidar frames per scene). A scene ships no recorded frames to point at, so no camera or lidar modality is registered.
   * - ``map.xodr`` (when clipgt is present)
     - The OpenDRIVE copy of the map describes the same roads in less detail and in a different coordinate frame, so the richer clipgt source is converted instead. It is read only as a fallback, or when ``map_source`` asks for it (see Conversion).
   * - The scene reconstruction
     - ``checkpoint.ckpt`` and ``volume.nurec``, which render camera views at arbitrary poses. 123D has no concept for a renderable scene. Camera data could still be obtained by replaying the groundtruth trajectories in AlpaSim.


Derived Values
~~~~~~~~~~~~~~

Most fields are read straight from clipgt. The values below are computed, because the
123D schema asks for something the source does not state directly. The first is invented;
the rest are derived from recorded data. Everything not listed, including the ego
dimensions and wheel base, is read as recorded.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - **Value**
     - **How it is produced**
   * - :class:`~py123d.datatypes.StopZone` outline
     - A wait line is a two-point segment and :class:`~py123d.datatypes.StopZone` needs a surface, so it is widened to a fixed 1 m depth. NuRec does not record how deep a stopping area is, so this number is invented.
   * - Which wait lines become stop zones
     - Those whose ``intersection_subtype`` is ``ENTRY`` or ``CROSSWALK_ENTRY``. ``EXIT`` marks where traffic leaves an intersection, and ``NOT_APPLICABLE``/``BUFFER_ZONE`` do not oblige a stop. Any other value is dropped with a warning.
   * - :class:`~py123d.datatypes.StopZoneType`
     - Taken from the traffic light or sign controlling the lane, then from the crossing the line guards (``CROSSWALK_ENTRY`` becomes ``PEDESTRIAN_CROSSING``), and last from the wait line's own category. That category marks a painted stop bar and is set on signal-controlled lines too, so it only types the lines nothing else covers.
   * - :class:`~py123d.datatypes.IntersectionType`
     - From the lights and signs on the intersection's lanes; the clipgt category describes shape (``FOUR_WAY``, ...) rather than control.
   * - Lane centerline
     - Midpoint of the two rails, paired by normalized arc-length. The rails rarely share a point count, so pairing by index would skew the centre.
   * - Lane ordering within a group
     - Geometric, by offset along the normal of the shared heading. The left/right relations are incomplete for roads whose neighbouring lanes leave the clip.
   * - Lane speed limit
     - clipgt stores mph, converted to m/s. A limit of 0 means unset and becomes ``None``.
   * - Frame timestamps
     - An exact 10 Hz grid anchored at the second rig timestamp, with ego poses and cuboid tracks interpolated onto it (see Dataset Issues).
   * - Ego velocity and acceleration
     - Not recorded; derived during conversion by ``infer_ego_dynamics``.


Dataset Issues
~~~~~~~~~~~~~~

- **No traffic-light states.** The map layers contain traffic-light geometry, but the
  dataset records no per-timestep light states, so no traffic-light modality is emitted.
  A converted map records where traffic must stop for a signal, but not the signal state.
- **Some scenes carry no clipgt layers.** The whole ``26.04`` release carries clipgt. In
  ``26.01``, 184 of 916 scenes ship only ``map.xodr``, which is then converted instead through
  :mod:`py123d.parser.opendrive` and is less detailed than a clipgt map. NuRec's
  OpenDRIVE files omit several attributes that OpenDRIVE 1.4 makes optional. In a
  12-scene sample, ``header``'s ``north``/``south``/``east``/``west`` are absent in every
  scene, ``controller``'s ``sequence`` in all 24 controllers, and ``object``'s ``roll``
  and ``pitch`` in all 538 objects. A further 76 of 416 junction connections reference
  roads outside the clip and are skipped. The ``geoReference`` is malformed as well
  (``+=alt_0=0`` instead of ``+alt_0=0``, which PROJ rejects) and names an EGM96 geoid
  grid that ships with neither pyproj nor PROJ, so only its ``+lat_0``/``+lon_0`` origin
  is read. The map is then moved into the clip-local frame of the ego poses using the
  world-from-base pose in ``rig_trajectories.json``.
- **Speed limits are sparse.** Lane speed limits are present in recent releases and
  absent in older ones; lanes without a speed limit convert with ``speed_limit_mps=None``.
- **Non-uniform source timestamps.** Rig timestamps are nominally 10 Hz but drift by
  milliseconds, and cuboid-track timestamps run on a separate clock. Conversion places
  frames on an exact 10 Hz grid and interpolates onto it (position lerp, quaternion
  slerp), so converted timestamps differ slightly from the recorded ones.


Citation
~~~~~~~~

- `NuRec on Hugging Face <https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec>`_
