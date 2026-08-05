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
assets for closed-loop simulation. Each scene is a single ``.usdz`` archive holding one
~20 s clip — rig-to-world ego poses, auto-labeled 3D cuboid tracks, an HD map, and the
3D Gaussian reconstruction used for rendering. The parser converts the driving log and
the map; the reconstruction assets are left untouched.

Scenes carry the HD map in two forms: the MADS ``clipgt/*.parquet`` layers and a
USDZ-internal OpenDRIVE map (``map.xodr``). The parser reads the clipgt layers, which
NVIDIA's own simulator prefers; the OpenDRIVE map is not converted, so a scene without
clipgt layers is rejected rather than converted at lower fidelity. Every scene of the
``26.04`` release carries them (1607 of 1607, verified across the whole release), but
``26.01`` does not — 184 of its 916 scenes ship only ``map.xodr`` (see Dataset Issues).


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
      - ``nurec_train`` (NuRec ships as a single collection; the split is synthetic)


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
     - Rig-to-world poses, resampled to a uniform 10 Hz. NuRec stores poses only; ``infer_ego_dynamics: true`` derives velocity/acceleration during conversion. Vehicle dimensions and the rig-to-box-centre offset come from the rig bounding box, and the wheel base from the rig calibration's axle positions — the release spans several platforms, from 2.73 m to 3.22 m. See :class:`~py123d.datatypes.EgoStateSE3`.
   * - Map
     - ✓
     - Lanes with connectivity, neighbours, lane groups and speed limits, road edges, crosswalks, stop zones (typed by the light or sign controlling their lane, and linked to it), painted road lines, and intersection areas typed by their control. See :class:`~py123d.datatypes.Lane`.
   * - Bounding Boxes
     - ✓
     - Auto-labeled 3D cuboid tracks, interpolated onto the same 10 Hz grid as the ego poses. NuRec shares the Physical AI AV taxonomy (:class:`~py123d.parser.registry.PhysicalAIAVBoxDetectionLabel`). See :class:`~py123d.datatypes.BoxDetectionsSE3`.
   * - Traffic Lights
     - X
     - No per-timestep light states are recorded. Light-controlled stopping points are converted as :class:`~py123d.datatypes.StopZone` instead.
   * - Cameras
     - X
     - Camera frames are rendered from the reconstruction rather than stored as sensor recordings.
   * - Lidars
     - X
     - Not converted.


Download
~~~~~~~~

The dataset is gated on Hugging Face. You need (1) an HF account that has accepted the
NVIDIA AV dataset license and (2) an HF token exported as ``HF_TOKEN``. Scenes are
plain ``.usdz`` files (~1.7 GB each), so any Hugging Face client works, for example:

.. code-block:: bash

  export HF_TOKEN=hf_...

  huggingface-cli download nvidia/PhysicalAI-Autonomous-Vehicles-NuRec \
      --repo-type dataset --revision 26.04 \
      --include "sample_set/26.04_release/*" \
      --local-dir $NUREC_DATA_ROOT/all-usdzs

The parser expects every scene in a single flat directory:

.. code-block:: none

  $NUREC_DATA_ROOT
  └── all-usdzs/
      ├── {scene_uuid}.usdz
      └── ...


Installation
~~~~~~~~~~~~

NuRec conversion requires the ``nurec`` extras group (``csaps`` for the cubic smoothing
spline used by the AlpaSim-parity profile):

.. code-block:: bash

  pip install py123d[nurec]


Conversion
~~~~~~~~~~

.. code-block:: bash

  export NUREC_DATA_ROOT=/path/to/nurec
  export PY123D_DATA_ROOT=/path/to/py123d_data

  py123d-conversion dataset=nurec

``dataset=nurec`` places frames on a uniform 10 Hz grid and interpolates ego poses and
cuboid tracks onto it, since the recorded timestamps are only nominally uniform and
tracks run on their own clock (see Dataset Issues).

The ``nurec-alpasim`` variant additionally applies the transforms NVIDIA's simulator
performs at replay time — smoothing track positions with a cubic smoothing spline and
dropping tracks shorter than 3 s within the scene window:

.. code-block:: bash

  py123d-conversion dataset=nurec-alpasim


Not Converted
~~~~~~~~~~~~~

NuRec labels more of the road than the 123D map schema can currently hold. The
following clipgt layers and fields are read past rather than dropped silently —
they are listed here in case the schema grows a home for them:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - **Source**
     - **Content**
   * - ``road_island``
     - Traffic islands and pedestrian refuges (polygons). Closest existing layers are ``WALKWAY`` and ``GENERIC_DRIVABLE``, neither of which matches.
   * - ``gore_area``
     - Painted no-drive triangles where roads diverge (polygons).
   * - ``road_marking``
     - Arrows, text and symbols painted on the road (polygons). ``ROAD_LINE`` covers painted *lines* only.
   * - ``pole``
     - Sign and signal poles (polylines).
   * - ``traffic_light`` / ``traffic_sign`` geometry
     - 3D boxes with position, dimensions, orientation and sign category (``..._R1_STOP``, ``..._R2_SPEED_LIMIT``, ...). The map schema has no layer for a physical roadside device, so only their effect is converted, as the type of the :class:`~py123d.datatypes.StopZone` and :class:`~py123d.datatypes.Intersection` they control.
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
   * - ``map.xodr``
     - The OpenDRIVE copy of the map, present in every scene alongside the clipgt layers. It describes the same roads in less detail and in a different coordinate frame, so the richer clipgt source is converted instead (see Dataset Issues).
   * - The reconstruction itself
     - ``checkpoint.ckpt`` and ``volume.nurec``, roughly 97% of each archive. These render camera views at arbitrary poses, which is what makes NuRec a closed-loop simulation asset; 123D has no concept for a renderable scene.


Derived Values
~~~~~~~~~~~~~~

Most fields are read straight from clipgt. The following are computed instead, because
the 123D schema asks for something the source does not state directly. Only the first is
a number the dataset does not contain in any form; the rest are derivations from recorded
data. Everything else, including the ego dimensions and wheel base, is read as recorded.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - **Value**
     - **How it is produced**
   * - :class:`~py123d.datatypes.StopZone` outline
     - **Fabricated.** A wait line is a two-point segment and the schema wants a surface, so it is widened to a fixed 1 m depth. Nothing in the dataset states how deep a stopping area is, so any distance measured across a stop zone is this constant, not a measurement.
   * - Which wait lines become stop zones
     - Those whose ``intersection_subtype`` is ``ENTRY`` or ``CROSSWALK_ENTRY``. ``EXIT`` marks where traffic leaves an intersection, and ``NOT_APPLICABLE``/``BUFFER_ZONE`` do not oblige a stop. Any other value is dropped with a warning.
   * - :class:`~py123d.datatypes.StopZoneType`
     - From the traffic light or sign controlling the lane, then the crossing the line guards (``CROSSWALK_ENTRY`` becomes ``PEDESTRIAN_CROSSING``), and only then the wait line's own category. That category marks a painted stop bar and is set for signal-controlled lines too, so it types just the lines nothing else accounts for.
   * - :class:`~py123d.datatypes.IntersectionType`
     - From the lights and signs on the intersection's lanes; the clipgt category describes shape (``FOUR_WAY``, ...) rather than control.
   * - Lane centerline
     - Midpoint of the two rails, paired by normalized arc-length. The rails rarely share a point count, so pairing by index would skew the centre.
   * - Lane ordering within a group
     - Geometric, by offset along the normal of the shared heading. The left/right relations are incomplete for roads whose neighbouring lanes leave the clip.
   * - Lane speed limit
     - clipgt stores mph; converted to m/s. A limit of 0 becomes ``None`` rather than a standstill.
   * - Frame timestamps
     - An exact 10 Hz grid anchored at the second rig timestamp, with ego poses and cuboid tracks interpolated onto it (see Dataset Issues).
   * - Ego velocity and acceleration
     - Not recorded; derived during conversion by ``infer_ego_dynamics``.


Dataset Issues
~~~~~~~~~~~~~~

- **No traffic-light states.** The map layers contain traffic-light geometry, but the
  dataset records no per-timestep light states, so no traffic-light modality is emitted.
  A converted map says where traffic must stop for a signal, never when.
- **The bundled ``map.xodr`` is not converted, so 184 scenes of the ``26.01`` release
  cannot be converted at all.** clipgt is the richer source and the whole ``26.04``
  release carries it, so the OpenDRIVE copy is unused there; in ``26.01`` those 184
  scenes ship no clipgt layers and the parser rejects them with a clear error. Reading
  the OpenDRIVE copy instead would need work first:
  :mod:`py123d.parser.opendrive` raises on it, because NuRec omits attributes the
  parser reads unconditionally that OpenDRIVE 1.4 makes optional: in a 12-scene sample,
  ``header``'s ``north``/``south``/``east``/``west`` are absent in every scene (where
  parsing stops first), ``controller``'s ``sequence`` in all 24 controllers, and
  ``object``'s ``roll`` and ``pitch`` in all 538 objects, while 76 of 416 junction
  connections reference roads outside the clip. Its ``geoReference`` is malformed too
  (``+=alt_0=0`` for ``+alt_0=0``, which PROJ rejects) and names an EGM96 geoid grid that
  ships with neither pyproj nor PROJ, so heights would need care as well. Supporting it
  means fixing the OpenDRIVE parser first, which is out of scope for this dataset.
- **Speed limits are sparse.** Lane speed limits are present in recent releases and
  absent in older ones; lanes without a speed limit convert with ``speed_limit_mps=None``.
- **Non-uniform source timestamps.** Rig timestamps are nominally 10 Hz but drift by
  milliseconds, and cuboid-track timestamps run on a separate clock. Conversion places
  frames on an exact 10 Hz grid and interpolates onto it (position lerp, quaternion
  slerp), so converted timestamps differ slightly from the recorded ones.


Citation
~~~~~~~~

- `NuRec on Hugging Face <https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec>`_
