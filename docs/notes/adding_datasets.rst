Adding Datasets
===============

This tutorial walks through adding a new source dataset to the 123D conversion
pipeline. A dataset is integrated by implementing a set of **parser**
classes that read the raw data and yield 123D's unified datatypes. The
framework then handles configurable synchronization, log writing, and exposing the
result through the unified API.


1. General
----------

Architecture overview
~~~~~~~~~~~~~~~~~~~~~~~

Conversion is organized around three base classes defined in
:class:`~py123d.parser.base_dataset_parser.BaseDatasetParser`. A top-level
*dataset parser* produces lightweight, per-log and per-map handles that .
The orchestrator constructs these once on the main process, then distributes them
to parallel workers, which perform the heavy I/O internally and hand the parsed data to the
log/map writers:

.. code-block:: text

  BaseDatasetParser
    ├── get_log_parsers()  ──▶  [ LogParser_1, ..., LogParser_N ]  ──▶  ArrowLogWriter
    └── get_map_parsers()  ──▶  [ MapParser_1, ..., MapParser_M ]  ──▶  ArrowMapWriter
                                                                            │
                                                                            ▼
                                                                  Arrow storage ──▶ unified API

Each new dataset adds a package under ``src/py123d/parser/<dataset>/``. Existing
packages (i.e., ``nuscenes/``, ``nuplan/``, ``wod/``, ``av2/``, ``kitti360/``,
``pandaset/``) are implemented as references.

The dataset parser
~~~~~~~~~~~~~~~~~~~~

Subclass :class:`~py123d.parser.base_dataset_parser.BaseDatasetParser` and
implement two methods:

.. code-block:: python

  from py123d.parser.base_dataset_parser import BaseDatasetParser

  class MyDatasetParser(BaseDatasetParser):

      def get_log_parsers(self) -> list[BaseLogParser]:
          """One lightweight, picklable handle per log/scene."""
          ...

      def get_map_parsers(self) -> list[BaseMapParser]:
          """One handle per map region (return [] if the dataset has no maps)."""
          ...

.. important::
  The returned handles are shipped to worker processes. Keep them
  **lightweight**, i.e., store only paths and parameters, not open file handles,
  database connections, or decoded sensor data. All expensive I/O belongs inside
  the iterators (covered below), which run on the worker.


Optional and gated dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dataset-specific third-party packages (devkits, format readers, cloud SDKs) should
**not** be hard dependencies of 123D. Declare them as an extra in
``pyproject.toml`` under ``[project.optional-dependencies]``:

.. code-block:: toml

  [project.optional-dependencies]
  mydataset = ["mydataset-devkit==1.2.3"]

Then gate the import at the top of your parser (or lazily inside the method that
needs it) with :func:`~py123d.common.utils.dependencies.check_dependencies`, which
raises a helpful, install-instruction error if the package is missing:

.. code-block:: python

  from py123d.common.utils.dependencies import check_dependencies

  check_dependencies(["mydataset"], optional_name="mydataset")
  # ImportError: Missing 'mydataset'. Install with: pip install py123d[mydataset]

Entry point
~~~~~~~~~~~

Conversion is driven by the ``py123d-conversion`` CLI (defined in
``[project.scripts]``, mapping to
:func:`py123d.script.run_conversion.main`). It uses `Hydra
<https://hydra.cc>`_ to instantiate your parser from a YAML config via its
``_target_`` field (see :ref:`section 5 <adding-datasets-hydra>`):

.. code-block:: bash

  py123d-conversion dataset=mydataset


2. Log Parser
-------------

A *log parser* is a handle to a single continuous driving sequence (i.e., called 'log' in 123D). Subclass
:class:`~py123d.parser.base_dataset_parser.BaseLogParser` and implement the log
metadata getter and the synchronized-frame iterator.

Log metadata
~~~~~~~~~~~~~

``get_log_metadata()`` returns a
:class:`~py123d.datatypes.metadata.log_metadata.LogMetadata` describing the log:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Description
   * - ``dataset``
     - Dataset name, lowercase and dashed (e.g. ``"my-dataset"``).
   * - ``split``
     - Data split, prefixed by dataset (e.g. ``"my-dataset_train"``).
   * - ``log_name``
     - Unique name of the log.
   * - ``location``
     - Geographic location (optional); links a log to a per-location map.
   * - ``map_metadata``
     - Associated :class:`~py123d.datatypes.MapMetadata`, if any.
   * - ``version``
     - 123D version that produced the metadata (defaults to the current version).

Iterating modalities
~~~~~~~~~~~~~~~~~~~~~~

``iter_modalities_sync()`` lazily yields one
:class:`~py123d.parser.base_dataset_parser.ModalitiesSync` per synchronized frame.
Each frame bundles a timestamp with a list of 123D *modalities*. The unified
:class:`~py123d.datatypes.modalities.base_modality.BaseModality` types parsed from
the dataset's native formats (ego state, bounding boxes, cameras, lidar, radar,
…):

.. code-block:: python

  from py123d.parser.base_dataset_parser import ModalitiesSync

  def iter_modalities_sync(self):
      for frame in self._read_frames():
          modalities = [
              self._parse_ego_state(frame),
              self._parse_boxes(frame),
              *self._parse_cameras(frame),
              self._parse_lidar(frame),
          ]
          yield ModalitiesSync(timestamp=frame.timestamp, modalities=modalities)

Parsed sensor datatypes and lazy I/O
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sensor data is the bulk of any dataset, so 123D may optionally skip decoding it during parsing.
Instead, you can pass a *parsed* modality helpers, which carry **only a
file path (or raw bytes) plus metadata**. Depending on the storage option for sensor data,
the writer can just store the sensor reference, or decode and re-compress the data internally for self-contained storage.

* :class:`~py123d.parser.base_dataset_parser.ParsedCamera`: camera metadata, a
  ``camera_to_global_se3`` pose, and either a ``dataset_root``/``relative_path``
  pair **or** an in-memory ``byte_string``.
* :class:`~py123d.parser.base_dataset_parser.ParsedLidar`: lidar metadata,
  ``start_timestamp``/``end_timestamp`` (the sweep window), a
  ``dataset_root``/``relative_path``, and optional ``load_kwargs``.
* :class:`~py123d.parser.base_dataset_parser.ParsedRadar`: radar metadata, a
  single ``timestamp`` (a radar scan is an instantaneous snapshot), a file path,
  and optional ``load_kwargs``.

If sensor storage is configured as "path", the whole conversion process is substantially faster,
as the conversion is not memory- or I/O-bound. On the other hand, the converted logs are not self-contained and require
access to the original dataset files for loading.

.. tip::
  ``load_kwargs`` is a generic, dataset-specific dictionary forwarded verbatim to
  the point-cloud loader and persisted in the Arrow frame under the ``"path"``
  store option. nuScenes uses it to carry the keyframe's panoptic-label path
  alongside the lidar file.

Arrival times (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Every sensor row stores its measurement time in the column ``<modality_key>.timestamp_us``. A
recorded dataset can also store the time at which the recording system received each measurement,
for example the publish time of the message that carried it. A replay of the log in the order and at
the times in which the data arrived needs this second time.

To store it, set ``has_arrival_time=True`` in the sensor's metadata and pass ``arrival_timestamp`` to
each row. This is supported by :class:`~py123d.datatypes.Imu`, :class:`~py123d.datatypes.Gnss`,
:class:`~py123d.datatypes.Barometer`, :class:`~py123d.datatypes.Magnetometer`,
:class:`~py123d.datatypes.Camera`, :class:`~py123d.datatypes.Lidar` and
:class:`~py123d.datatypes.Radar`, by the parsed helpers
:class:`~py123d.parser.base_dataset_parser.ParsedCamera`,
:class:`~py123d.parser.base_dataset_parser.ParsedLidar` and
:class:`~py123d.parser.base_dataset_parser.ParsedRadar`, and by the metadata classes of these
sensors (pinhole, fisheye MEI and f-theta cameras).

* The writer adds the column ``<modality_key>.arrival_us`` (int64, microseconds) as the last
  column of the file. A row without an arrival time stores null.
* On read, :attr:`~py123d.datatypes.BaseModality.arrival_timestamp` returns the stored time, or
  ``None`` if the row or the file has none.
* With the flag unset (the default) the file has no such column, and its schema and metadata are
  those of a file written before the column existed. A row that carries an arrival time while the
  flag is unset is rejected by the writer, so the value cannot be lost without notice.
* Synchronization does not use the arrival time. The sync table, timestamp lookups, scene
  construction and scene UUIDs use ``timestamp_us`` only, and arrival times do not need to be
  monotonic.
* A merged lidar or radar stores arrival times when all of its sensors do. A merged row carries the
  latest arrival time of its parts.

Where to add sensor I/O
~~~~~~~~~~~~~~~~~~~~~~~~~

If your dataset uses a point-cloud or image format not already supported, add a
loader under ``src/py123d/common/io/{camera,lidar,radar}/`` and add it in the
dispatcher
:func:`py123d.common.io.lidar.path_lidar_io.load_point_cloud_data_from_path`
(i.e. for lidar). The dispatcher routes by file extension / store option and is where
``load_kwargs`` is consumed.

Sync vs. async parsing
~~~~~~~~~~~~~~~~~~~~~~~~

For simplicity, the default configuration writes all modalities as synchronized frames.
Dataset with asynchronous modalities can be implemented by overriding the async iterator
:meth:`~py123d.parser.base_dataset_parser.BaseLogParser.iter_modalities_async`,
which yields each modality independently as it is produced. The default
implementation simply unwraps the synchronized frames, so overriding is optional.
Async writing is selected by the ``async_conversion`` flag in the config
(:ref:`section 5 <adding-datasets-hydra>`). Importantly, an individual modality must be parsed in
increasing timestamp order, but modalities can be interleaved in any way.


3. Map Parser
-------------
Maps are usually the trickiest part to unify since HD-map schemas vary widely across datasets.
Maps are **optional**: if your dataset ships none, return ``[]`` from ``get_map_parsers()`` and skip this section.
We generally recommend starting with log parsing and adding maps later.

A *map parser* is a handle to one map region. Subclass
:class:`~py123d.parser.base_dataset_parser.BaseMapParser` and implement:

.. code-block:: python

  from py123d.parser.base_dataset_parser import BaseMapParser

  class MyMapParser(BaseMapParser):

      def get_map_metadata(self) -> MapMetadata:
          ...

      def iter_map_objects(self):
          """Yield map objects lazily, one at a time."""
          yield from self._parse_lanes()
          yield from self._parse_crosswalks()
          # ...

Since map elements are static datatypes, they have no timestamp information and
are expected to be expressed in the logs global coordinate frame.

Map metadata
~~~~~~~~~~~~~

``get_map_metadata()`` returns a
:class:`~py123d.datatypes.metadata.map_metadata.MapMetadata`. Two fields deserve
particular attention:

* ``map_has_z``: whether the map geometry carries elevation (3D) or is 2D only.
* ``map_is_per_log``: whether each log has its own map (i.e., as in Waymo datasets) or maps
  are shared per location across logs (e.g., for each city in nuScenes/nuPlan).

Supported map elements
~~~~~~~~~~~~~~~~~~~~~~~~

``iter_map_objects()`` yields
:class:`~py123d.datatypes.map_objects.base_map_objects.BaseMapObject` instances.
Each object is one of the supported element types (the
:class:`~py123d.datatypes.map_objects.map_layer_types.MapLayer` enum), defined in
``src/py123d/datatypes/map_objects/map_objects.py``:

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Element
     - Geometry
     - Notes
   * - ``Lane``
     - surface
     - Individual driving lane with boundaries and centerline. Lane connectors are
       also yielded as ``Lane`` objects.
   * - ``LaneGroup``
     - surface
     - Group of lanes travelling in the same direction.
   * - ``Intersection``
     - surface
     - Intersection area.
   * - ``Crosswalk``
     - surface
     - Pedestrian crossing.
   * - ``Walkway``
     - surface
     - Pedestrian walkway.
   * - ``Carpark``
     - surface
     - Parking area.
   * - ``GenericDrivable``
     - surface
     - Other drivable surface.
   * - ``StopZone``
     - surface
     - Zone governed by a stop sign / light.
   * - ``RoadEdge``
     - line
     - Road boundary.
   * - ``RoadLine``
     - line
     - Painted lane marking.
   * - ``SpeedBump``
     - surface
     - Speed-reduction area.

Surface elements derive from ``BaseMapSurfaceObject`` (a polygon ``outline``), and
line elements derive from ``BaseMapLineObject`` (a ``polyline``).

.. note::
  If a map element type isn't listed above, then new layers can be added. Please
  open an issue or PR (see :doc:`contributing`) so the schema stays consistent
  across datasets.


4. Downloader (optional)
------------------------

A downloader is **optional**, but it can significantly lower the onboarding
barrier: A dataset or subset can be downloaded directly from the 123D CLI.
The existing implementations in 123D also show examples how to download and convert
in one script (i.e. using the `streaming during conversion` approach described below).


Subclass :class:`~py123d.parser.base_downloader.BaseDownloader`, placing it at
``src/py123d/parser/<dataset>/<dataset>_download.py``:

.. code-block:: python

  from py123d.parser.base_downloader import BaseDownloader

  class MyDatasetDownloader(BaseDownloader):

      def download(self) -> None:
          """Fetch the selected data into ``self.output_dir``."""
          ...

The base class provides ``output_dir`` (destination; ``None`` means a temp
directory is assigned at runtime) and ``dry_run`` (log the plan without writing).

Integration follows two paths:

* **Standalone CLI**: ``py123d-download dataset=mydataset`` instantiates the
  downloader and calls ``download()``.
* **Streaming during conversion**: the parser accepts an optional ``downloader=``
  argument, redirects its ``output_dir`` to a session-scoped temp directory,
  downloads before reading, and cleans up on destruction.


.. _adding-datasets-hydra:

5. Hydra config
---------------

Dataset Paths
~~~~~~~~~~~~~

123D centralizes the dataset roots that parsers read from in a single Hydra
config group at ``src/py123d/script/config/common/default_dataset_paths.yaml``,
typed by the :class:`~py123d.common.runtime.dataset_paths.DatasetPaths`
dataclass in ``src/py123d/common/runtime/dataset_paths.py``. For a new dataset,
add a ``mydataset_data_root: ${oc.env:MYDATASET_DATA_ROOT,null}`` entry to the
YAML.

By default, we use a environment variable (e.g. ``MYDATASET_DATA_ROOT``) to avoid hardcoding local paths in the config.
In your dataset config (see below), you can reference the path to pass it to the log/map parsers.


Dataset Parsers
~~~~~~~~~~~~~~~

Conversion configs live in
``src/py123d/script/config/conversion/dataset/``, one ``<name>.yaml`` per dataset
variant. The base defaults are in ``default_conversion.yaml``. Each dataset config
has four sections:

* ``parser``: instantiates your ``BaseDatasetParser`` (via ``_target_``) and its
  data-root paths / options.
* ``log_writer_config``: storage options specific to the arrow log writer, i.e. configures sensor storage
  (``camera_store_option``, ``lidar_store_option``, ``lidar_codec``),
  the ``async_conversion`` flag, and inference toggles.
* ``log_writer`` — the :class:`~py123d.api.scene.arrow.arrow_log_writer.ArrowLogWriter`,
  including a ``sync_config`` (the ``reference_column`` and ``direction`` used to
  snap modalities onto a common timeline).
* ``map_writer`` — the :class:`~py123d.api.map.arrow.arrow_map_writer.ArrowMapWriter`.

Provide a **local** config and, if you wrote a downloader, a separate **stream**
config. Keep them as distinct files as they differ in data roots and storage:

.. tab-set::

  .. tab-item:: Local (``mydataset.yaml``)

    Data is pre-downloaded; roots point at it and sensors are stored as paths.

    .. code-block:: yaml

      parser:
        _target_: py123d.parser.mydataset.mydataset_parser.MyDatasetParser
        _convert_: 'all'
        splits: ["mydataset_train", "mydataset_val"]
        mydataset_data_root: ${dataset_paths.mydataset_data_root}

      log_writer_config:
        _target_: py123d.api.scene.arrow.utils.log_writer_config.LogWriterConfig
        _convert_: 'all'
        camera_store_option: "path"   # "path", "jpeg_binary", "png_binary"
        lidar_store_option: "path"    # "path", "binary"
        lidar_codec: null
        async_conversion: ${async_conversion}
        # ...

  .. tab-item:: Stream (``mydataset-stream.yaml``)

    Roots are ``null`` (auto-detected in a temp dir); a ``downloader`` block is
    added and sensors are stored as **binary** (temp files vanish after
    conversion).

    .. code-block:: yaml

      parser:
        _target_: py123d.parser.mydataset.mydataset_parser.MyDatasetParser
        _convert_: 'all'
        splits: ["mydataset_train", "mydataset_val"]
        mydataset_data_root: null     # auto-detected
        downloader:
          _target_: py123d.parser.mydataset.mydataset_download.MyDatasetDownloader
          _convert_: 'all'
          output_dir: null            # temp dir assigned at runtime
          splits: ${dataset.parser.splits}
          dry_run: false

      log_writer_config:
        # ...
        camera_store_option: "jpeg_binary"
        lidar_store_option: "binary"
        lidar_codec: "laz"


6. Documentation
----------------

Finally, document the dataset so users know how to obtain, install, and convert
it:

#. Add a page ``docs/datasets/<name>.rst``: You can mirror an existing one. The established structure is: an *Overview*
   dropdown (paper, download, license, splits), an *Available Modalities* table, *Download*, *Installation* (the pip extra),
   *Conversion* (local + streaming), any dataset-specific sections, *Dataset
   Issues*, and *Citation*.
#. Register the page in the ``docs/datasets/index.rst`` toctree.
#. Build the docs locally and confirm the page renders without warnings before
   opening your PR.


Checklist
---------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Step
     - Location
   * - ``BaseDatasetParser`` subclass
     - ``src/py123d/parser/<dataset>/<dataset>_parser.py``
   * - ``BaseLogParser`` subclass
     - same package
   * - ``BaseMapParser`` subclass (if maps)
     - same package
   * - Sensor I/O loader (if new format)
     - ``src/py123d/common/io/{camera,lidar,radar}/``
   * - Optional dependency extra + gating
     - ``pyproject.toml`` + ``check_dependencies(...)``
   * - Downloader (optional)
     - ``src/py123d/parser/<dataset>/<dataset>_download.py``
   * - Hydra config pair
     - ``src/py123d/script/config/conversion/dataset/<name>{,-stream}.yaml``
   * - Dataset doc page + index entry
     - ``docs/datasets/<name>.rst`` + ``docs/datasets/index.rst``

Thank you for contributing a dataset to 123D! If anything in this guide is unclear
or you hit a case it doesn't cover, please open an issue or PR.
