.. _once:

ONCE
----

The ONCE dataset (One millioN sCenEs) is a multi-modal dataset from Huawei with a 40-beam roof lidar and seven cameras,
recorded in several Chinese cities. 123D converts the labelled splits: 20 sequences of 1 to 21 minutes with 3D bounding boxes
for five classes, released at 2 Hz. The 560 unlabelled sequences (about 2 TB) are not supported.


.. dropdown:: Overview
  :open:

  .. list-table::
    :header-rows: 0
    :widths: 20 60

    * -
      -
    * - :octicon:`file` Paper
      - `One Million Scenes for Autonomous Driving: ONCE Dataset <https://arxiv.org/abs/2106.11037>`_
    * - :octicon:`download` Download
      - `once-for-auto-driving.github.io/download.html <https://once-for-auto-driving.github.io/download.html>`_ (Google Drive and Baidu Yunpan)
    * - :octicon:`mark-github` Code
      - `github.com/once-for-auto-driving/once_devkit <https://github.com/once-for-auto-driving/once_devkit>`_
    * - :octicon:`law` License
      - `CC BY-NC-SA 4.0 <https://creativecommons.org/licenses/by-nc-sa/4.0/>`_
    * - :octicon:`database` Available splits
      - ``once_train`` (6 sequences), ``once_val`` (4 sequences), ``once_test`` (10 sequences, no boxes)


Sensor Overview
~~~~~~~~~~~~~~~

ONCE records the lidar and the cameras at 10 Hz but releases every fifth frame, so the lidar sweeps, all camera
images and the ego poses are at 2 Hz and share one timestamp per frame. The 3D boxes are annotated on every second
released frame, i.e. at 1 Hz. The ``interpolate_labels`` option of the parser (enabled by default) associates the boxes of
consecutive annotated frames into tracks and interpolates them at the unannotated frames in between, so the converted
logs carry boxes at 2 Hz. With ``dataset.parser.interpolate_labels=false`` the unannotated frames carry no boxes.

ONCE publishes no track ids. The track tokens in 123D come from the frame-to-frame association of the parser
(same class, nearest constant-velocity prediction within a class-specific speed gate), which is also what the
interpolation relies on.


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
     - The lidar pose relative to the first frame of the sequence, rotated into ISO 8855, see :class:`~py123d.datatypes.EgoStateSE3`.
       The roof lidar is the IMU origin. ONCE publishes no vehicle dimensions; the parameters are estimates for a mid-size car.
   * - Map
     - X
     - n/a
   * - Bounding Boxes
     - ✓
     - Bounding boxes are available with the :class:`~py123d.parser.registry.OnceBoxDetectionLabel`. For more information, see :class:`~py123d.datatypes.BoxDetectionsSE3`.
   * - Traffic Lights
     - X
     - n/a
   * - Cameras
     - ✓
     -
       ONCE has 7x :class:`~py123d.datatypes.Camera` (1920x1020, undistorted, calibration per sequence):

       - :class:`~py123d.datatypes.CameraID.PCAM_F0`: cam01 (front, wide)
       - :class:`~py123d.datatypes.CameraID.PCAM_F1`: cam03 (front, narrow)
       - :class:`~py123d.datatypes.CameraID.PCAM_L0`: cam06 (front left)
       - :class:`~py123d.datatypes.CameraID.PCAM_R0`: cam07 (front right)
       - :class:`~py123d.datatypes.CameraID.PCAM_L1`: cam05 (back left)
       - :class:`~py123d.datatypes.CameraID.PCAM_R1`: cam08 (back right)
       - :class:`~py123d.datatypes.CameraID.PCAM_B0`: cam09 (back)

       Cameras whose folder is missing under a sequence are skipped, so a lidar-only download converts as well.

   * - Lidars
     - ✓
     -
      ONCE has 1x :class:`~py123d.datatypes.Lidar`:

      - :class:`~py123d.datatypes.LidarID.LIDAR_TOP`: lidar_roof (40 beams, with intensity)


.. dropdown:: Dataset Specific

  .. autoclass:: py123d.parser.registry.OnceBoxDetectionLabel
    :members:
    :no-index:
    :no-inherited-members:


Download
~~~~~~~~

The archives are listed on the `download page <https://once-for-auto-driving.github.io/download.html>`_, one tar per split
and modality (``train_infos.tar``, ``train_lidar.tar``, ``train_cam01.tar``, ...; 22 GB for train, 13 GB for val, 31 GB for test).
All of them extract into the same ``data/`` tree. ``py123d-download`` fetches them from the Google Drive folders without an account:

.. code-block:: bash

  export ONCE_DATA_ROOT=/path/to/once
  py123d-download dataset=once                                   # train, val and test, all cameras (66 GB)
  py123d-download dataset=once 'dataset.downloader.splits=[once_val]' 'dataset.downloader.cameras=[cam01]'

Google Drive limits how much of a popular file can be downloaded per day ("quota exceeded"). The archives are fetched
in 256 MB range requests, which Google keeps serving for a while after plain downloads are refused, but a large archive
can still be cut off. A failed download can simply be rerun later; it resumes the partial archive and skips archives
that are already extracted.

The 123D conversion expects the following directory structure:

.. code-block:: text

  $ONCE_DATA_ROOT/
  └── data/
      ├── 000027/
      │   ├── 000027.json
      │   ├── lidar_roof/
      │   │   ├── 1616100800400.bin
      │   │   └── ...
      │   ├── cam01/
      │   │   ├── 1616100800400.jpg
      │   │   └── ...
      │   ├── cam03/
      │   ├── cam05/
      │   ├── cam06/
      │   ├── cam07/
      │   ├── cam08/
      │   └── cam09/
      ├── 000028/
      └── ...


Conversion
~~~~~~~~~~~~

**Local mode** — data already extracted to ``$ONCE_DATA_ROOT`` (see the `Download`_ section above):

.. code-block:: bash

  py123d-conversion dataset=once
  py123d-conversion dataset=once 'dataset.parser.splits=[once_val]' dataset.parser.interpolate_labels=false

.. note::
  The conversion of ONCE by default does not store sensor data in the logs, but only
  relative file paths. To change this behavior, you need to adapt the ``once.yaml``
  converter configuration.

**Streaming mode** — the archives of the selected splits are downloaded into
``$PY123D_DATA_ROOT/.download/once``, extracted, converted, and deleted when the conversion ends:

.. code-block:: bash

  py123d-conversion dataset=once-stream 'dataset.parser.splits=[once_val]'
  py123d-conversion dataset=once-stream 'dataset.parser.splits=[once_val]' 'dataset.parser.downloader.cameras=[]'

.. warning::
  ONCE ships one archive per split and modality, so even ``dataset.parser.log_names`` with a single sequence
  downloads the whole split. Use ``dataset.parser.downloader.cameras`` to drop cameras you do not need, and
  ``keep_downloads=true`` to keep the extracted data for another run.

.. note::
  Streaming mode forces ``camera_store_option: "jpeg_binary"`` and
  ``lidar_store_option: "binary"`` — the download directory is deleted after the
  conversion, so any ``"path"`` references would point at vanished sources.
