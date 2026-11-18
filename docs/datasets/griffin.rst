.. _griffin:

Griffin
-------

Griffin is an aerial-ground cooperative perception dataset collected in CARLA, pairing a
ground vehicle with a UAV (drone) for cooperative 3D detection and tracking. It contains
255 scenes (~15 s each, ~150 frames at 10 Hz) across four UAV-altitude subsets, totaling
over 37.7k samples and 914.8k 3D annotations.

The 123D parser converts the **vehicle-side** (ground agent): four cardinal pinhole
cameras, the 80-beam top LiDAR, ego poses, and 3D boxes. Each scene becomes one log,
routed to its official ``train`` / ``val`` partition.

.. note::
  The **drone-side** (UAV) agent is also available as a *separate single-agent* conversion
  (``dataset=griffin_drone``): five pinhole cameras (four cardinal + one nadir), ego poses, and
  3D boxes, with aerial state (UAV flag, per-frame altitude) in a custom modality. The drone
  carries no LiDAR. True first-class cooperation (vehicle + drone with cross-agent transforms
  as a core datatype) remains a design-first follow-up; the vehicle and drone logs of a scene
  already share one global (ENU) frame, so they can be aligned by scene id + timestamp.

.. dropdown:: Overview
  :open:

  .. list-table::
    :header-rows: 0
    :widths: 20 60

    * -
      -
    * - :octicon:`file` Paper
      - `Griffin: Aerial-Ground Cooperative Detection and Tracking Dataset and Benchmark <https://arxiv.org/abs/2503.06983>`_ (arXiv preprint, 2025)
    * - :octicon:`download` Download
      -
        - `huggingface.co/datasets/wjh-svm/Griffin <https://huggingface.co/datasets/wjh-svm/Griffin>`_
        - `Baidu Netdisk <https://pan.baidu.com/s/1NDgsuHB-QPRiROV73NRU5g?pwd=u3cm>`_
    * - :octicon:`mark-github` Code
      - `github.com/wang-jh18-SVM/Griffin <https://github.com/wang-jh18-SVM/Griffin>`_
    * - :octicon:`law` License
      - `MIT License <https://github.com/wang-jh18-SVM/Griffin/blob/main/LICENSE.txt>`_
    * - :octicon:`database` Available splits
      - ``griffin_{50scenes_25m,50scenes_40m,50scenes_55m,100scenes_random}_{train,val}``


Available Modalities
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 5 70

   * - **Name**
     - **Available**
     - **Description**
   * - Ego Vehicle
     - ✓
     - Ego poses are provided in an ENU world frame; vehicle parameters follow the CARLA
       ego model, see :class:`~py123d.datatypes.vehicle_state.EgoStateSE3`.
   * - Map
     - ✓
     - Griffin scenes are rendered on four stock CARLA towns (``Town03``, ``Town06``,
       ``Town07``, ``Town10HD``). The bundled OpenDRIVE town maps are converted as global
       maps under the ``griffin`` dataset; each log links to its town via ``location``.
       Griffin's global (ENU) frame coincides with the CARLA map frame, so no additional
       transform is required.
   * - Bounding Boxes
     - ✓
     - Bounding boxes are available with the :class:`~py123d.parser.registry.GriffinBoxDetectionLabel`.
       For more information, see :class:`~py123d.datatypes.detections.BoxDetectionsSE3`.
   * - Traffic Lights
     - (✓)
     - No light states are recorded. The CARLA maps provide the traffic lights as :class:`~py123d.datatypes.TrafficLight` objects, placed at the signal post of the OpenDRIVE ``<signal>``.
   * - Cameras
     - ✓
     -
       Griffin (vehicle-side) has 4x :class:`~py123d.datatypes.sensors.Camera`:

       - :class:`~py123d.datatypes.sensors.CameraID.PCAM_F0`: front
       - :class:`~py123d.datatypes.sensors.CameraID.PCAM_B0`: back
       - :class:`~py123d.datatypes.sensors.CameraID.PCAM_L0`: left
       - :class:`~py123d.datatypes.sensors.CameraID.PCAM_R0`: right

   * - Lidars
     - ✓
     -
      Griffin (vehicle-side) has 1x :class:`~py123d.datatypes.sensors.Lidar`:

      - :class:`~py123d.datatypes.sensors.LidarID.LIDAR_TOP`: lidar_top (80-beam, 10 Hz, ego frame)


.. dropdown:: Dataset Specific

  .. autoclass:: py123d.parser.registry.GriffinBoxDetectionLabel
    :members:
    :no-index:
    :no-inherited-members:


Download
~~~~~~~~

Download the dataset from `Hugging Face <https://huggingface.co/datasets/wjh-svm/Griffin>`_
or `Baidu Netdisk <https://pan.baidu.com/s/1NDgsuHB-QPRiROV73NRU5g?pwd=u3cm>`_ and extract
the KITTI-style ``griffin-release`` trees. The 123D conversion expects the official
directory structure with subset folders directly under ``$GRIFFIN_DATA_ROOT``:

.. code-block:: text

  $GRIFFIN_DATA_ROOT/
  ├── griffin_50scenes_25m/
  │   └── griffin-release/
  │       └── vehicle-side/
  │           ├── calib/            # {front,back,left,right,lidar_top}.json (extrinsic + intrinsic)
  │           ├── camera/
  │           │   ├── front/        # {frame}.png  (frame = 6-digit, e.g. 000620)
  │           │   ├── back/
  │           │   ├── left/
  │           │   └── right/
  │           ├── label/            # {frame}.txt  (ego-frame 3D boxes)
  │           ├── lidar/
  │           │   └── lidar_top/    # {frame}.ply  (ego-frame, intensity field "I")
  │           ├── pose/             # {frame}.json (ENU pose, degrees)
  │           └── scene_infos.json
  ├── griffin_50scenes_40m/
  │   └── ...
  ├── griffin_50scenes_55m/
  │   └── ...
  └── griffin_100scenes_random/
      └── ...

The official ``train`` / ``val`` partitions are embedded in the package
(:mod:`py123d.parser.griffin.splits`), so only the ``griffin-release`` trees are required.


Installation
~~~~~~~~~~~~~

Griffin conversion requires the ``griffin`` extras group (``plyfile`` for the LiDAR
``.ply`` files and ``scipy`` for the pose/box Euler-angle conversion):

.. code-block:: bash

  pip install py123d[griffin]


Conversion
~~~~~~~~~~~

Data already extracted to ``$GRIFFIN_DATA_ROOT`` (see `Download`_):

.. code-block:: bash

  py123d-conversion dataset=griffin

Convert a different subset or partition by overriding ``splits``:

.. code-block:: bash

  py123d-conversion dataset=griffin \
      'dataset.parser.splits=[griffin_100scenes_random_train, griffin_100scenes_random_val]'

.. note::
  By default the conversion stores relative sensor file paths rather than the sensor data
  itself. To embed sensors, adapt the ``lidar_store_option`` / ``camera_store_option`` in
  the ``griffin.yaml`` converter configuration.


Dataset Issues
~~~~~~~~~~~~~~~

* **Ego Vehicle:** Vehicle parameters are representative CARLA values; the ego frame is
  taken to coincide with the IMU frame.
* **Coordinate frames:** LiDAR points and labels are stored in the ego frame (X-forward,
  Y-left, Z-up); poses use an ENU world frame with ``xyz`` Euler angles in degrees. The
  parser lifts boxes to the global frame and keeps LiDAR points ego-relative.
* **Labels:** Non-traffic CARLA categories (e.g. military props) are outside the Griffin
  perception taxonomy and are skipped during conversion.
* **Drone-side:** Converted separately (``dataset=griffin_drone``) as a single-agent UAV log:
  5 cameras incl. a nadir ``bottom`` camera (:class:`~py123d.datatypes.sensors.CameraID.PCAM_D0`),
  no LiDAR, and aerial state in a :class:`~py123d.datatypes.custom.CustomModality` (id ``aerial``).
  Cooperative (joint vehicle + drone) perspectives are not modelled as a first-class type yet.

Citation
~~~~~~~~

If you use Griffin in your research, please cite:

.. code-block:: bibtex

  @misc{wang2025griffin,
    title={Griffin: Aerial-Ground Cooperative Detection and Tracking Dataset and Benchmark},
    author={Jiahao Wang and Xiangyu Cao and Jiaru Zhong and Yuner Zhang and Haibao Yu and Lei He and Shaobing Xu},
    year={2025},
    eprint={2503.06983},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2503.06983},
  }
