.. _pandaset:

PandaSet
--------

The PandaSet dataset is a multi-modal dataset that includes data from cameras and Lidars, along with detailed 3D bounding box annotations.
It includes 103 logs of 8 second duration, resulting in about 0.2 hours of data.
PandaSet stands out, due to its no-cost commercial license.


.. dropdown:: Overview
  :open:

  .. list-table::
    :header-rows: 0
    :widths: 20 60

    * -
      -
    * - :octicon:`file` Paper
      - `PandaSet: Advanced Sensor Suite Dataset for Autonomous Driving <https://arxiv.org/abs/2112.12610>`_
    * - :octicon:`download` Download
      -
        - `scale.com/open-av-datasets/pandaset <https://scale.com/open-av-datasets/pandaset>`_ (official but discontinued).
        - `huggingface.co/datasets/georghess/pandaset <https://huggingface.co/datasets/georghess/pandaset>`_ (unofficial).
    * - :octicon:`mark-github` Code
      - `github.com/scaleapi/pandaset-devkit <https://github.com/scaleapi/pandaset-devkit>`_
    * - :octicon:`law` License
      -
        - `CC BY 4.0 <https://creativecommons.org/licenses/by/4.0/deed.en>`_
        - No-cost commercial license*
        - Apache License 2.0
    * - :octicon:`database` Available splits
      - n/a

.. dropdown:: Dataset Terms of Use*

    Dataset Terms of Use
    Scale AI, Inc. and Hesai Photonics Technology Co., Ltd and their affiliates (hereinafter "Licensors") strive to enhance public access to and use of data that they collect, annotate, and publish. The data are organized in datasets (the “Datasets”) listed at pandaset.org (the “Website”). The Datasets are collections of data, managed by Licensors and provided in a number of machine-readable formats. Licensors provide any individual or entity (hereinafter You” or “Your”) with access to the Datasets free of charge subject to the terms of this agreement (hereinafter “Dataset Terms”). Use of any data derived from the Datasets, which may appear in any format such as tables, charts, devkit, documentation, or code, is also subject to these Dataset Terms. By downloading any Datasets or using any Datasets, You are agreeing to be bound by the Dataset Terms. If you are downloading any Datasets or using any Datasets for an organization, you are agreeing to these Dataset Terms on behalf of that organization. If you do not have the right to agree to these Dataset Terms, do not download or use the Datasets.

    Licenses
    Unless specifically labeled otherwise, these Datasets are provided to You under a Creative Commons Attribution 4.0 International Public License (“CC BY 4.0”), with the additional terms included in these Dataset Terms. The CC BY 4.0 may be accessed at https://creativecommons.org/licenses/by/4.0/. When You download or use the Datasets from the Website or elsewhere, You are agreeing to comply with the terms of CC BY 4.0. Where these Dataset Terms conflict with the terms of CC BY 4.0, these Dataset Terms will control.

    Privacy
    Licensors prohibit You from using the Datasets in any manner to identify or invade the privacy of any person whose personally identifiable information or personal data may have been incidentally collected in the creation of this Dataset, even when such use is otherwise legal. An individual with any privacy concerns, including a request to remove your personally identifiable information or personal data from the Dataset, may contact us by sending an e-mail to privacy@scaleapi.com.

    No Publicity Rights
    You may not use the name, any trademark, official mark, official emblem, or logo of either Licensor, or any of either Licensor’s other means of promotion or publicity without the applicable Licensor’s prior written consent nor in any event to represent or imply an association or affiliation with a Licensor, except as required to comply with the attribution requirements of the CC BY 4.0 license.

    Termination
    Licensors may terminate Your access to all or any part of the Datasets or the Website at any time, with or without cause, with or without notice, effective immediately. All provisions of the Dataset Terms which by their nature should survive termination will survive termination, including, without limitation, warranty disclaimers, indemnity, and limitations of liability.

    Indemnification
    You will indemnify and hold Licensors harmless from and against any and all claims, loss, cost, expense, liability, or damage, including, without limitation, all reasonable attorneys’ fees and court costs, arising from (i) Your use or misuse of the Website or the Datasets; (ii) Your access to the Website; (iii) Your violation of the Dataset Terms; or (iv) infringement by You, or any third party using Your account, of any intellectual property or other right of any person or entity. Such losses, costs, expenses, damages, or liabilities will include, without limitation, all actual, general, special, indirect, incidental, and consequential damages.

    Dispute Resolution
    These Dataset Terms will be governed by and interpreted in accordance with the laws of California (excluding the conflict of laws rules thereof). All disputes under these Dataset Terms will be resolved in the applicable state or federal courts of San Francisco, California. You consent to the jurisdiction of such courts and waive any jurisdictional or venue defenses otherwise available.

    Miscellaneous
    You agree that it is Your responsibility to comply with all applicable laws with respect to Your use and publication of the Datasets or derivatives thereof, including any applicable privacy, data protection, security, and export control laws. These Dataset Terms constitute the entire agreement between You and Licensors with respect to the subject matter of these Dataset Terms and supersedes any prior or contemporaneous agreements whether written or oral. If a court of competent jurisdiction finds any term of these Dataset Terms to be unenforceable, the unenforceable term will be modified to reflect the parties’ intention and only to the extent necessary to make the term enforceable. The remaining provisions of these Dataset Terms will remain in effect. You may not assign these Dataset Terms without the prior written consent of the Licensors. The Licensors may assign, transfer, or delegate any of their rights and obligations under these Dataset Terms without consent. The parties are independent contractors. No failure or delay by either party in exercising a right under these Dataset Terms will constitute a waiver of that right. A waiver of a default is not a waiver of any subsequent default. These Dataset Terms may be amended by the Licensors from time to time in our discretion. If an update affects your use of the Dataset, Licensors will notify you before the updated terms are effective for your use.


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
     - The poses and vehicle parameters are provided or inferred from the documentation, see :class:`~py123d.datatypes.vehicle_state.EgoStateSE3`.
   * - Map
     - X
     - n/a
   * - Bounding Boxes
     - ✓
     - Bounding boxes are available with the :class:`~py123d.parser.registry.PandasetBoxDetectionLabel`. For more information, see :class:`~py123d.datatypes.detections.BoxDetectionsSE3`.
   * - Traffic Lights
     - X
     - n/a
   * - Cameras
     - ✓
     -
       Pandaset has 6x :class:`~py123d.datatypes.sensors.Camera`:

       - :class:`~py123d.datatypes.sensors.CameraID.PCAM_F0`: front_camera
       - :class:`~py123d.datatypes.sensors.CameraID.PCAM_L0`: front_left_camera
       - :class:`~py123d.datatypes.sensors.CameraID.PCAM_R0`: front_right_camera
       - :class:`~py123d.datatypes.sensors.CameraID.PCAM_L1`: left_camera
       - :class:`~py123d.datatypes.sensors.CameraID.PCAM_R1`: right_camera
       - :class:`~py123d.datatypes.sensors.CameraID.PCAM_B0`: back_camera

   * - Lidars
     - ✓
     -
      Pandaset has 2x :class:`~py123d.datatypes.sensors.Lidar`:

      - :class:`~py123d.datatypes.sensors.LidarID.LIDAR_TOP`: main_pandar64
      - :class:`~py123d.datatypes.sensors.LidarID.LIDAR_FRONT`: front_gt


.. dropdown:: Dataset Specific

  .. autoclass:: py123d.parser.registry.PandasetBoxDetectionLabel
    :members:
    :no-index:
    :no-inherited-members:


Download
~~~~~~~~

Since a few years, the official PandaSet dataset download no longer available (see `Issue #151 <https://github.com/scaleapi/pandaset-devkit/issues/151>`_).
However, unofficial copies of the dataset can be found on `Hugging Face <https://huggingface.co/datasets/georghess/pandaset>`_ or `Kaggle <https://www.kaggle.com/datasets/usharengaraju/pandaset-dataset/dataset>`_.

The 123D conversion expects the following directory structure:

.. code-block:: text

  $PANDASET_DATA_ROOT/
  ├── 001/
  │   ├── annotations/
  │   │   ├── cuboids/
  │   │   │   ├── 00.pkl.gz
  │   │   │   ├── ...
  │   │   │   └── 79.pkl.gz
  │   │   └── semseg/  (currently not used)
  │   │       ├── 00.pkl.gz
  │   │       ├── ...
  │   │       ├── 79.pkl.gz
  │   │       └── classes.json
  │   ├── camera/
  │   │   ├── back_camera/
  │   │   │   ├── 00.jpg
  │   │   │   ├── ...
  │   │   │   ├── 79.jpg
  │   │   │   ├── intrinsics.json
  │   │   │   ├── poses.json
  │   │   │   └── timestamps.json
  │   │   ├── front_camera/
  │   │   │   └── ...
  │   │   ├── front_left_camera/
  │   │   │   └── ...
  │   │   ├── front_right_camera/
  │   │   │   └── ...
  │   │   ├── left_camera/
  │   │   │   └── ...
  │   │   └── right_camera/
  │   │       └── ...
  │   ├── LICENSE.txt
  │   ├── lidar/
  │   │   ├── 00.pkl.gz
  │   │   ├── ...
  │   │   ├── 79.pkl.gz
  │   │   ├── poses.json
  │   │   └── timestamps.json
  │   └── meta/
  │       ├── gps.json
  │       └── timestamps.json
  ├── ...
  └── 158/
      └── ...



Installation
~~~~~~~~~~~~

No additional installation steps are required beyond the standard ``py123d`` installation.


Conversion
~~~~~~~~~~~~

You can convert the PandaSet by running:

.. code-block:: bash

  py123d-conversion datasets=["pandaset"]

.. note::
  The conversion of PandaSet by default does not store sensor data in the logs, but only relative file paths.
  To change this behavior, you need to adapt the ``pandaset.yaml`` converter configuration.


Dataset Issues
~~~~~~~~~~~~~~

* **Ego Vehicle:** The ego vehicle parameters are estimates from the vehicle model. The exact location of the IMU/GPS sensor and the bounding box dimensions of the ego vehicle may not be accurate.
* **Bounding Boxes:** PandaSet provides bounding boxes that fall in the overlap of the Lidar region twice (for each point cloud). The current implementation only uses the bounding boxes of the top Lidar sensor.
* **Lidar:** PandaSet does not motion compensate the Lidar sweeps (in contrast to other datasets). Artifacts remain visible.

Citation
~~~~~~~~

If you use PandaSet in your research, please cite:

.. code-block:: bibtex

  @article{Xiao2021ITSC,
    title={Pandaset: Advanced sensor suite dataset for autonomous driving},
    author={Xiao, Pengchuan and Shao, Zhenlei and Hao, Steven and Zhang, Zishuo and Chai, Xiaolin and Jiao, Judy and Li, Zesong and Wu, Jian and Sun, Kai and Jiang, Kun and others},
    booktitle={2021 IEEE international intelligent transportation systems conference (ITSC)},
    year={2021},
  }
