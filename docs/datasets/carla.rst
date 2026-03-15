.. _carla:

CARLA
-----

CARLA is an open-source simulator for autonomous driving research.
As such CARLA data is synthetic and can be generated with varying sensor and environmental conditions.
The following documentation is largely incomplete and merely describes the provided demo data.

.. note::
  Data from the CARLA simulator can be collected using the `LEAD framework <https://github.com/autonomousvision/lead>`_, which provides a state-of-the-art expert driver in CARLA.

.. dropdown:: Quick Links
  :open:

  .. list-table::
    :header-rows: 0
    :widths: 40 60

    * -
      -
    * - :octicon:`file` Paper
      - `CARLA: An Open Urban Driving Simulator <https://arxiv.org/abs/1711.03938>`_
    * - :octicon:`globe` Website
      - `carla.org/ <https://carla.org/>`_
    * - :octicon:`mark-github` Code
      - `github.com/carla-simulator/carla <https://github.com/carla-simulator/carla>`_
    * - :octicon:`law` License
      - MIT License
    * - :octicon:`database` Available splits
      - n/a


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
     - Depending on the collected dataset. For further information, see :class:`~py123d.datatypes.vehicle_state.EgoStateSE3`.
   * - Map
     - ✓
     - We included a conversion method of OpenDRIVE maps. For further information, see :class:`~py123d.api.MapAPI`.
   * - Bounding Boxes
     - ✓
     - Depending on the collected dataset. For further information, see :class:`~py123d.datatypes.detections.BoxDetectionsSE3`.
   * - Traffic Lights
     - X
     - n/a
   * - Cameras
     - ✓
     - Depending on the collected dataset. For further information, see :class:`~py123d.datatypes.sensors.Camera`.
   * - Lidars
     - ✓
     - Depending on the collected dataset. For further information, see :class:`~py123d.datatypes.sensors.Lidar`.


Download
~~~~~~~~

n/a

Installation
~~~~~~~~~~~~

n/a

Dataset Specific
~~~~~~~~~~~~~~~~

.. dropdown:: Box Detection Labels

  .. autoclass:: py123d.parser.registry.DefaultBoxDetectionLabel
    :members:
    :no-index:
    :no-inherited-members:


Dataset Issues
~~~~~~~~~~~~~~

n/a

Citation
~~~~~~~~

If you use CARLA in your research, please cite:

.. code-block:: bibtex

  @article{Dosovitskiy2017CORL,
    title = {{CARLA}: {An} Open Urban Driving Simulator},
    author = {Alexey Dosovitskiy and German Ros and Felipe Codevilla and Antonio Lopez and Vladlen Koltun},
    booktitle = {Proceedings of the 1st Annual Conference on Robot Learning},
    year = {2017}
  }
