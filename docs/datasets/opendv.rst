.. _opendv:

OpenDV-YouTube
--------------

OpenDV-YouTube is the YouTube part of OpenDV-2K, the dataset of the GenAD paper: more than 1700 hours of front-view
driving videos from more than 244 cities in 40 countries. The dataset is released as a list of YouTube videos, which are
downloaded with the OpenDV toolkit. 123D converts each downloaded video into one camera-only log with 10 images per second.


.. dropdown:: Overview
  :open:

  .. list-table::
    :header-rows: 0
    :widths: 20 60

    * -
      -
    * - :octicon:`file` Paper
      - `Generalized Predictive Model for Autonomous Driving <https://arxiv.org/abs/2403.09630>`_
    * - :octicon:`download` Download
      - `OpenDV toolkit <https://github.com/OpenDriveLab/DriveAGI/tree/main/opendv>`_ (video list and download scripts)
    * - :octicon:`mark-github` Code
      - `github.com/OpenDriveLab/DriveAGI <https://github.com/OpenDriveLab/DriveAGI>`_
    * - :octicon:`law` License
      - The toolkit is released under Apache 2.0. The videos stay under the YouTube license and are not redistributed.
    * - :octicon:`database` Available splits
      - ``opendv_train``, ``opendv_val``


Available Modalities
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 5 70

   * - **Name**
     - **Available**
     - **Description**
   * - Ego Vehicle
     - X
     - n/a (no ego motion is known for the videos)
   * - Map
     - X
     - n/a
   * - Bounding Boxes
     - X
     - n/a
   * - Traffic Lights
     - X
     - n/a
   * - Cameras
     - ✓
     -
       OpenDV-YouTube has 1x :class:`~py123d.datatypes.Camera` at 10 images per second, in the resolution of the video:

       - :class:`~py123d.datatypes.CameraID.PCAM_F0`: youtube_front (front view)

       The camera has no intrinsics (unknown for YouTube videos), an identity extrinsic and no global pose.

   * - Lidars
     - X
     - n/a


Frame Selection
~~~~~~~~~~~~~~~

The video list gives, per video, a number of seconds to discard at the beginning and at the end (title screens, parking).
The toolkit extracts 10 images per second from the rest. The parser keeps the same images, read from the video
without writing image files:

- The grid times are ``start_discard + k x 100 ms`` after the first frame of the video, up to the discarded part at the end.
- Per grid time, the decoded frame nearest in presentation time is kept if it lies at most 50 ms away. Each frame is kept at most once.
- The timestamp of an image is the time of its frame in microseconds after the first frame of the video.

The name of a log is the YouTube ID of its video.


Download
~~~~~~~~

The videos are downloaded with the `OpenDV toolkit <https://github.com/OpenDriveLab/DriveAGI/tree/main/opendv>`_.
Download the `OpenDV-YouTube Google Sheet <https://docs.google.com/spreadsheets/d/1bHWWP_VXeEe5UzIG-QgKFBdH7mNlSC4GFSJkEhFnt2I>`_
as CSV, convert it to the JSON video list and run the download script:

.. code-block:: bash

  export OPENDV_DATA_ROOT=/path/to/opendv

  git clone https://github.com/OpenDriveLab/DriveAGI.git
  cd DriveAGI/opendv
  pip install -r requirements.txt

  # The Google Sheet, saved as meta/OpenDV-YouTube.csv:
  python scripts/meta_preprocess.py -i meta/OpenDV-YouTube.csv -o $OPENDV_DATA_ROOT/meta/OpenDV-YouTube.json

  # In configs/download.json, set "video_list" to the JSON file above and "root" to $OPENDV_DATA_ROOT/videos. Then:
  python scripts/youtube_download.py            # all videos, about 3 TB
  python scripts/youtube_download.py --mini     # or OpenDV-mini only: about 28 hours, 44 GB

The frame extraction of the toolkit (``scripts/video2img.py``) is not needed. The parser expects the following structure:

.. code-block:: none

  $OPENDV_DATA_ROOT
  ├── meta/
  │   └── OpenDV-YouTube.json
  └── videos/
      └── {youtuber}/                      (spaces replaced by "_")
          └── {video_id}.{mp4,webm,mkv}


Installation
~~~~~~~~~~~~

No additional installation steps are required beyond the standard ``py123d`` installation.


Conversion
~~~~~~~~~~

To run the conversion, you need to set the environment variable ``$OPENDV_DATA_ROOT``.
The parser supports the asynchronous conversion only, so ``async_conversion=true`` is required:

.. code-block:: bash

  py123d-conversion dataset=opendv async_conversion=true \
  dataset_paths.opendv_data_root=$OPENDV_DATA_ROOT # optional if env variable is set

  # Selected videos, or the first videos of the list:
  py123d-conversion dataset=opendv async_conversion=true 'dataset.parser.video_ids=[VIDEO_ID_1,VIDEO_ID_2]'
  py123d-conversion dataset=opendv async_conversion=true dataset.parser.max_videos=10

.. note::
  By default, the conversion stores the images as JPEG binary in the logs. With ``codec=kedec``, the images of a log are
  stored as one AV1 video instead (see :doc:`/notes/kedec`):

  .. code-block:: bash

    py123d-conversion dataset=opendv codec=kedec async_conversion=true


Dataset Issues
~~~~~~~~~~~~~~

- **Camera only:** The logs have no calibration, ego states, boxes or map.
- **Missing videos:** Videos of the list that are not on disk are skipped with a warning, so a partial download
  (e.g. OpenDV-mini) converts as well. Some videos have been removed from YouTube since the release.
- **Image size:** The toolkit downloads each video in its own resolution (720p to 1080p where available), so the image
  size differs between logs.
- **Skipped grid times:** A grid time without a frame within 50 ms has no image, e.g. in a gap of the video.
- **Language annotations:** The commands and contexts of `OpenDV-YouTube-Language <https://huggingface.co/datasets/OpenDriveLab/OpenDV-YouTube-Language>`_ are not converted.


Citation
~~~~~~~~

If you use OpenDV-YouTube in your research, please cite:

.. code-block:: bibtex

  @inproceedings{yang2024genad,
    title={Generalized Predictive Model for Autonomous Driving},
    author={Jiazhi Yang and Shenyuan Gao and Yihang Qiu and Li Chen and Tianyu Li and Bo Dai and Kashyap Chitta and Penghao Wu and Jia Zeng and Ping Luo and Jun Zhang and Andreas Geiger and Yu Qiao and Hongyang Li},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2024}
  }
