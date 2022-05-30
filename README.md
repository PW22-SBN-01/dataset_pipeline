# Dataset Helpers

TODO:
1. Merged Dataset Recorder
2. Merged Dataset Iterator
3. Documentation and usage examples
4. Static Code Analysis (pylint, pycodestyle)
5. Dynamic Code Analysis 

## Dataset Iterators

The dataset is logged in 3 seperated sources:
1. Android Phone(s)
    a. CSV
    b. Video File
2. Panda (CAN BUS Data)
    a. CSV file only
3. DVR Video
    a. Multiple Video Streams
    b. Log files

The `dataset_iterators.py` file provides classes to easily access the recorded data as panda's `DataFrame` objects

### Pandas CSV Iterator

The raw CAN data comes in with the format `timestamp,CAN_ID,MESSAGE` and is logges as such for the sake of maintaining high speed data logging. The `PandaDatasetIterator` creates and caches a new CSV of the format `timestamp,ID1,ID2,...,IDn` which is easier to use for plotting.

### Phone Data Iterator

The Android App saved the IMU and GPS data in csv format along with a video to accompany it. The video has a known duration, start time and a fixed frame rate. Using these, the `PhoneDatasetIterator` can generate a mapping from timestamp to video frame.

## Dataset Recorders

## Panda Recorder

The Panda provides CAN frames from the vehicle upon request. The `PandaDatasetRecorder` class provides a wrapper to write said frames to disk.

## DVR Recorder

The DVR provides a RSTP video stream. The `DVRDatasetRecorder` class provides a wrapper to write said video streams from multiple channels to disk simultaneously. It internally uses `ffmpeg` and python multiprocessing to acheive the same. It will internally launch N processes (plus the main process) to record the N channels.

# Code Quality

## Static Code Analysis

```bash
python -m pylint dataset_helper.py
python -m pycodestyle dataset_helper.py
```

## Unit Testing

TODO