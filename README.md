# Dataset Helpers


![pylint workflow](https://github.com/PW22-SBN-01/dataset_pipeline/actions/workflows/pylint.yml/badge.svg)
![pypi workflow](https://github.com/PW22-SBN-01/dataset_pipeline/actions/workflows/pypi.yml/badge.svg)
![pytest workflow](https://github.com/PW22-SBN-01/dataset_pipeline/actions/workflows/pytest.yml/badge.svg)

[Raw Dataset Drive Link](https://drive.google.com/drive/folders/1ouZk8stDobJtpDvYKBPKg0bWNR6bIX-U?usp=sharing)

TODO:
1. Merged Dataset Recorder
2. Merged Dataset Iterator
3. Documentation and usage examples
4. Static Code Analysis (pylint, pycodestyle)
5. Dynamic Code Analysis 

# Project Structure

All the raw dataset files are in the `dataset/` directory.

```
├── dataset
│   ├── android
│   │   ├── 1652937970859
│   │   │   ├── 1652937970859.csv
│   │   │   └── 1652937970859.mp4
│   │   └── 1653972957447
│   │  	 ├── 1653972957447.csv
│   │  	 └── 1653972957447.mp4
│   ├── dvr_logs
│   └── panda_logs
│  	 ├── PANDA_2022-05-31_10:25:56.624274.csv
│  	 ├── PANDA_2022-05-31_19:31:36.973255.csv
│  	 ├── PANDA_2022-05-31_19:33:02.578401.csv
│  	 ├── PANDA_2022-05-31_19:35:44.368032.csv
│  	 ├── PANDA_2022-05-31_19:39:50.021715.csv
│  	 └── PANDA_2022-05-31_19:58:32.016402.csv
├── dataset_helper
│   ├── dataset_constants.py
│   ├── dataset_iterators.py
│   ├── dataset_recorders.py
│   ├── dataset_visualizer.py
│   └── __init__.py
├── README.ipynb
├── README.md
└── record.py
```

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

The Android App saved the IMU and GPS data in csv format along with a video to accompany it. The video has a known duration, start time and a fixed frame rate. Using these, the `AndroidDatasetIterator` can generate a mapping from timestamp to video frame.

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

```bash
python -m pytest --import-mode=append tests/
```