import glob
import os

ROOT_DATASET_DIR = "dataset"
DATASET_DIR = os.path.join(ROOT_DATASET_DIR, "dataset")
DATASET_LIST = sorted(glob.glob(DATASET_DIR + "/*"))
PANDA_DIR = os.path.join(ROOT_DATASET_DIR, "panda_logs")
PANDA_LIST = sorted(glob.glob(PANDA_DIR + "/*.csv"))

PANDA_CACHE_DIR = ".panda_cache"

WRITE_BUFFER_SIZE = 10

DVR_CAPTURE_FORMAT = "rtsp://{}:{}@{}:{}/cam/realmonitor?channel={}&subtype=0"
DVR_NUM_CAMS = 8
DVR_DIR = os.path.join(ROOT_DATASET_DIR, "dvr_logs")
DVR_SCALE_FACTOR = 0.2