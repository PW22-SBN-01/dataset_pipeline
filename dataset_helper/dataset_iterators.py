"""
DatasetHelper.py
    AndroidDatasetIterator
    PandaDatasetRecorder
"""

import os
from datetime import datetime, timedelta

import cv2
import pandas as pd
from tqdm import tqdm

from .dataset_constants import *

class PandaDatasetIterator:

    """
        PandaDatasetIterator
    """

    def __init__(self, csv_path=PANDA_LIST[-1], invalidate_cache=False) -> None:
        print("Init path:", csv_path)
        self.csv_path = csv_path
        self.folder_path = os.path.dirname(csv_path)
        cached_csv_folder = os.path.join(self.folder_path, PANDA_CACHE_DIR)
        os.makedirs(cached_csv_folder, exist_ok=True)
        self.cached_csv_path = os.path.join(cached_csv_folder, os.path.basename(csv_path))

        if not os.path.exists(self.cached_csv_path) or invalidate_cache:
            print("Generating Cache: ", self.cached_csv_path)
            self.csv_dat = pd.read_csv(self.csv_path)
            self.csv_dat = self.csv_dat.sort_values("timestamp")
            addresses = list(map(str, self.csv_dat['address'].unique()))
            columns = ["timestamp", ] + addresses
            reformatted_data = []
            #reformatted_data.append(columns)

            for index, row in tqdm(self.csv_dat.iterrows(), total=self.csv_dat.shape[0]):
                #if reformatted_data[-1][0] == row['timestamp']:
                if len(reformatted_data)>1 and row['timestamp'] - reformatted_data[-1][0] < 0.001:
                    reformatted_data[-1][
                        addresses.index(str(row['address']))+1
                    ] = (row['d1'], row['dddat'], row['d2'])
                else:
                    data_points = [row['timestamp'], ] + [None, ] * len(addresses)
                    if len(reformatted_data)>1:
                        data_points = [row['timestamp'], ] + reformatted_data[-1][1:]
                    data_points[
                        addresses.index(str(row['address']))+1
                    ] = (row['d1'], row['dddat'], row['d2'])
                    reformatted_data.append(data_points)

            reformatted_data = pd.DataFrame(reformatted_data)
            #reformatted_data.columns = ['index', ] + columns
            reformatted_data.columns = columns
            reformatted_data.set_index('timestamp')
            reformatted_data = reformatted_data.sort_values("timestamp")
            reformatted_data.to_csv(self.cached_csv_path, index=False)

        self.csv_dat = pd.read_csv(self.cached_csv_path)
        self.csv_dat.set_index('timestamp')

        self.start_time_csv = min(self.csv_dat['timestamp'])
        self.end_time_csv = max(self.csv_dat['timestamp'])

        self.duration_sec = (self.end_time_csv - self.start_time_csv)
        self.frame_count = self.csv_dat.shape[0]
        self.fps = self.frame_count / self.duration_sec
        self.line_no = 0

    def __iter__(self):
        self.line_no = 0
        return self

    def __next__(self) -> None:
        data = self.__getitem__(self.line_no)
        self.line_no += 1
        return data

    def __len__(self):
        return len(self.csv_dat)

    def __getitem__(self, key):
        if key > len(self):
            raise IndexError("Out of bounds; key=", key)
        timestamp = self.csv_dat.loc[key][0]
        return self.csv_dat.loc[key]

    def get_item_by_timestamp(self, timestamp, fault_delay=0.5):
        # TODO: Return frame closest to given timestamp
        # TODO: Raise exception if delta between timestamp and frame is greaterthan fault_delay
        pass

    def __str__(self) -> str:
        res = "----------------------------------------------------" + '\n'
        res += "PandaDatasetIterator('" + self.csv_path + "')" + '\n'
        res += "----------------------------------------------------" + '\n'
        res += "self.fps:        \t" + str(self.fps) + '\n'
        res += "self.frame_count:\t" + str(self.frame_count) + '\n'
        res += "self.start_time_csv:\t" + \
            str(datetime.fromtimestamp(self.start_time_csv)) + '\n'
        res += "self.end_time_csv:\t" + \
            str(datetime.fromtimestamp(self.end_time_csv)) + '\n'
        res += "self.duration:    \t" + \
            str(timedelta(seconds=self.duration_sec)) + '\n'
        res += "----------------------------------------------------"
        return res

    def __repr__(self) -> str:
        return str(self)


class AndroidDatasetIterator:

    """
        AndroidDatasetIterator
        Iterates through dataset, given the folder_path
    """

    def __init__(self, folder_path=DATASET_LIST[-1], scale_factor=1.0) -> None:
        print("Init path:", folder_path)
        self.folder_path = folder_path
        self.scale_factor = scale_factor
        self.old_frame_number = 0
        self.line_no = 0

        self.id = folder_path.split("/")[1]
        self.start_time = int(self.id)
        self.csv_path = os.path.join(folder_path, self.id + ".csv")
        self.mp4_path = os.path.join(folder_path, self.id + ".mp4")
        self.depth_mp4_path = os.path.join(folder_path, "depth_" + self.id + ".mp4")

        # CSV stores time in ms
        self.csv_dat = pd.read_csv(self.csv_path)
        self.csv_dat = self.csv_dat.sort_values("Timestamp")

        self.cap = cv2.VideoCapture(self.mp4_path)

        # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Computed video duration from FPS and number of video frames
        self.duration = self.frame_count / self.fps

        self.start_time_csv = min(self.csv_dat["Timestamp"])
        self.end_time_csv = max(self.csv_dat["Timestamp"])
        # Computed Duration the CSV file runs for
        self.expected_duration = (
            self.end_time_csv - self.start_time_csv
        ) / 1000.0

        # Expected FPS from CSV duration and number of frames
        self.expected_fps = self.frame_count / self.expected_duration
        # TODO: Perform Plausibility check on self.expected_fps and self.fps

    def __len__(self):
        return len(self.csv_dat)

    def __getitem__(self, key):
        if key > len(self):
            raise IndexError("Out of bounds; key=", key)
        timestamp = self.csv_dat.loc[key][0]
        time_from_start = timestamp - self.start_time_csv
        frame_number = round(time_from_start * self.fps / 1000)

        delta = abs(frame_number - self.old_frame_number)
        if frame_number >= self.old_frame_number and delta < 5:
            for _ in range(delta-1):
                ret, frame = self.cap.read()
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            print("cap.set: ", delta)

        self.old_frame_number = frame_number

        ret, frame = self.cap.read()
        if ret:
            w = int(frame.shape[1] * self.scale_factor)
            h = int(frame.shape[0] * self.scale_factor)
            final_frame = cv2.resize(frame, (w, h))
            return self.csv_dat.loc[key], final_frame
        
        raise IndexError(
            "Frame number not catured: ",
            frame_number,
            ", key=", key
        )

    def generate_depth_map(self):
        if not os.path.exists(self.depth_mp4_path):
            # TODO: Generate Depth Map data
            pass


    def get_item_by_timestamp(self, timestamp, fault_delay=0.5):
        # TODO: Return frame closest to given timestamp
        # TODO: Raise exception if delta between timestamp and frame is greaterthan fault_delay
        pass

    def __iter__(self):
        self.line_no = 0
        return self

    def __next__(self) -> None:
        data = self.__getitem__(self.line_no)
        self.line_no += 1
        return data

    def __str__(self) -> str:
        res = "----------------------------------------------------" + '\n'
        res += "AndroidDatasetIterator('" + self.folder_path + "')" + '\n'
        res += "----------------------------------------------------" + '\n'
        res += "self.fps:        \t" + str(self.fps) + '\n'
        res += "self.frame_count:\t" + str(self.frame_count) + '\n'
        res += "self.start_time_csv:\t" + \
            str(datetime.fromtimestamp(self.start_time_csv/1000)) + '\n'
        res += "self.end_time_csv:\t" + \
            str(datetime.fromtimestamp(self.end_time_csv/1000)) + '\n'
        res += "self.expected_duration:\t" + \
            str(timedelta(seconds=self.expected_duration)) + '\n'
        res += "self.expected_fps:\t" + str(self.expected_fps) + '\n'
        res += "----------------------------------------------------"
        return res

    def __repr__(self) -> str:
        return str(self)

    def __del__(self):
        self.cap.release()


class MergedDatasetIterator:

    """
        MergedDatasetIterator
        Iterates through dataset, given a AndroidDatasetIterator and a PandaDatasetIterator
    """

    def __init__(self, phone_iter: AndroidDatasetIterator, panda_iter: PandaDatasetIterator) -> None:
        print("phone_iter:", phone_iter)
        print("panda_iter:", panda_iter)
        self.phone_iter = phone_iter
        self.panda_iter = panda_iter
        # TODO: Compute the intersection of the two iters

    def __len__(self) -> int:
        # TODO: Return intersection size
        pass

    def __getitem__(self, key):
        if key > len(self):
            raise IndexError("Out of bounds; key=", key)
        
        # TODO: Return the key-th item
        
        raise IndexError(
            "key=", key
        )

    def __iter__(self):
        self.line_no = 0
        return self

    def __next__(self) -> None:
        data = self.__getitem__(self.line_no)
        self.line_no += 1
        return data

    def __str__(self) -> str:
        res = "----------------------------------------------------" + '\n'
        res += "MergedDatasetIterator('" + self.folder_path + "')" + '\n'
        res += "----------------------------------------------------" + '\n'
        res += "self.frame_count:\t" + str(self.len()) + '\n'
        res += "self.start_time_csv:\t" + \
            str(datetime.fromtimestamp(self.start_time_csv/1000)) + '\n'    # TODO
        res += "self.end_time_csv:\t" + \
            str(datetime.fromtimestamp(self.end_time_csv/1000)) + '\n'      # TODO
        res += "self.duration:\t" + \
            str(timedelta(seconds=self.duration)) + '\n'                    # TODO
        res += "self.fps:\t" + str(self.fps) + '\n'                         # TODO
        res += "----------------------------------------------------"
        return res

    def get_item_by_timestamp(self, timestamp, fault_delay=0.5):
        # TODO: Return frame closest to given timestamp
        # TODO: Raise exception if delta between timestamp and frame is greaterthan fault_delay
        pass

    def __repr__(self) -> str:
        return str(self)

    def __del__(self):
        pass


if __name__ == '__main__':
    d = PandaDatasetIterator()
    #d = PandaDatasetIterator('panda_logs/2022-05-24_21:58:04.466338.csv', )
    print(d)
    #for i in range(len(d)): print(d[i])
    exit()
    d = PandaDatasetRecorder()
    d.start_rec()
    exit()
    d = AndroidDatasetIterator(scale_factor=0.2)
    print(d)

    for i in range(len(d)):
        dat, frame = d[i]
        cv2.imshow('frame', frame)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
