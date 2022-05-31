"""
DatasetHelper.py
    AndroidDatasetIterator
    PandaDatasetRecorder
"""

from calendar import c
import os
import subprocess
import signal
from datetime import datetime, timedelta
from multiprocessing import Process, Array, Pool, Queue
import time
import binascii

import cv2
import panda
from tqdm import tqdm


from .dataset_constants import *


class DVRDatasetRecorder:

    """
        DVRDatasetRecorder
    """

    def __init__(self, login_id, password, channels=list(range(1, DVR_NUM_CAMS+1)), ip_addr='192.168.0.111', port='554', scale_factor=DVR_SCALE_FACTOR) -> None:
        self.scale_factor = scale_factor
        self.dvr_url = {}
        self.channels = channels
        for channel in channels:
            print("Connecting to channel:", channel)
            dvr_url = DVR_CAPTURE_FORMAT.format(login_id, password, ip_addr, port, channel)
            self.dvr_url[channel] = dvr_url


    def start_rec(self):
        ffmpeg_procs = {}
        recording_id = 'DVR_' + str(datetime.fromtimestamp(time.time())).replace(" ", "_")
        recording_path = os.path.join(DVR_DIR, recording_id)
        os.makedirs(recording_path)
        for channel in self.channels:
            log_file_path = os.path.join(
                recording_path,
                recording_id + '_' + str(channel) + '.txt'
            )
            video_file_path = os.path.join(
                recording_path,
                recording_id + '_' + str(channel) +  '.mp4'
            )
            # Open log file
            log_file = open(log_file_path, 'w')
            # Launch the sim process
            proc_command = [
                #'ffmpeg', '-i', '"' + self.dvr_url[channel] + '"', 
                'ffmpeg', '-i', self.dvr_url[channel], 
                '-acodec', 'copy', '-vcodec', 'copy', video_file_path
            ]
            print(" ".join(proc_command))
            proc = subprocess.Popen(proc_command, stdout=log_file)
            ffmpeg_procs[channel] = proc
        try:
            print("Recording to: ", recording_path)
            while True:
                pass
        except KeyboardInterrupt:
            for channel in self.channels:
                print("Waiting for channel: ", channel)
                ffmpeg_procs[channel].send_signal(signal.SIGINT)
                #os.killpg(os.getpgid(ffmpeg_procs[channel].pid), signal.SIGTERM)
                ffmpeg_procs[channel].wait(timeout=10)
                #ffmpeg_procs[channel].terminate()
                os.killpg(os.getpgid(ffmpeg_procs[channel].pid), signal.SIGTERM)
            print("Stopped rec: ", recording_path)
   

    def __str__(self) -> str:
        return "DVRDatasetRecorder"

    def __repr__(self) -> str:
        return "DVRDatasetRecorder"


class AndroidDatasetRecorder:

    """
        AndroidDatasetRecorder
    """

    def __init__(self) -> None:
        # Start TCP Server
        # Default message to be 'standby'
        pass
    

    def start_rec(self, save_file_path=None):
        try:
            # Set Server message to 'start_rec @ T+3seconds'
            while True:
                pass

        except KeyboardInterrupt:
            # Set Server message to 'stop_rec'
            print("Stopped rec at: ", time.time())
   

    def __str__(self) -> str:
        return "AndroidDatasetRecorder"

    def __repr__(self) -> str:
        return "AndroidDatasetRecorder"

class MergedDatasetRecorder:

    """
        MergedDatasetRecorder
    """

    def __init__(self) -> None:
        # Init everything
        pass


    def start_rec(self, save_file_path=None):
        try:
            # start recording @ T+3seconds
            while True:
                pass

        except KeyboardInterrupt:
            # stop recording
            print("Stopped rec at: ", time.time())


    def __str__(self) -> str:
        return "MergedDatasetRecorder"

    def __repr__(self) -> str:
        return "MergedDatasetRecorder"

class PandaDatasetRecorder:

    """
        PandaDatasetRecorder
    """

    def __init__(self) -> None:
        self._p = panda.Panda()
        pass

    def get_frame(self):
        rec = self._p.can_recv()
        while len(rec)==0:
            rec = self._p.can_recv()
        return rec

    def start_rec(self, save_file_path=None):
        if not save_file_path:
            save_file_path = os.path.join(
                PANDA_DIR,
                'PANDA_' + str(datetime.fromtimestamp(time.time())).replace(" ", "_") + '.csv'
            )
        log_file = open(save_file_path, "w")
        log_file.write("timestamp,address,d1,dddat,d2\n")
        try:
            print("Recording to: ", save_file_path)
            while True:
                # TODO: Raise waring if no frames recieved in last 5 seconds
                frames = self.get_frame() 

                lines = ""
                for address, d1, dddat, d2 in frames:
                    timestamp = time.time()
                    lines += "{:.6f},{},{},{},{}\n".format(
                        timestamp, address, d1, binascii.hexlify(bytearray(dddat)), d2
                    )
                    lines_count = len(lines)
                    if lines_count > WRITE_BUFFER_SIZE:
                        t0 = time.time()
                        log_file.write(lines)
                        d = time.time() - t0
                        print("Wrote {} bytes in {:.6f} us".format(lines_count, d*10**6))

        except KeyboardInterrupt:
            log_file.close()
            print("Stopped rec: ", save_file_path)
   

    def __str__(self) -> str:
        return "PandaDatasetRecorder"

    def __repr__(self) -> str:
        return "PandaDatasetRecorder"


def get_rec_in_process(recorder):
    #if 'start_rec' not in dir(recorder) or not (recorder.start_rec is callable):
        #raise Exception("Not a recorder")
    rec_proc = Process(target=recorder.start_rec, args=())
    return rec_proc

def stop_process_safely(proc):
    proc.send_signal(signal.SIGINT)
    proc.wait(timeout=10)
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)

if __name__ == '__main__':
    username = os.environ.get("DVR_USERNAME")
    password = os.environ.get("DVR_PASSWORD")
    #d = DVRDatasetRecorder(username, password, channels=[1, ])
    d = DVRDatasetRecorder(username, password)
    d.start_rec()
    exit()