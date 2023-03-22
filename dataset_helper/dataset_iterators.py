"""
DatasetHelper.py
	AndroidDatasetIterator
	PandaDatasetRecorder
"""

from operator import inv
import os
from datetime import datetime, timedelta
import binascii
import math
import numpy as np
import sys
import pathlib
import pickle

import cv2
import pandas as pd
from tqdm import tqdm
import cantools
import yaml
import open3d as o3d
from scipy.spatial.transform import Rotation

from .dataset_constants import *
from . import helper

class BengaluruDepthDatasetIterator:
	
	def __init__(self, dataset_path="~/Datasets/Depth_Dataset_Bengaluru/1653972957447", settings_doc="calibration/pocoX3/calib.yaml") -> None:
		self.dataset_path = os.path.expanduser(dataset_path)
		self.dataset_id = self.dataset_path.split("/")[-1]
		self.rgb_img_folder = os.path.join(self.dataset_path, "rgb_img")
		self.depth_img_folder = os.path.join(self.dataset_path, "depth_img")
		self.csv_path = os.path.join(self.dataset_path, self.dataset_id + ".csv")
		
		os.path.isdir(self.dataset_path)
		os.path.isdir(self.rgb_img_folder)
		os.path.isdir(self.depth_img_folder)
		os.path.isfile(self.csv_path)
	
		self.settings_doc = settings_doc
		with open(self.settings_doc, 'r') as stream:
			try:
				self.cam_settings = yaml.load(stream, Loader=yaml.FullLoader)
			except yaml.YAMLError as exc:
				print(exc)
		k1 = self.cam_settings['Camera.k1']
		k2 = self.cam_settings['Camera.k2']
		p1 = self.cam_settings['Camera.p1']
		p2 = self.cam_settings['Camera.p2']
		k3 = 0
		if 'Camera.k3' in self.cam_settings:
			k3 = self.cam_settings['Camera.k3']
		self.DistCoef = np.array([k1, k2, p1, p2, k3])
		self.intrinsic_matrix = np.array([
			[self.cam_settings['Camera.fx']	, 0.0							, self.cam_settings['Camera.cx']],
			[0.0							, self.cam_settings['Camera.fy'], self.cam_settings['Camera.cy']],
			[0.0							, 0.0							, 1.0]
		])

		self.width = self.cam_settings['Camera.width']
		self.height = self.cam_settings['Camera.height']

		self.csv_dat = pd.read_csv(self.csv_path)
		
	def __iter__(self):
		self.line_no = 0
		return self

	def __next__(self):
		data = self.__getitem__(self.line_no)
		self.line_no += 1
		return data

	def __len__(self):
		return len(self.csv_dat)

	def __getitem__(self, key):
		if key > len(self):
			raise IndexError("Out of bounds; key=", key)
		csv_frame = self.csv_dat.loc[key]
		timestamp = str(int(csv_frame[1]))
		# timestamp = str(csv_frame[1])
		disparity_frame_path = os.path.join(self.depth_img_folder, timestamp + ".png")
		rgb_frame_path = os.path.join(self.rgb_img_folder, timestamp + ".png")
		
		assert os.path.isfile(disparity_frame_path), "File missing " + disparity_frame_path
		assert os.path.isfile(rgb_frame_path), "File missing " + rgb_frame_path

		disparity_frame = cv2.imread(disparity_frame_path)
		rgb_frame = cv2.imread(rgb_frame_path)

		disparity_frame = cv2.cvtColor(disparity_frame, cv2.COLOR_BGR2GRAY)
		
		frame = {
			'rgb_frame': rgb_frame,
			'disparity_frame': disparity_frame,
		}
		
		for key in csv_frame.keys():
			frame[key] = csv_frame[key]
		return frame

class BengaluruOccupancyDatasetIterator(BengaluruDepthDatasetIterator):

	def __init__(self, dataset_path="~/Datasets/Depth_Dataset_Bengaluru/1653972957447", settings_doc="calibration/pocoX3/calib.yaml") -> None:
		super().__init__(dataset_path, settings_doc)

		self.baseline = 1.0
		self.fx = self.intrinsic_matrix[0,0]
		self.fy = self.intrinsic_matrix[1,1]
		self.cx = self.intrinsic_matrix[0,2]
		self.cy = self.intrinsic_matrix[1,2]
		self.focal_length = self.fx

		self.intrinsics = o3d.camera.PinholeCameraIntrinsic(
			width=self.width, height=self.height,
			intrinsic_matrix=self.intrinsic_matrix
		)

		self.transformation = np.eye(4,4)
		# self.transformation[:3,:3] = Rotation.from_euler("xyz", (-1.70000000e+02,  4.83655339e-15, -6.46097591e-14),degrees=True).as_matrix() # Great Top Down View
		# self.transformation[:3,3] = [10000.0, 0.0, 0.0]
		self.transformation[3,:3] = [0.0, 0.0, 10.0]

	def __getitem__(self, key):
		frame = super().__getitem__(key)
		disparity = frame['disparity_frame'].astype(np.float32)
		rgb_frame = cv2.cvtColor(frame['rgb_frame'], cv2.COLOR_BGR2RGB)

		depth = self.baseline * self.focal_length * np.reciprocal(disparity)
		depth[np.isinf(depth)] = self.baseline * self.focal_length
		# depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
		depth = depth.astype(np.float32)

		print('depth.dtype', depth.dtype, np.max(depth), np.min(depth))
		print('disparity.dtype', disparity.dtype, np.max(disparity), np.min(disparity))

		# depth[depth>1250.6] = float('inf')
		# depth[depth>50.0] = float('inf')
		depth[depth>100.0] = float('inf')
		# depth[
		# 	:,
		# 	0:depth.shape[1]//2
		# ] = float('inf')
		depth[
			0:depth.shape[0]//2
			:,
		] = float('inf')

		# U, V = np.ix_(np.arange(self.image_resolution[1]), np.arange(self.image_resolution[0])) # pylint: disable=unbalanced-tuple-unpacking
		# Z = depth.copy()
		
		# X = (V - self.cx) * Z / self.fx
		# Y = (U - self.cy) * Z / self.fy

		# X = X.flatten()
		# Y = Y.flatten()
		# Z = Z.flatten()

		# points = np.array([X,Y,Z]).T

		# print(X.shape)
		# print(points.shape)

		rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
			o3d.geometry.Image(rgb_frame), o3d.geometry.Image(depth),
			depth_scale=10**2,
			convert_rgb_to_intensity=False
		)
		pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
			rgbd, self.intrinsics
		)

		pcd.remove_non_finite_points()
		pcd.transform(self.transformation)

		points = np.asarray(pcd.points)

		print('points.shape', points.shape)

		frame['disparity'] = disparity
		frame['depth'] = depth
		frame['points'] = points
		frame['pcd'] = pcd

		return frame

class PandaDatasetIterator:

	"""
		PandaDatasetIterator
	"""

	def __init__(self, csv_path, dbc_interp, invalidate_cache=False) -> None:
		print("Init path:", csv_path)
		assert type(dbc_interp) == DBCInterpreter
		self.dbc_interp = dbc_interp 
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

	def __next__(self):
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

	def get_item_by_timestamp(self, timestamp, fault_delay=1.0):
		"""
		Return frame closest to given timestamp
		Raise exception if delta between timestamp and frame is greaterthan fault_delay
		"""
		closest_frames = self.get_item_between_timestamp(timestamp-fault_delay, timestamp+fault_delay, fault_delay=float('inf'))
		closest_frames = closest_frames.reset_index(drop=True)
		closest_frame = closest_frames.iloc[(closest_frames['timestamp']-timestamp).abs().argsort()[0]]
		closest_ts = closest_frame['timestamp']
		if abs(timestamp - closest_ts) > fault_delay:
			raise Exception("No such timestamp, fault delay exceeded: abs(timestamp - closest_ts)=" + str(abs(timestamp - closest_ts)))
		return closest_frame

	def get_item_between_timestamp(self, start_ts, end_ts, fault_delay=0.5):
		"""
		Return frame between two given timestamps
		Raise exception if delta between start_ts and minimum_ts is greater than fault_delay
		Raise exception if delta between end_ts and maximum_ts is greater than fault_delay
		"""
		ts_dat = self.csv_dat[self.csv_dat['timestamp'].between(start_ts, end_ts)]
		minimum_ts = min(ts_dat['timestamp'])
		if abs(minimum_ts - start_ts) > fault_delay:
			raise Exception("start_ts is out of bounds: abs(minimum_ts - start_ts) > fault_delay")
		maximum_ts = max(ts_dat['timestamp'])
		if abs(maximum_ts - end_ts) > fault_delay:
			raise Exception("end_ts is out of bounds: abs(maximum_ts - end_ts) > fault_delay")
		return ts_dat

	def __str__(self) -> str:
		res = "----------------------------------------------------" + '\n'
		res += "PandaDatasetIterator('" + self.csv_path + "')" + '\n'
		res += "----------------------------------------------------" + '\n'
		res += "self.fps:		\t" + str(self.fps) + '\n'
		res += "self.frame_count:\t" + str(self.frame_count) + '\n'
		res += "self.start_time_csv:\t" + \
			str(datetime.fromtimestamp(self.start_time_csv)) + '\n'
		res += "self.end_time_csv:\t" + \
			str(datetime.fromtimestamp(self.end_time_csv)) + '\n'
		res += "self.duration:	\t" + \
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

		self.id = folder_path.split("/")[-1]
		#self.start_time = int(self.id)
		self.csv_path = os.path.join(folder_path, self.id + ".csv")
		self.mp4_path = os.path.join(folder_path, self.id + ".mp4")
		self.depth_mp4_path = os.path.join(folder_path, "depth_" + self.id + ".mp4")

		# CSV stores time in ms
		self.csv_dat = pd.read_csv(self.csv_path)
		self.start_time = self.csv_dat["Timestamp"].iloc[0]
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

		self.csv_fps = len(self.csv_dat) / self.expected_duration

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

	def get_item_by_timestamp(self, timestamp, fault_delay=1000):
		"""
		Return frame closest to given timestamp
		Raise exception if delta between timestamp and frame is greaterthan fault_delay
		"""
		closest_frames = self.get_item_between_timestamp(timestamp-fault_delay, timestamp+fault_delay, fault_delay=float('inf'))
		closest_frames = closest_frames.reset_index(drop=True)
		closest_frame = closest_frames.iloc[(closest_frames['Timestamp']-timestamp).abs().argsort()[0]]
		closest_ts = closest_frame['Timestamp']
		if abs(timestamp - closest_ts) > fault_delay:
			raise Exception("No such timestamp, fault delay exceeded: abs(timestamp - closest_ts)=" + str(abs(timestamp - closest_ts)))
		
		closest_ts_index = self.csv_dat.index[self.csv_dat['Timestamp'] == closest_ts].tolist()[0]
		return self.__getitem__(closest_ts_index)
		# return closest_frame

		
	def get_item_between_timestamp(self, start_ts, end_ts, fault_delay=500):
		"""
		Return frame between two given timestamps
		Raise exception if delta between start_ts and minimum_ts is greater than fault_delay
		Raise exception if delta between end_ts and maximum_ts is greater than fault_delay
		"""
		ts_dat = self.csv_dat[self.csv_dat['Timestamp'].between(start_ts, end_ts)]
		if len(ts_dat)==0:
			raise Exception("No such timestamp")
		minimum_ts = min(ts_dat['Timestamp']) #/ 1000.0
		if abs(minimum_ts - start_ts) > fault_delay:
			raise Exception("start_ts is out of bounds: abs(minimum_ts - start_ts)=" + str(abs(minimum_ts - start_ts)))
		maximum_ts = max(ts_dat['Timestamp']) #/ 1000.0
		if abs(maximum_ts - end_ts) > fault_delay:
			raise Exception("end_ts is out of bounds: abs(minimum_ts - start_ts)=" + str(abs(maximum_ts - end_ts)))
		return ts_dat

	def __iter__(self):
		self.line_no = 0
		return self

	def __next__(self):
		data = self.__getitem__(self.line_no)
		self.line_no += 1
		return data

	def __str__(self) -> str:
		res = "----------------------------------------------------" + '\n'
		res += "AndroidDatasetIterator('" + self.folder_path + "')" + '\n'
		res += "----------------------------------------------------" + '\n'
		res += "self.fps:		\t" + str(self.fps) + '\n'
		res += "self.frame_count:\t" + str(self.frame_count) + '\n'
		res += "self.start_time_csv:\t" + \
			str(datetime.fromtimestamp(self.start_time_csv/1000)) + '\n'
		res += "self.end_time_csv:\t" + \
			str(datetime.fromtimestamp(self.end_time_csv/1000)) + '\n'
		res += "self.expected_duration:\t" + \
			str(timedelta(seconds=self.expected_duration)) + '\n'
		res += "self.expected_fps:\t" + str(self.expected_fps) + '\n'
		res += "self.csv_fps:		\t" + str(self.csv_fps) + '\n'
		res += "----------------------------------------------------"
		return res

	def __repr__(self) -> str:
		return str(self)

	def __del__(self):
		pass


class MergedDatasetIterator:

	"""
		MergedDatasetIterator
		Iterates through dataset, given a AndroidDatasetIterator and a PandaDatasetIterator
	"""

	def __init__(self, 
		phone_iter: AndroidDatasetIterator, 
		panda_iter: PandaDatasetIterator, 
		settings_doc="calibration/pocoX3/calib.yaml", 
		compute_trajectory=False, 
		invalidate_cache=False,
		start_index=None,
		stop_index=None,
		step_indices=None
	) -> None:
		assert type(phone_iter)==AndroidDatasetIterator
		assert type(panda_iter)==PandaDatasetIterator
		# assert (start_index==None and stop_index==None and step_indices==None) or (start_index!=None and stop_index!=None and step_indices!=None), "All start, end and skip indices must be provided or none"
		# TODO: Generate ID for this pair of phone_iter and panda_iter

		self.compute_trajectory = compute_trajectory
		self.phone_iter = phone_iter
		self.panda_iter = panda_iter
		self.group = [
			(self.phone_iter.start_time_csv /1000.0, self.phone_iter.end_time_csv /1000.0),
			(self.panda_iter.start_time_csv, self.panda_iter.end_time_csv)
		]
		self.start_time, self.end_time = helper.intersection_of_group(self.group)
		
		self.duration = self.end_time - self.start_time
		self.IOU = helper.IOU_of_group(self.group)
		
		self.phone_dat = self.phone_iter.get_item_between_timestamp(self.start_time*1000.0, self.end_time*1000.0)
		#self.phone_dat = self.phone_iter.get_item_between_timestamp(self.start_time, self.end_time)
		self.phone_frame_count = len(self.phone_dat)
		self.phone_fps = self.phone_frame_count / self.duration
		
		self.panda_dat = self.panda_iter.get_item_between_timestamp(self.start_time, self.end_time)
		self.panda_frame_count = len(self.panda_dat)
		self.panda_fps = self.panda_frame_count / self.duration

		self.frame_count_original = max(self.phone_frame_count, self.panda_frame_count)
		
		if start_index==None:
			start_index = 0
		if stop_index==None:
			stop_index = self.frame_count_original
		if step_indices==None:
			step_indices = 1
		assert type(start_index)==int
		assert type(stop_index)==int
		assert type(step_indices)==int
		assert start_index>=0
		assert stop_index<=self.frame_count_original
		self.start_index = start_index
		self.stop_index = stop_index
		self.step_indices = step_indices

		self.frame_count = self.frame_count_original
		self.fps = max(self.phone_fps, self.panda_fps)

		self.settings_doc = settings_doc
		with open(self.settings_doc, 'r') as stream:
			try:
				self.cam_settings = yaml.load(stream, Loader=yaml.FullLoader)
			except yaml.YAMLError as exc:
				print(exc)
		k1 = self.cam_settings['Camera.k1']
		k2 = self.cam_settings['Camera.k2']
		p1 = self.cam_settings['Camera.p1']
		p2 = self.cam_settings['Camera.p2']
		k3 = 0
		if 'Camera.k3' in self.cam_settings:
			k3 = self.cam_settings['Camera.k3']
		self.DistCoef = np.array([k1, k2, p1, p2, k3])
		self.camera_matrix = np.array([
			[self.cam_settings['Camera.fx']	, 0.0							, self.cam_settings['Camera.cx']],
			[0.0							, self.cam_settings['Camera.fy'], self.cam_settings['Camera.cy']],
			[0.0							, 0.0							, 1.0]
		])

		self.folder_path = os.path.dirname(self.phone_iter.csv_path)
		cached_trajectory_folder = os.path.join(self.folder_path, TRAJECTORY_CACHE_DIR)
		os.makedirs(cached_trajectory_folder, exist_ok=True)
		self.cached_trajectory_path = os.path.join(cached_trajectory_folder, os.path.basename(self.phone_iter.csv_path) + ".pkl")
		if self.compute_trajectory:
			if not os.path.exists(self.cached_trajectory_path) or invalidate_cache:
				self.compute_slam()
			else:
				print("Loading trajectory from cache: ", self.cached_trajectory_path)
				with open(self.cached_trajectory_path, 'rb') as handle:
					self.trajectory = pickle.load(handle)

				# self.trajectory = pd.read_csv(self.cached_trajectory_path)
		else: 
			self.trajectory = pd.DataFrame({
				'x':[], 'y':[], 'z': [], 'rot': []
			})
		
		if self.start_index!=0 or self.step_indices!=self.frame_count_original or self.step_indices!=1:
			# Recompute 
			self.start_time, self.end_time = (
				self.start_time + self.start_index / self.fps,
				self.start_time + self.stop_index / self.fps
			)
			self.duration = self.end_time - self.start_time

			self.phone_dat = self.phone_iter.get_item_between_timestamp(self.start_time*1000.0, self.end_time*1000.0)
			self.phone_dat = self.phone_dat[self.phone_dat.index % self.step_indices == 0]  
			self.phone_frame_count = len(self.phone_dat)
			self.phone_fps = self.phone_frame_count / self.duration

			self.panda_dat = self.panda_iter.get_item_between_timestamp(self.start_time, self.end_time)
			self.panda_dat = self.panda_dat[self.panda_dat.index % self.step_indices == 0]  
			self.panda_frame_count = len(self.panda_dat)
			self.panda_fps = self.panda_frame_count / self.duration

			self.frame_count = max(self.phone_frame_count, self.panda_frame_count)
			self.fps = max(self.phone_fps, self.panda_fps)
		

	def compute_slam(self, scale_factor=0.25, enable_plot=False, plot_3D_x=250, plot_3D_y=500,):
		sys.path.append(os.path.join(pathlib.Path(__file__).parent.resolve(), "extras/pyslam"))
		# from extras.pyslam.visual_odometry import VisualOdometry
		from visual_odometry import VisualOdometry
		from visual_imu_gps_odometry import Visual_IMU_GPS_Odometry
		from camera import PinholeCamera
		from feature_tracker_configs import FeatureTrackerConfigs
		from feature_tracker import feature_tracker_factory

		self.trajectory = {
			'x':[], 'y':[], 'z': [], 'rot': []
		}

		cam = PinholeCamera(
			self.cam_settings['Camera.width'] * scale_factor, 
			self.cam_settings['Camera.height'] * scale_factor,
			self.cam_settings['Camera.fx'] * scale_factor,
			self.cam_settings['Camera.fy'] * scale_factor,
			self.cam_settings['Camera.cx'] * scale_factor,
			self.cam_settings['Camera.cy'] * scale_factor,
			self.DistCoef,
			self.cam_settings['Camera.fps']
		)
		num_features=2000  # how many features do you want to detect and track?

		# select your tracker configuration (see the file feature_tracker_configs.py) 
		# LK_SHI_TOMASI, LK_FAST
		# SHI_TOMASI_ORB, FAST_ORB, ORB, BRISK, AKAZE, FAST_FREAK, SIFT, ROOT_SIFT, SURF, SUPERPOINT, FAST_TFEAT
		tracker_config = FeatureTrackerConfigs.LK_SHI_TOMASI
		tracker_config['num_features'] = num_features
		
		feature_tracker = feature_tracker_factory(**tracker_config)
		print(feature_tracker)
		# create visual odometry object 
		self.vo = Visual_IMU_GPS_Odometry(cam, None, feature_tracker)
		print("Computing Trajectory")
		plot_3D = np.zeros((plot_3D_x, plot_3D_y, 3))
		for img_id in tqdm(range(0, self.frame_count, 1)):
			data_frame = self.__getitem__(img_id)

			phone_data_frame, phone_img_frame = data_frame['phone_frame']
			panda_data_frame = data_frame['panda_frame']

			phone_img_frame_scaled = cv2.resize(phone_img_frame, (0,0), fx=scale_factor, fy=scale_factor)

			self.vo.track(phone_img_frame_scaled, img_id,
				accel_data=np.array([
					phone_data_frame['linear_acc_x'],
					phone_data_frame['linear_acc_y'],
					phone_data_frame['linear_acc_z']
				]).reshape((3,1)),
				gyro_data=np.array([
					phone_data_frame['RotationV X'],
					phone_data_frame['RotationV Y'],
					phone_data_frame['RotationV Z'],
					phone_data_frame['RotationV W'],
					phone_data_frame['RotationV Acc']
				]),
				gps_data=np.array([
					phone_data_frame['Longitude'],
					phone_data_frame['Latitude'],
					phone_data_frame['speed'],
					phone_data_frame['heading']
				]),
				timestamp=phone_data_frame['Timestamp'],
			)
			if img_id>2:
				x, y, z = self.vo.traj3d_est[-1]
				rot = np.array(self.vo.cur_R, copy=True)
			else:
				# x, y, z = [0.0], [0.0], [0.0]
				x, y, z = 0.0, 0.0, 0.0
				rot = np.eye(3,3)

			if type(x)!=float:
				x = float(x[0])
			if type(y)!=float:
				y = float(y[0])
			if type(z)!=float:
				z = float(z[0])

			self.trajectory['x'] += [x]
			self.trajectory['y'] += [y]
			self.trajectory['z'] += [z]
			self.trajectory['rot'] += [rot]

			if enable_plot:
				p3x = int(x / 10 + plot_3D_x//2)
				p3y = int(z / 10 + plot_3D_y//2)
				if p3x in range(0, plot_3D_x) and p3y in range(0, plot_3D_y):
					plot_3D = cv2.circle(plot_3D, (p3y, p3x), 2, (0,255,0), 1)

			if enable_plot:
				cv2.imshow('plot_3D', plot_3D)
				cv2.imshow('Camera', self.vo.draw_img)
				key = cv2.waitKey(1)
				if key == ord('q'):
					break

		self.trajectory = pd.DataFrame(self.trajectory)
		# self.trajectory.to_csv(self.cached_trajectory_p]ath, index=False)
		with open(self.cached_trajectory_path, 'wb') as handle:
			pickle.dump(self.trajectory, handle, protocol=pickle.HIGHEST_PROTOCOL)


	def __len__(self) -> int:
		return self.frame_count

	def __getitem__(self, key):
		if type(key)==int:
			key_original = key
			if key > len(self):
				raise IndexError("Out of bounds; key=", key)
			
			# key = key * self.step_indices
			
			frame_ts = self.start_time + key / self.fps # frame timestamp in seconds
			panda_frame = self.panda_iter.get_item_by_timestamp(frame_ts)
			phone_frame = self.phone_iter.get_item_by_timestamp(frame_ts*1000.0)
			#phone_frame = self.phone_iter[key]
			
			return {
				'panda_frame': panda_frame,
				'phone_frame': phone_frame,
			}
		elif type(key)==slice:
			return MergedDatasetIterator(
				phone_iter=self.phone_iter,
				panda_iter=self.panda_iter,
				settings_doc=self.settings_doc,
				compute_trajectory=self.compute_trajectory,
				invalidate_cache=False,
				start_index=key.start,
				stop_index=key.stop,
				step_indices=key.step
			)
		else:
			raise IndexError("Unknown key type; key=" + str(key) + ", type(key)=" + str(type(key)))

	def __iter__(self):
		self.line_no = 0
		return self

	def __next__(self):
		if self.line_no > self.__len__():
			raise StopIteration
		data = self.__getitem__(self.line_no)
		self.line_no += 1
		return data

	def __str__(self) -> str:
		res = "----------------------------------------------------" + '\n'
		res += "MergedDatasetIterator" + '\n'
		res += "----------------------------------------------------" + '\n'
		res += "self.start_time:\t" + \
			 str(datetime.fromtimestamp(self.start_time)) + '\n'
		res += "self.end_time:\t\t" + \
			 str(datetime.fromtimestamp(self.end_time)) + '\n'
		res += "self.duration:\t\t" + \
			 str(timedelta(seconds=self.duration)) + '\n'
		res += "self.IOU:\t\t" + \
			 str(round(self.IOU*100, 2)) + " %" + '\n'
		res += "self.frame_count:\t" + str(self.frame_count) + '\n'
		res += "self.fps:\t\t" + str(self.fps) + '\n'
		res += "----------------------------------------------------"
		return res

	def get_item_by_timestamp(self, timestamp, fault_delay=1):
		"""
		Return frame closest to given timestamp
		Raise exception if delta between timestamp and frame is greaterthan fault_delay
		"""
		closest_frames = self.get_item_between_timestamp(timestamp-fault_delay, timestamp+fault_delay, fault_delay=float('inf'))
		closest_frames['panda_frame'] = closest_frames['panda_frame'].reset_index(drop=True)
		closest_frames['phone_frame'] = closest_frames['phone_frame'].reset_index(drop=True)
		closest_frame = {
			'panda_frame': closest_frames['panda_frame'].iloc[(closest_frames['panda_frame']['timestamp']-timestamp).abs().argsort()[0]],
			'phone_frame': closest_frames['phone_frame'].iloc[(closest_frames['phone_frame']['timestamp']-timestamp).abs().argsort()[0]]
		}
		panda_closest_ts = closest_frame['panda_frame']['timestamp']
		phone_closest_ts = closest_frame['phone_frame']['timestamp']

		if abs(panda_closest_ts - phone_closest_ts) > fault_delay:
			raise Exception("Phone Panda delta too large" + str(abs(panda_closest_ts - phone_closest_ts)))
		return closest_frame

	def get_item_between_timestamp(self, start_ts, end_ts, fault_delay=0.5):
		"""
		Return frame between two given timestamps
		Raise exception if delta between start_ts and minimum_ts is greater than fault_delay
		Raise exception if delta between end_ts and maximum_ts is greater than fault_delay
		"""
		start_frame_ts = self.start_time + start_ts / self.fps # frame timestamp in seconds
		end_frame_ts = self.start_time + end_frame_ts / self.fps # frame timestamp in seconds
		panda_frame = self.panda_iter.get_item_between_timestamp(start_frame_ts, end_frame_ts)
		phone_frame = self.phone_iter.get_item_by_timestamp(start_frame_ts*1000.0, end_frame_ts*1000)
		return {
			'panda_frame': panda_frame,
			'phone_frame': phone_frame,
		}

	def __repr__(self) -> str:
		return str(self)

	def __del__(self):
		pass


class DBCInterpreter:

	"""
		DBCInterpreter
		Interprets DBC file, given a DBC file path
	"""

	def __init__(self, dbc_path="dbc/honda_city.dbc") -> None:
		print("dbc_path:", dbc_path)
		self.dbc_path = dbc_path
		self.db = cantools.database.load_file(self.dbc_path)
		self.id_to_datatype = {}

	def interpret_can_frame(self, data):
		# Given a CAN DataFrame from PandaCSVInterpreter, produce output in dict
		# For example, result = {'throttle':30, 'rpm': 1200, ....}
		result = {}
		for key in data.keys():
			if key!='timestamp':
				try:
					if type(data[key])==str:
						tuple_dat = eval(data[key])
						hex_frame = binascii.unhexlify(eval(tuple_dat[1]))
						new_dict = self.db.decode_message(int(key), hex_frame)
						for k in new_dict:
							result[k] = new_dict[k]
				except KeyError:
					pass
				except ValueError:
					pass
				
		return result

	def __getitem__(self, key):
		return self.id_to_datatype[key]

	def __len__(self) -> int:
		return len(self.id_to_datatype)

	def __str__(self) -> str:
		res = "----------------------------------------------------" + '\n'
		res += "DBCInterpreter('" + self.dbc_path + "')" + '\n'
		res += "----------------------------------------------------" + '\n'
		for k in self.id_to_datatype: # Print out mapping from ID to data
			res += str(k) + ":\t" + str(self.id_to_datatype[k]) + '\n'
		res += "----------------------------------------------------"
		return res

	def __repr__(self) -> str:
		return str(self)

	def __del__(self):
		pass


if __name__ == '__main__':
	depth_dataset = BengaluruOccupancyDatasetIterator()
	scale = 0.3
	plot2D = True
	plot3D = True
	 

	def rotate_W(vis):
		inc_rot = Rotation.from_euler("xyz", (10, 0.0, 0.0),degrees=True).as_matrix()
		depth_dataset.transformation[:3,:3] = depth_dataset.transformation[:3,:3] @ inc_rot
		print("ROT:", Rotation.from_matrix(depth_dataset.transformation[:3,:3]).as_rotvec(degrees=True))

	def rotate_S(vis):
		inc_rot = Rotation.from_euler("xyz", (-10, 0.0, 0.0),degrees=True).as_matrix()
		depth_dataset.transformation[:3,:3] = depth_dataset.transformation[:3,:3] @ inc_rot
		print("ROT:", Rotation.from_matrix(depth_dataset.transformation[:3,:3]).as_rotvec(degrees=True))
	
	def rotate_D(vis):
		inc_rot = Rotation.from_euler("xyz", (0.0, 10, 0.0),degrees=True).as_matrix()
		depth_dataset.transformation[:3,:3] = depth_dataset.transformation[:3,:3] @ inc_rot
		print("ROT:", Rotation.from_matrix(depth_dataset.transformation[:3,:3]).as_rotvec(degrees=True))

	def rotate_A(vis):
		inc_rot = Rotation.from_euler("xyz", (0.0, -10, 0.0),degrees=True).as_matrix()
		depth_dataset.transformation[:3,:3] = depth_dataset.transformation[:3,:3] @ inc_rot
		print("ROT:", Rotation.from_matrix(depth_dataset.transformation[:3,:3]).as_rotvec(degrees=True))

	def rotate_E(vis):
		inc_rot = Rotation.from_euler("xyz", (0.0, 0.0, -10.0),degrees=True).as_matrix()
		depth_dataset.transformation[:3,:3] = depth_dataset.transformation[:3,:3] @ inc_rot
		print("ROT:", Rotation.from_matrix(depth_dataset.transformation[:3,:3]).as_rotvec(degrees=True))

	def rotate_R(vis):
		inc_rot = Rotation.from_euler("xyz", (0.0, 0.0, 10.0),degrees=True).as_matrix()
		depth_dataset.transformation[:3,:3] = depth_dataset.transformation[:3,:3] @ inc_rot
		print("ROT:", Rotation.from_matrix(depth_dataset.transformation[:3,:3]).as_rotvec(degrees=True))	

	def exit_Q(vis):
		exit()
	
	if plot3D:
		vis = o3d.visualization.VisualizerWithKeyCallback() # pylint: disable=E1101
		vis.create_window(
			window_name='Point Cloud', top=0, visible=True
		)

		vis.register_key_callback(ord('A'), rotate_A)
		vis.register_key_callback(ord('S'), rotate_S)
		vis.register_key_callback(ord('D'), rotate_D)
		vis.register_key_callback(ord('W'), rotate_W)
		vis.register_key_callback(ord('E'), rotate_E)
		vis.register_key_callback(ord('R'), rotate_R)
		vis.register_key_callback(ord('Q'), exit_Q)
		vis.register_key_callback(ord('Q'), exit)

		pcd = o3d.geometry.PointCloud()
		vis.add_geometry(pcd)
	
	for frame in depth_dataset:
		disparity_frame = frame['disparity_frame']
		# disparity_frame_rgb = cv2.cvtColor(disparity_frame, cv2.COLOR_GRAY2BGR)
		disparity_frame_rgb = cv2.applyColorMap(
			disparity_frame,
			cv2.COLORMAP_PLASMA
		)


		rgb_frame = frame['rgb_frame']
		frame_vis = np.concatenate([
			rgb_frame,
			disparity_frame_rgb,
		], 1)
		frame_vis = cv2.resize(frame_vis, (0,0), fx=scale, fy=scale)

		if plot3D:
			vis.remove_geometry(pcd)
			points = frame['points']
			pcd = o3d.geometry.PointCloud()
			pcd.points = o3d.utility.Vector3dVector(points)

			pcd = frame['pcd']
			vis.add_geometry(pcd)
			vis.poll_events()
			vis.update_renderer()

		if plot2D:
			cv2.imshow('frame', frame_vis)
			# key = cv2.waitKey(0)
			key = cv2.waitKey(1)
			# key = cv2.waitKey(500)
			if key==ord('q'):
				break

	exit()
	dbc_interp = DBCInterpreter("dbc/honda_city.dbc")
	panda_path = os.path.join("dataset/panda_logs/PANDA_2022-07-21_11:54:32.114482.csv")
	panda_iter = PandaDatasetIterator(panda_path, dbc_interp=dbc_interp)
	phone_iter = AndroidDatasetIterator('dataset/android/1658384924059')
	merged_iter = MergedDatasetIterator(panda_iter=panda_iter, phone_iter=phone_iter, compute_trajectory=False)

	# print(panda_iter)
	# print(phone_iter)
	print(merged_iter)
	small = merged_iter[
		int(merged_iter.fps*30):
		int(merged_iter.fps*60)
	]
	small.compute_slam()
	# # small = merged_iter
	# print(small)

	# for data_frame in small:
	# 	phone_data_frame, phone_img_frame = data_frame['phone_frame']
	# 	panda_data_frame = data_frame['panda_frame']
	# 	frame = cv2.resize(phone_img_frame, (0,0), fx=0.25, fy=0.25)
	# 	cv2.imshow('frame', frame)
	# 	key = cv2.waitKey(1000)
	# 	if key==ord('q'):
	# 		break
	# for frame in merged_iter:
	# 	print(frame)

	
