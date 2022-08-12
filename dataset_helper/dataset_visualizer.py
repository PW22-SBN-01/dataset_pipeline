import time
import os
import cv2
import numpy as np
import matplotlib.cm

from .dataset_iterators import MergedDatasetIterator, AndroidDatasetIterator, PandaDatasetIterator, DBCInterpreter

class AndroidDatasetVisualizer:

	"""
		AndroidDatasetVisualizer
	"""

	def __init__(self, phone_iter, gps_size=256, line_thickness=1, padding=5, scale_factor=1.0) -> None:
		self.phone_iter = phone_iter
		self.gps_size = gps_size
		self.scale_factor = scale_factor
		self.line_thickness = line_thickness
		self.padding = padding
		self.gps_map = np.zeros(shape=(self.gps_size + self.padding, self.gps_size + self.padding, 4), dtype=np.uint8)
		
		self.min_longitude = min(self.phone_iter.csv_dat['Longitude'])
		self.max_longitude = max(self.phone_iter.csv_dat['Longitude'])
		self.min_latitude = min(self.phone_iter.csv_dat['Latitude'])
		self.max_latitude = max(self.phone_iter.csv_dat['Latitude'])

		self.max_speed = max(self.phone_iter.csv_dat['speed'])
		self.min_speed = min(self.phone_iter.csv_dat['speed'])

		self.cmap = matplotlib.cm.get_cmap('viridis')
		
		#for index in range(0, len(self.phone_iter), self.phone_iter.fps):
		#for data_frame, img_frame in self.phone_iter:
		for index, data_frame in self.phone_iter.csv_dat.iterrows():
			#data_frame = self.phone_iter.csv_dat[index]
			longitude = data_frame['Longitude']
			latitude = data_frame['Latitude']
			x_pos = int((longitude - self.min_longitude) / (self.max_longitude - self.min_longitude) * self.gps_size + self.padding//2)
			y_pos = int((latitude - self.min_latitude) / (self.max_latitude - self.min_latitude) * self.gps_size + self.padding//2)
			self.gps_map[
				max(0, x_pos-self.line_thickness) : min(self.gps_size+self.padding, x_pos+self.line_thickness), 
				max(0, y_pos-self.line_thickness) : min(self.gps_size+self.padding, y_pos+self.line_thickness), 
			] = (125,125,125,125)

	def generate_frame_visuals(self, data_frame, img_frame):
		longitude = data_frame['Longitude']
		latitude = data_frame['Latitude']
		x_pos = int((longitude - self.min_longitude) / (self.max_longitude - self.min_longitude) * self.gps_size + self.padding//2)
		y_pos = int((latitude - self.min_latitude) / (self.max_latitude - self.min_latitude) * self.gps_size + self.padding//2)
		color = self.cmap((data_frame['speed'] - self.min_speed)/(self.max_speed-self.min_speed))
		color = list(map(lambda k: int(k*255), color))
		self.gps_map[
			max(0, x_pos-self.line_thickness) : min(self.gps_size+self.padding, x_pos+self.line_thickness), 
			max(0, y_pos-self.line_thickness) : min(self.gps_size+self.padding, y_pos+self.line_thickness), 
		] = (color[2],color[1],color[0],255)

		self.final_frame = cv2.cvtColor(img_frame.copy(), cv2.COLOR_RGB2RGBA)
		self.gps_frame = np.zeros_like(self.final_frame)
		gps_start = 20
		self.gps_frame[
			gps_start:gps_start+self.gps_size+self.padding,
			gps_start:gps_start+self.gps_size+self.padding
		] = self.gps_map

		self.final_frame = cv2.addWeighted(self.gps_frame,1.0,self.final_frame,1.0,0)
		if self.scale_factor!=1.0:
			self.final_frame = cv2.resize(self.final_frame, (0,0), fx=self.scale_factor, fy=self.scale_factor)

	def playback(self):
		try:
			# Start playing
			for data_frame, img_frame in self.phone_iter:

				self.generate_frame_visuals(data_frame, img_frame)
				
				cv2.imshow('final_frame', self.final_frame)
				key = cv2.waitKey(1)
				if key == ord('q'):
					break
				pass

		except KeyboardInterrupt:
			# Stop playing
			print("Stopped rec at: ", time.time())
   

	def __str__(self) -> str:
		return "AndroidDatasetVisualizer"

	def __repr__(self) -> str:
		return "AndroidDatasetVisualizer"


class DVRDatasetVisualizer:

	"""
		DVRDatasetVisualizer
	"""

	def __init__(self) -> None:
		# Start TCP Server
		# Default message to be 'standby'
		pass
	

	def playback(self):
		try:
			# Start playing
			while True:
				pass

		except KeyboardInterrupt:
			# Stop playing
			print("Stopped rec at: ", time.time())
   

	def __str__(self) -> str:
		return "DVRDatasetVisualizer"

	def __repr__(self) -> str:
		return "DVRDatasetVisualizer"



class PandaDatasetVisualizer:

	"""
		PandaDatasetVisualizer
	"""

	def __init__(self, panda_iter, scale_x=256, scale_y=50) -> None:
		assert type(panda_iter) == PandaDatasetIterator
		self.scale_x = scale_x
		self.scale_y = scale_y
		self.cmap = matplotlib.cm.get_cmap('viridis')
		self.panda_iter = panda_iter
		self.brake_bar = np.zeros((self.scale_x,self.scale_y,3), dtype=np.uint8)
		self.brake_pressed = np.zeros((self.scale_x,self.scale_y,3), dtype=np.uint8)
		self.throttle_bar = np.zeros((self.scale_x,self.scale_y,3), dtype=np.uint8)
		self.rpm_bar = np.zeros((self.scale_x,self.scale_y,3), dtype=np.uint8)
		self.transmission_speed = np.zeros((60,250,3), dtype=np.uint8)
		self.full_car = np.zeros((200,200,3), dtype=np.uint8)
		pass
	
	def generate_frame_visuals(self, data_frame):
		dat = self.panda_iter.dbc_interp.interpret_can_frame(data_frame)
		if 'CAR_GAS' in dat:
			self.throttle_bar = np.zeros((self.scale_x,self.scale_y,3), dtype=np.uint8)
			self.throttle_bar[
				self.scale_x-min(self.scale_x, int(dat['CAR_GAS']/100.0 * self.scale_x)): self.scale_x,
				0: self.scale_y
			] = (0,0,255)

		if 'XMISSION_SPEED' in dat:
			self.transmission_speed = np.zeros((60,250,3), dtype=np.uint8)
			cv2.putText(
				self.transmission_speed, str(round(dat['XMISSION_SPEED'] * 18/5, 1)) + ' Km/h', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),
				2, cv2.LINE_AA
			)

		if 'ENGINE_RPM' in dat:
			self.rpm_bar = np.zeros((self.scale_x,self.scale_y,3), dtype=np.uint8)
			self.rpm_bar[
				self.scale_x-min(self.scale_x, int(dat['ENGINE_RPM']/8000.0 * self.scale_x)): self.scale_x,
				0: self.scale_y
			] = (0,0,255)

		if 'BRAKE_SWITCH' in dat:
			self.brake_pressed = np.zeros((self.scale_x,self.scale_y,3), dtype=np.uint8)
			if dat['BRAKE_SWITCH']:
				self.brake_pressed[:,:] = (0,0,255)

		if 'WHEEL_SPEED_FL' in dat:
			self.full_car = np.zeros((200,200,3), dtype=np.uint8)
			dat['XMISSION_SPEED'] = (dat['WHEEL_SPEED_FL'] + dat['WHEEL_SPEED_FR'] + dat['WHEEL_SPEED_RL'] + dat['WHEEL_SPEED_RR']) /4
			deltas = [
				abs(dat['WHEEL_SPEED_FL'] - dat['XMISSION_SPEED']),
				abs(dat['WHEEL_SPEED_FR'] - dat['XMISSION_SPEED']),
				abs(dat['WHEEL_SPEED_RL'] - dat['XMISSION_SPEED']),
				abs(dat['WHEEL_SPEED_RR'] - dat['XMISSION_SPEED'])
			]
			# max_del = max(deltas)
			max_del = 0.2
			if max_del>0:
				color_fl = self.cmap(deltas[0]/max_del)
				color_fl = list(map(lambda k: int(k*255), color_fl))
				self.full_car[
					10:90,
					10:90,
				] = color_fl[:3]

				color_fr = self.cmap(deltas[1]/max_del)
				color_fr = list(map(lambda k: int(k*255), color_fr))
				self.full_car[
					110:190,
					10:90,
				] = color_fr[:3]

				color_rl = self.cmap(deltas[2]/max_del)
				color_rl = list(map(lambda k: int(k*255), color_rl))
				self.full_car[
					10:90,
					110:190,
				] = color_rl[:3]

				color_rr = self.cmap(deltas[3]/max_del)
				color_rr = list(map(lambda k: int(k*255), color_rr))
				self.full_car[
					110:190,
					110:190,
				] = color_rr[:3]

	def playback(self):
		try:
			# Start playing
			#for data_frame in self.panda_iter:
			for index in range(int(self.panda_iter.fps*120), len(self.panda_iter), 1):
				data_frame = self.panda_iter[index]
				
				self.generate_frame_visuals(data_frame)
					
				#self.throttle_bar = np.zeros((50,500,3))

				cv2.imshow('brake_bar', self.brake_bar)
				cv2.imshow('full_car', self.full_car)
				cv2.imshow('transmission_speed', self.transmission_speed)
				cv2.imshow('brake_pressed', self.brake_pressed)
				cv2.imshow('throttle_bar', self.throttle_bar)
				cv2.imshow('rpm_bar', self.rpm_bar)
				key = cv2.waitKey(1)
				if key == ord('q'):
					break

		except KeyboardInterrupt:
			# Stop playing
			print("Stopped rec at: ", time.time())
   

	def __str__(self) -> str:
		return "PandaDatasetVisualizer"

	def __repr__(self) -> str:
		return "PandaDatasetVisualizer"


class MergedDatasetVisualizer:

	"""
		MergedDatasetVisualizer
	"""

	def __init__(self, merged_iter: MergedDatasetIterator, scale_factor=1.0) -> None:
		# Init everything
		assert type(merged_iter) == MergedDatasetIterator
		self.merged_iter = merged_iter
		self.phone_vis = AndroidDatasetVisualizer(self.merged_iter.phone_iter, line_thickness=2)
		self.panda_vis = PandaDatasetVisualizer(self.merged_iter.panda_iter)
		self.scale_factor = scale_factor


	def playback(self):
		try:
			# start playback 
			# for data_frame in self.merged_iter:
			# for index in range(int(self.merged_iter.fps*120), len(self.merged_iter), int(self.merged_iter.fps*0.5)):
			for index in range(0, len(self.merged_iter)):
				data_frame = self.merged_iter[index]

				phone_data_frame, phone_img_frame = data_frame['phone_frame']
				panda_data_frame = data_frame['panda_frame']

				h,  w = phone_img_frame.shape[:2]
				newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.merged_iter.camera_matrix, self.merged_iter.DistCoef, (w,h), 1, (w,h))
				phone_img_frame = cv2.undistort(phone_img_frame, self.merged_iter.camera_matrix, self.merged_iter.DistCoef, None, newcameramtx)
				x, y, w, h = roi
				phone_img_frame = phone_img_frame[y:y+h, x:x+w]

				self.phone_vis.generate_frame_visuals(phone_data_frame, phone_img_frame)
				self.panda_vis.generate_frame_visuals(panda_data_frame)
				
				#self.final_frame = phone_img_frame.copy()
				self.final_frame = cv2.cvtColor(phone_img_frame.copy(), cv2.COLOR_RGB2RGBA)
				self.final_frame = cv2.addWeighted(self.phone_vis.gps_frame,1.0,self.final_frame,1.0,0)
				
				self.panda_frame = np.zeros_like(self.final_frame)
				start_point_x = self.phone_vis.padding
				start_point_y = self.phone_vis.gps_size + self.phone_vis.padding + 50
				self.panda_frame[
					start_point_x: start_point_x + self.panda_vis.rpm_bar.shape[0],
					start_point_y : start_point_y + self.panda_vis.rpm_bar.shape[1]
				] = cv2.cvtColor(self.panda_vis.rpm_bar, cv2.COLOR_RGB2RGBA)

				self.panda_frame[
					start_point_x : start_point_x + self.panda_vis.throttle_bar.shape[0],
					start_point_y + self.panda_vis.rpm_bar.shape[1]: start_point_y + self.panda_vis.rpm_bar.shape[1] + self.panda_vis.throttle_bar.shape[1]
				] = cv2.cvtColor(self.panda_vis.throttle_bar, cv2.COLOR_RGB2RGBA)

				self.panda_frame[
					start_point_x : start_point_x + self.panda_vis.brake_pressed.shape[0],
					start_point_y + self.panda_vis.rpm_bar.shape[1] + self.panda_vis.throttle_bar.shape[1]: start_point_y + self.panda_vis.rpm_bar.shape[1] + self.panda_vis.throttle_bar.shape[1] + self.panda_vis.brake_pressed.shape[1] 
				] = cv2.cvtColor(self.panda_vis.brake_pressed, cv2.COLOR_RGB2RGBA)
				# import pandas as pd
				# self.merged_iter.trajectory = pd.DataFrame(self.merged_iter.trajectory)
				point_cur = self.merged_iter.trajectory.iloc[index]
				# for row, point in self.merged_iter.trajectory[index:index+int(self.merged_iter.fps)].iterrows():
				for row, point in self.merged_iter.trajectory[index+5:index+500].iterrows():
					# rot = np.eye(4,4)
					# rot[0:3,0:3] = point_cur['rot']
					# p3d = np.array([
					# 	point['x'] - point_cur['x'],
					# 	point['y'] - point_cur['y'],
					# 	point['z'] - point_cur['z'],
					# 	1.0
					# ]).reshape((4,1))
					# p3d = rot @ p3d
					
					rot = point_cur['rot']
					p4d = np.ones((4,1))
					p3d = np.array([
						point['x'] - point_cur['x'],
						point['y'] - point_cur['y'],
						point['z'] - point_cur['z'],
					]).reshape((3,1))
					p4d[:3,:] = np.linalg.inv(rot) @ p3d
					

					homo_cam_mat = np.hstack((newcameramtx, np.zeros((3,1))))
					p2d = (self.scale_factor*homo_cam_mat) @ p4d
					# p2d = (self.scale_factor*self.merged_iter.camera_matrix) @ p3d
					if p2d[2][0]!=0.0:
						px, py = int(p2d[0][0] / p2d[2][0]), int(p2d[1][0] / p2d[2][0])
						self.final_frame = cv2.circle(self.final_frame, (px, py), 5, (0,255,0), -1)

				self.final_frame = cv2.addWeighted(self.panda_frame,1.0,self.final_frame,1.0,0)

				if self.scale_factor!=1.0:
					self.final_frame = cv2.resize(self.final_frame, (0,0), fx=self.scale_factor, fy=self.scale_factor)
				cv2.imshow('frame', self.final_frame)
				key = cv2.waitKey(1)
				if key == ord('q'):
					break
		except KeyboardInterrupt:
			# stop 
			print("Stopped at: ", time.time())


	def __str__(self) -> str:
		return "MergedDatasetVisualizer"

	def __repr__(self) -> str:
		return "MergedDatasetVisualizer"


if __name__=="__main__":
	dbc_interp = DBCInterpreter("dbc/honda_city.dbc")

	panda_path = os.path.join("dataset/panda_logs/PANDA_2022-07-21_11:54:32.114482.csv")
	panda_iter = PandaDatasetIterator(panda_path, dbc_interp=dbc_interp)
	phone_iter = AndroidDatasetIterator('dataset/android/1658384924059')
	merged_iter = MergedDatasetIterator(
		panda_iter=panda_iter, 
		phone_iter=phone_iter,
		compute_trajectory=True,
		invalidate_cache=False
	)

	# merged_vis = MergedDatasetVisualizer(merged_iter, 0.25)
	# merged_vis.playback()
	# phone_vis = AndroidDatasetVisualizer(phone_iter, gps_size=384, scale_factor=0.5, line_thickness=3)
	# phone_vis.playback()
	# panda_vis = PandaDatasetVisualizer(panda_iter=panda_iter)
	# panda_vis.playback()

	merged_iter = merged_iter[
		int(merged_iter.fps*30):
		int(merged_iter.fps*35)
	]
	# merged_iter.compute_slam()

	merged_vis = MergedDatasetVisualizer(merged_iter=merged_iter, scale_factor=0.5)
	merged_vis.playback()