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

	def playback(self):
		try:
			# Start playing
			for data_frame, img_frame in self.phone_iter:

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

				final_frame = cv2.cvtColor(img_frame.copy(), cv2.COLOR_RGB2RGBA)
				gps_frame = np.zeros_like(final_frame)
				gps_start = 20
				gps_frame[
					gps_start:gps_start+self.gps_size+self.padding,
					gps_start:gps_start+self.gps_size+self.padding
				] = self.gps_map

				final_frame = cv2.addWeighted(gps_frame,1.0,final_frame,1.0,0)
				
				if self.scale_factor!=1.0:
					final_frame = cv2.resize(final_frame, (0,0), fx=self.scale_factor, fy=self.scale_factor)
				cv2.imshow('final_frame', final_frame)
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

	def __init__(self, panda_iter) -> None:
		assert type(panda_iter) == PandaDatasetIterator
		self.panda_iter = panda_iter
		pass
	

	def playback(self):
		try:
			# Start playing
			for data_frame in self.panda_iter:
				# print(data_frame)
				# print(self.panda_iter.dbc_interp)
				print(self.panda_iter.dbc_interp.interpret_can_frame(data_frame))

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
		self.merged_iter = merged_iter
		self.scale_factor = scale_factor


	def playback(self):
		try:
			# start playback 
			for data_frame in self.merged_iter:
				android_frame = data_frame['phone_frame'][1]
				print(data_frame['phone_frame'][0])
				print(data_frame['panda_frame'])
				final_frame = android_frame
				if self.scale_factor!=1.0:
					final_frame = cv2.resize(final_frame, (0,0), fx=self.scale_factor, fy=self.scale_factor)
				cv2.imshow('frame', final_frame)
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
	merged_iter = MergedDatasetIterator(panda_iter=panda_iter, phone_iter=phone_iter)

	# merged_vis = MergedDatasetVisualizer(merged_iter, 0.25)
	# merged_vis.playback()
	# phone_vis = AndroidDatasetVisualizer(phone_iter, gps_size=384, scale_factor=0.5, line_thickness=3)
	# phone_vis.playback()
	panda_vis = PandaDatasetVisualizer(panda_iter=panda_iter)
	panda_vis.playback()