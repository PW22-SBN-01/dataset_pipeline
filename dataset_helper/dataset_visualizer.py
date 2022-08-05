import time
import os
import cv2

from .dataset_iterators import MergedDatasetIterator, AndroidDatasetIterator, PandaDatasetIterator

class AndroidDatasetVisualizer:

	"""
		AndroidDatasetVisualizer
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

	def __init__(self) -> None:
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
	panda_path = os.path.join("dataset/panda_logs/PANDA_2022-07-21_11:54:32.114482.csv")
	panda_iter = PandaDatasetIterator(panda_path)
	phone_iter = AndroidDatasetIterator('dataset/android/1658384924059')
	merged_iter = MergedDatasetIterator(panda_iter=panda_iter, phone_iter=phone_iter)

	merged_vis = MergedDatasetVisualizer(merged_iter, 0.25)
	merged_vis.playback()
	