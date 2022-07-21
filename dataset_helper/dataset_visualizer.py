import time


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

	def __init__(self) -> None:
		# Init everything
		pass


	def playback(self):
		try:
			# start recording @ T+3seconds
			while True:
				pass

		except KeyboardInterrupt:
			# stop recording
			print("Stopped rec at: ", time.time())


	def __str__(self) -> str:
		return "MergedDatasetVisualizer"

	def __repr__(self) -> str:
		return "MergedDatasetVisualizer"
