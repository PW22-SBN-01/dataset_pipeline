from dataset_helper import dataset_recorders

if __name__=="__main__":
	d = dataset_recorders.PandaDatasetRecorder()
	p = dataset_recorders.get_rec_in_process(d)
	try:
		p.run()
	except:
		print(p)
		print(repr(p))
		dataset_recorders.stop_process_safely(p)