import dataset_helper.dataset_iterators as dataset_iterators

d = dataset_iterators.AndroidDatasetIterator("dataset/android/1653972957447")
print(d)
print(d.get_item_by_timestamp(d.start_time + 12431))