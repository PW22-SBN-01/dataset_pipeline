
def test_intersection():
	from dataset_helper.helper import intersection
	assert intersection(1,3,2,4)==(2,3)
	assert intersection(2,4,1,3)==(2,3)
	assert intersection(1,2,3,4)==(None,None)

def test_union():
	from dataset_helper.helper import union
	assert union(1,3,2,4)==(1,4)
	assert union(2,4,1,3)==(1,4)
	# assert union(1,2,3,4)==(None,None)
