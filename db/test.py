

from lset import *
# Build once
writer = DatasetWriter("mnist.ds")
def gen():
	for i in range(10):
		x = np.random.rand(32, 32).astype(np.float32)
		y = np.array([i % 10], dtype=np.int64)
		yield x, y
writer.add_many(gen())
writer.close()

# Stream during training (unbatched, sequential)
for x, y in DatasetIterable("mnist.ds"):
	# train step with single (x, y)
	print(x, y)



