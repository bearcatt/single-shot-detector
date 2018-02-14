import tensorflow as tf


EPSILON = 1e-8
SCALE_FACTORS = [10.0, 10.0, 5.0, 5.0]


# input pipeline settings.
# you need to tweak these numbers for your system,
# it can accelerate training
SHUFFLE_BUFFER_SIZE = 100
PREFETCH_BUFFER_SIZE = 1000
NUM_THREADS = 8
# read here about buffer sizes:
# https://stackoverflow.com/questions/46444018/meaning-of-buffer-size-in-dataset-map-dataset-prefetch-and-dataset-shuffle

RESIZE_METHOD = tf.image.ResizeMethod.NEAREST_NEIGHBOR
MATCHING_THRESHOLD = 0.5

# for tf.map_fn when creating training targets or doing nms
PARALLEL_ITERATIONS = 10
