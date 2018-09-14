import os

import numpy as np
import tensorflow as tf

k = 5
n = 1000
variables = 2

points = np.random.uniform(0, 1, [n, variables])
input_fn = lambda: tf.train.limit_epochs(tf.convert_to_tensor(points, dtype=tf.float32), num_epochs=1)
kmeans = tf.contrib.factorization.KMeansClustering(num_clusters=k, use_mini_batch=False)
previous_centers = None
for _ in range(10):
    kmeans.train(input_fn)
    centers = kmeans.cluster_centers()
    if previous_centers is not None:
      print('delta:', centers - previous_centers)
    previous_centers = centers
    print('score:', kmeans.score(input_fn))