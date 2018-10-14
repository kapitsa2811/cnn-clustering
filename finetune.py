import numpy as np
import tensorflow as tf

from alexnet import AlexNet
from datetime import datetime

import itertools
import operator


def most_common(l):
    # get an iterable of (item, iterable) pairs
    sl = sorted((xx, ii) for ii, xx in enumerate(l))
    # print 'SL:', SL
    groups = itertools.groupby(sl, key=operator.itemgetter(0))

    # auxiliary function to get "quality" for an item
    def _auxfun(g):
        item, iterable = g
        count = 0
        min_index = len(l)
        for _, where in iterable:
            count += 1
            min_index = min(min_index, where)
        # print 'item %r, count %r, minind %r' % (item, count, min_index)
        return count, -min_index

    # pick the highest-count/earliest item
    return max(groups, key=_auxfun)[0]


"""
Configuration Part.
"""

# Learning params
learning_rate = 0.01
num_epochs = 10
batch_size = 5000

mnist = tf.keras.datasets.mnist
(x_train, yy_train), (x_test, y_test) = mnist.load_data()
x_train_size = len(x_train)
x_test_size = len(x_test)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Get the number of training/validation steps per epoch
train_batches_per_epoch = int(np.floor(x_train_size / batch_size))
val_batches_per_epoch = int(np.floor(x_test_size / batch_size))

# Creating batches
train_batches = np.zeros((train_batches_per_epoch, batch_size, 28, 28, 3))
for step in range(train_batches_per_epoch):
    a = x_train[step * batch_size:(step + 1) * batch_size]
    first_batch = np.zeros((batch_size, 28, 28, 3))
    first_batch[:, :, :, 0] = a
    first_batch[:, :, :, 1] = a
    first_batch[:, :, :, 2] = a

    train_batches[step, :, :, :, :] = first_batch

test_batches = np.zeros((val_batches_per_epoch, batch_size, 28, 28, 3))
for step in range(val_batches_per_epoch):
    a = x_test[step * batch_size:(step + 1) * batch_size]
    first_batch = np.zeros((batch_size, 28, 28, 3))
    first_batch[:, :, :, 0] = a
    first_batch[:, :, :, 1] = a
    first_batch[:, :, :, 2] = a

    test_batches[step, :, :, :, :] = first_batch

# Network params
num_classes = 10
skip_layers = ['conv3', 'conv4', 'conv5', 'fc8', 'fc7', 'fc6']
train_layers = ['conv3', 'fc']

"""
Main Part of the finetuning Script.
"""

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, 28, 28, 3])
y = tf.placeholder(tf.float32, [batch_size, num_classes])

# Initialize model
model = AlexNet(x, num_classes, skip_layers)

# Link variable to model output
featureNode = model.fc6

# List of trainable variables of the layers we want to train
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

# Op for calculating the loss
with tf.name_scope("cross_ent"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=featureNode,
                                                                  labels=y))

# Train op
with tf.name_scope("train"):
    # Get gradients of all trainable variables
    gradients = tf.gradients(loss, var_list)
    gradients = list(zip(gradients, var_list))

    # Create optimizer and apply gradient descent to the trainable variables
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients)

# Clustering
kmeans = tf.contrib.factorization.KMeansClustering(num_clusters=num_classes, use_mini_batch=True)
previous_centers = None

# Start Tensorflow session
with tf.Session() as sess:
    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Load the pretrained weights into the non-trainable layer
    model.load_initial_weights(sess)

    print("{} Start training...".format(datetime.now()))

    # Loop over number of epochs
    for epoch in range(num_epochs):
        print("{} Epoch number: {}".format(datetime.now(), epoch + 1))

        # Initialize iterator with the training dataset
        # TODO- init mnist I guess?
        for step in range(train_batches_per_epoch):
            print("training cnn", step, "out of", train_batches_per_epoch)

            featureVector = sess.run(featureNode, feed_dict={x: train_batches[step]})
            featureTensor = lambda: tf.train.limit_epochs(tf.convert_to_tensor(featureVector, dtype=tf.float32), num_epochs=1)
            kmeans.train(featureTensor)
            predictResult = list(kmeans.transform(featureTensor))
            cluster_centers = kmeans.cluster_centers()
            if previous_centers is not None:
                print('delta:', cluster_centers - previous_centers)
            previous_centers = cluster_centers

            cluster_indexes = list(kmeans.predict_cluster_index(featureTensor))
            # cluster_indexes = list(kmeans.train(featureTensor).predict_cluster_index(featureTensor))
            # predictResult = list(kmeans.predict(featureTensor))
            # print("kmeans indexes:", predictResult)
            print("cluster indexes:", cluster_indexes)
            print("cluster predict result:", predictResult)
            print("kmeans score:", kmeans.score(featureTensor))

            label_batch = np.zeros((batch_size, 10))
            for i in range(batch_size):
                label_batch[i, cluster_indexes[i]] = 1

            # And run the training op
            sess.run(train_op, feed_dict={x: train_batches[step], y: label_batch})

        # Evaluation
        print("{} Start evaluation".format(datetime.now()))
        y_label = np.zeros(y_test.shape, dtype=int)
        label_map = np.zeros(10)
        current_x = 0
        for step in range(val_batches_per_epoch):
            featureVector = sess.run(featureNode, feed_dict={x: test_batches[step]})
            featureTensor = lambda: tf.train.limit_epochs(tf.convert_to_tensor(featureVector, dtype=tf.float32), num_epochs=1)
            cluster_indexes = list(kmeans.predict_cluster_index(featureTensor))
            print("test input:", featureVector)
            print("test indexes", cluster_indexes)
            y_label[current_x:current_x + batch_size] = cluster_indexes
            current_x = current_x + batch_size

        result = list(zip(y_label, y_test))
        for i in range(10):
            listI = [x[1] for x in result if x[0] == i]
            labelI = most_common(listI) if (len(listI) > 0) else -1
            label_map[i] = labelI

        print("label map:", label_map)

        n_correct = 0
        for i in range(len(y_label)):
            y_label[i] = label_map[y_label[i]]
            if y_label[i] == y_test[i]:
                n_correct = n_correct + 1

        print("accuracy:", n_correct / len(y_test))
