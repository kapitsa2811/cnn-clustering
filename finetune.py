import os

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
mnist = tf.keras.datasets.mnist
(x_train, yy_train), (x_test, y_test) = mnist.load_data()
x_train_size = len(x_train)
x_test_size = len(x_test)
x_train, x_test = x_train / 255.0, x_test / 255.0
#TODO- Create batches here

# Learning params
learning_rate = 0.01
num_epochs = 10
batch_size = 5000

# Network params
dropout_rate = 0.5
num_classes = 10
train_layers = ['conv3', 'conv4', 'conv5', 'fc8', 'fc7', 'fc6']

# How often we want to write the tf.summary data to disk
display_step = 20

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = "F:\\tensorboard"
checkpoint_path = "F:\\checkpoints"

"""
Main Part of the finetuning Script.
"""

# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, 28, 28, 3])
y = tf.placeholder(tf.float32, [batch_size, num_classes])
# keep_prob = tf.placeholder(tf.float32, name="keep_prob")

# Initialize model
model = AlexNet(x, num_classes, train_layers)

# Link variable to model output
score = model.pool5[:, 0, 0, :]

# List of trainable variables of the layers we want to train
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

# Op for calculating the loss
with tf.name_scope("cross_ent"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score,
                                                                  labels=y))  # TODO-Fix-y

# Train op
with tf.name_scope("train"):
    # Get gradients of all trainable variables
    gradients = tf.gradients(loss, var_list)
    gradients = list(zip(gradients, var_list))

    # Create optimizer and apply gradient descent to the trainable variables
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients)

# Add gradients to summary
for gradient, var in gradients:
    tf.summary.histogram(var.name + '/gradient', gradient)

# Add the variables we train to the summary
for var in var_list:
    tf.summary.histogram(var.name, var)

# Add the loss to summary
tf.summary.scalar('cross_entropy', loss)

# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    predicted_labels = tf.argmax(score, 1)

# Add the accuracy to the summary
# TODO- what should I do with this?
# tf.summary.scalar('accuracy', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Get the number of training/validation steps per epoch
train_batches_per_epoch = int(np.floor(x_train_size / batch_size))
val_batches_per_epoch = int(np.floor(x_test_size / batch_size))

# Clustering
kmeans = tf.contrib.factorization.KMeansClustering(num_clusters=num_classes, use_mini_batch=True)

# Start Tensorflow session
with tf.Session() as sess:
    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Add the model graph to TensorBoard
    writer.add_graph(sess.graph)

    # Load the pretrained weights into the non-trainable layer
    model.load_initial_weights(sess)

    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                      filewriter_path))

    flattened = tf.reshape(model.conv3, [-1, 2 * 2 * 10])

    # Loop over number of epochs
    for epoch in range(num_epochs):
        current_x = 0
        print("{} Epoch number: {}".format(datetime.now(), epoch + 1))

        # Initialize iterator with the training dataset
        # sess.run(training_init_op)
        # TODO- init mnist I guess?
        first_x = 0
        for step in range(train_batches_per_epoch):
            print("training kmeans", step, "out of", train_batches_per_epoch)

            # TODO- calculate the batches once!
            a = x_train[first_x:first_x + batch_size]
            first_x = first_x + batch_size
            first_batch = np.zeros((batch_size, 28, 28, 3))
            first_batch[:, :, :, 0] = a
            first_batch[:, :, :, 1] = a
            first_batch[:, :, :, 2] = a
            first_out = sess.run(flattened, feed_dict={x: first_batch})
            input_fn = lambda: tf.train.limit_epochs(tf.convert_to_tensor(first_out, dtype=tf.float32), num_epochs=1)
            kmeans.train(input_fn)

        print("kmeans trained")

        for step in range(train_batches_per_epoch):
            print("training cnn", step, "out of", train_batches_per_epoch)
            # TODO- calculate the batches once!
            a = x_train[current_x:current_x + batch_size]
            next_batch = np.zeros((batch_size, 28, 28, 3))
            next_batch[:, :, :, 0] = a
            next_batch[:, :, :, 1] = a
            next_batch[:, :, :, 2] = a
            current_x = current_x + batch_size

            next_out = sess.run(flattened, feed_dict={x: next_batch})
            input_fn = lambda: tf.train.limit_epochs(tf.convert_to_tensor(next_out, dtype=tf.float32), num_epochs=1)
            index = kmeans.predict_cluster_index(input_fn)

            b = np.zeros(batch_size, dtype=int)
            cI = 0
            for value in index:
                b[cI] = value
                cI = cI + 1

            label_batch = np.zeros((len(b), 10))
            for i in range(len(b)):
                label_batch[i, b[i]] = 1

            # img_batch, label_batch = sess.run(next_batch)

            # And run the training op
            sess.run(train_op, feed_dict={x: next_batch, y: label_batch})

            # Generate summary with the current batch of data and write to file
            if step % display_step == 0:
                s = sess.run(merged_summary, feed_dict={x: next_batch, y: label_batch})

                writer.add_summary(s, epoch * train_batches_per_epoch + step)

        # Evaluation
        print("{} Start evaluation".format(datetime.now()))

        y_label = np.zeros(y_test.shape, dtype=int)
        label_map = np.zeros(10)
        current_x = 0
        for _ in range(val_batches_per_epoch):
            a = x_test[current_x:current_x + batch_size]
            next_batch = np.zeros((batch_size, 28, 28, 3))
            next_batch[:, :, :, 0] = a
            next_batch[:, :, :, 1] = a
            next_batch[:, :, :, 2] = a

            label = sess.run(predicted_labels, feed_dict={x: next_batch})
            y_label[current_x:current_x + batch_size] = label
            current_x = current_x + batch_size

        result = list(zip(y_label, y_test))
        for i in range(10):
            listI = [x[1] for x in result if x[0] == i]
            labelI = most_common(listI) if (len(listI) > 0) else -1
            print("label:", i, labelI)
            label_map[i] = labelI

        n_correct = 0
        for i in range(len(y_label)):
            y_label[i] = label_map[y_label[i]]
            if y_label[i] == y_test[i]:
                n_correct = n_correct + 1

        print("accuracy:", n_correct / len(y_test))
        print(y_label)
        print("{} Saving checkpoint of model...".format(datetime.now()))

        # save checkpoint of the model
        checkpoint_name = os.path.join(checkpoint_path,
                                       'model_epoch' + str(epoch + 1) + '.ckpt')
        save_path = saver.save(sess, checkpoint_name)

        print("{} Model checkpoint saved at {}".format(datetime.now(),
                                                       checkpoint_name))
