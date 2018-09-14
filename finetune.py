"""Script to finetune AlexNet using Tensorflow.

With this script you can finetune AlexNet as provided in the alexnet.py
class on any given dataset. Specify the configuration settings at the
beginning according to your problem.
This script was written for TensorFlow >= version 1.2rc0 and comes with a blog
post, which you can find here:

https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

Author: Frederik Kratzert
contact: f.kratzert(at)gmail.com
"""

import os

import numpy as np
import tensorflow as tf

from alexnet import AlexNet
from datetime import datetime

"""
Configuration Part.
"""

# Path to the textfiles for the trainings and validation set
# train_file = '/path/to/train.txt'
# val_file = '/path/to/val.txt'

# Learning params
learning_rate = 0.01
num_epochs = 10
batch_size = 100

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
mnist = tf.keras.datasets.mnist

# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)

# Place data loading and preprocessing on the cpu
# with tf.device('/cpu:0'):
#     tr_data = ImageDataGenerator(train_file,
#                                  mode='training',
#                                  batch_size=batch_size,
#                                  num_classes=num_classes,
#                                  shuffle=True)
#     val_data = ImageDataGenerator(val_file,
#                                   mode='inference',
#                                   batch_size=batch_size,
#                                   num_classes=num_classes,
#                                   shuffle=False)
#
#     # create an reinitializable iterator given the dataset structure
#     iterator = Iterator.from_structure(tr_data.data.output_types,
#                                        tr_data.data.output_shapes)
#     next_batch = iterator.get_next()
#
# Ops for initializing the two different iterators
# training_init_op = iterator.make_initializer(tr_data.data)
# validation_init_op = iterator.make_initializer(val_data.data)
(x_train, yy_train), (x_test, y_test) = mnist.load_data()
x_train_size = len(x_train)
x_test_size = len(x_test)
x_train, x_test = x_train / 255.0, x_test / 255.0

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
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Add the accuracy to the summary
tf.summary.scalar('accuracy', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Get the number of training/validation steps per epoch
print(x_train_size / batch_size)
train_batches_per_epoch = int(np.floor(x_train_size / batch_size))
val_batches_per_epoch = int(np.floor(x_test_size / batch_size))

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

    flattened = tf.reshape(model.conv3, [100, -1])
    a = x_train[0:100]
    first_batch = np.zeros((100, 28, 28, 3))
    first_batch[:, :, :, 0] = a
    first_batch[:, :, :, 1] = a
    first_batch[:, :, :, 2] = a
    first_out = sess.run(flattened, feed_dict={x: first_batch})
    input_fn = lambda: tf.train.limit_epochs(tf.convert_to_tensor(first_out, dtype=tf.float32), num_epochs=1)
    index = kmeans.train(input_fn).predict_cluster_index(input_fn)
    centers = kmeans.cluster_centers()
    print("cluster index:", list(index))

    # Loop over number of epochs
    for epoch in range(num_epochs):
        current_x = 0

        print("{} Epoch number: {}".format(datetime.now(), epoch + 1))

        # Initialize iterator with the training dataset
        # sess.run(training_init_op)

        for step in range(train_batches_per_epoch):
            print("batch", step, train_batches_per_epoch)
            # get next batch of data
            a = x_train[current_x:current_x + batch_size]
            # b = y_train[current_x:current_x + batch_size]
            next_batch = np.zeros((batch_size, 28, 28, 3))
            next_batch[:, :, :, 0] = a
            next_batch[:, :, :, 1] = a
            next_batch[:, :, :, 2] = a
            current_x = current_x + batch_size

            next_out = sess.run(flattened, feed_dict={x: next_batch})
            input_fn = lambda: tf.train.limit_epochs(tf.convert_to_tensor(next_out, dtype=tf.float32), num_epochs=1)
            index = kmeans.train(input_fn).predict_cluster_index(input_fn)
            b = list(index)
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

        # Validate the model on the entire validation set
        print("{} Start validation".format(datetime.now()))
        # sess.run(validation_init_op)
        test_acc = 0.
        test_count = 0
        current_x = 0
        for _ in range(val_batches_per_epoch):

            a = x_test[current_x:current_x + batch_size]
            b = y_test[current_x:current_x + batch_size]
            next_batch = np.zeros((batch_size, 28, 28, 3))
            next_batch[:, :, :, 0] = a
            next_batch[:, :, :, 1] = a
            next_batch[:, :, :, 2] = a
            current_x = current_x + batch_size

            label_batch = np.zeros((b.shape[0], 10))
            for i in range(b.shape[0]):
                label_batch[i, b[i]] = 1
            # img_batch, label_batch = sess.run(next_batch)

            acc = sess.run(accuracy, feed_dict={x: next_batch,
                                                y: label_batch})
            test_acc += acc
            test_count += 1
        test_acc /= test_count
        print("{} Validation Accuracy = {:.4f}".format(datetime.now(),
                                                       test_acc))
        print("{} Saving checkpoint of model...".format(datetime.now()))

        # save checkpoint of the model
        checkpoint_name = os.path.join(checkpoint_path,
                                       'model_epoch' + str(epoch + 1) + '.ckpt')
        save_path = saver.save(sess, checkpoint_name)

        print("{} Model checkpoint saved at {}".format(datetime.now(),
                                                       checkpoint_name))
