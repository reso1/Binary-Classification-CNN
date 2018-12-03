"""
Utils and configurations file
"""
import os
import cv2
import numpy as np
import tensorflow as tf

from alexnet import AlexNet
from data_generator import ImageDataGenerator

import utils

FLAGS = tf.app.flags.FLAGS

# Learning params
learning_rate = 0.001
batch_size = 1

# Network params
num_classes = FLAGS.NUM_CLASSES
train_layers = ['fc8', 'fc7', 'fc6']

"""
Main Part of the finetuning Script.
"""

# Create parent path if it doesn't exist
if not os.path.isdir(FLAGS.CHECKPOINT_DIR):
    os.mkdir(FLAGS.CHECKPOINT_DIR)

# Place data loading and preprocessing on the cpu
with tf.device('/cpu:0'):
    tr_data = ImageDataGenerator(FLAGS.TRAIN_DATA_DESCRIPTION_FILE,
                                 mode='training',
                                 batch_size=batch_size,
                                 num_classes=num_classes,
                                 shuffle=True)
    val_data = ImageDataGenerator(FLAGS.VALID_DATA_DESCRIPTION_FILE,
                                  mode='inference',
                                  batch_size=batch_size,
                                  num_classes=num_classes,
                                  shuffle=False)
    # create an reinitializable iterator given the dataset structure

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3],name="x")
y = tf.placeholder(tf.float32, [batch_size, num_classes])
keep_prob = tf.placeholder(tf.float32)

# Initialize model
model = AlexNet(x, keep_prob, num_classes, train_layers)

# Link variable to model output
score = model.fc8
# List of trainable variables of the layers we want to train
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

# Op for calculating the loss
with tf.name_scope("cross_ent"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=score,
                                                                  labels=y))

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
writer = tf.summary.FileWriter(FLAGS.FILE_WRITER_DIR)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Get the number of training/validation steps per epoch
train_batches_per_epoch = int(np.floor(tr_data.data_size / batch_size))
val_batches_per_epoch = int(np.floor(val_data.data_size / batch_size))

tmppath = '../tmp/tmpimg/'
# Create temperory directory
if not os.path.isdir('../tmp'):
    os.mkdir('../tmp')
if not os.path.isdir(tmppath):
    os.mkdir(tmppath)

with tf.Session() as sess:
    # Initialize all variables
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "../model/checkpoints/model.ckpt")
    x1 = sess.graph.get_tensor_by_name("x:0")
    y1 = sess.graph.get_tensor_by_name("Placeholder_1:0")
    # Initialize image enhance variables
    imgpath = '../images/'
    path_list = os.listdir(imgpath)
    # For training image, use sort
    # path_list.sort(key=lambda x:int(x[:-4]))
    tmppath_list = os.listdir(tmppath)
    tmppath_list.sort(key=lambda x: int(x[:-4]))

    # Start
    # First read each image and divide into three parts
    if os.path.isfile('../result.txt'):
        os.remove('../result.txt')

    # data input:   
    for fn in path_list:
        if os.path.isfile('../tmp/tmp.txt'):
            os.remove('../tmp/tmp.txt')
        utils.remove_dir(tmppath) 

        units = utils.read_raw_image_single(imgpath+fn)
        for i in range(3):
            tmp_fn = tmppath + '%d.jpg'%i
            cv2.imwrite(tmp_fn, units[i])

            # Write to fileswe
            with open('../tmp/tmp.txt', 'a+') as f:
                f.write(tmp_fn + ' 0\n')

        # Make an Iterator of one image
        op_data = ImageDataGenerator('../tmp/tmp.txt',
                                     mode='training',
                                     batch_size=batch_size,
                                     num_classes=num_classes,
                                     shuffle=False)

        # get the output of network
        next_op = op_data.data.make_one_shot_iterator().get_next()
        op_softmax = sess.graph.get_operation_by_name("fc8/fc8").outputs[0]
        recog = tf.argmax(op_softmax, 1)

        # output the result of one image
        with open('../tmp/tmp.txt', 'r') as f:
            lines = f.readlines()
            li = [0,0,0]
            flag = 0
            for _ in range(3):
                outputdata, outputlabel = sess.run(next_op)
                result = sess.run(recog, feed_dict={x1: outputdata, y1: np.ones([1, 4096], dtype='float')})
                if result[0] == 1:
                    li[_] = 1
                    flag = 1 
        with open('../result.txt', 'a+') as fout:
            if flag == 0:
                fout.write('合格')
            elif flag == 1:
                fout.write('不合格\t')
                for i in range(3):
                    if li[i] == 1:
                        fout.write("%d " % (i+1))
            fout.write('\n')
        print('Successfully written to the result!')


