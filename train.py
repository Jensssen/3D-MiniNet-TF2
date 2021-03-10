# -*- coding: utf-8 -*-
"""
 Script to train 3D-MiniNet with Tensorflow 2.x
File:
    train.py

Authors: soe
Date:
    26.02.21

"""

import glob
import math
import os

import numpy as np
import tensorflow as tf
# from augmentation import augmentation
from tensorflow.python.framework.ops import disable_eager_execution

from config.config import config
from datasets.semanticKitti import DataGenerator
from model.mininet3d import MiniNet3D
from train_utils import _train_on_batch, _validate_on_batch

# os.environ["CUDA_VISIBLE_DEVICES"]="-1"


physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass


# Takes a sequence of channels and returns the corresponding indices in the rangeimage
def seq_to_idx(seq):
    idx = []
    if "x" in seq:
        idx.append(0)
    if "y" in seq:
        idx.append(1)
    if "z" in seq:
        idx.append(2)
    if "r" in seq:
        idx.append(3)
    if "d" in seq:
        idx.append(4)

    return np.array(idx, dtype=np.intp)


# Loads a queue of random examples, and returns a batch iterator for each input
# and output
def read_example(filename, batch_size):
    # Open tfrecord
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([filename], num_epochs=None)
    _, serialized_example = reader.read(filename_queue)

    # Create random queue
    min_queue_examples = 500
    batch = tf.train.shuffle_batch([serialized_example], batch_size=batch_size,
                                   capacity=min_queue_examples + 100 * batch_size, min_after_dequeue=min_queue_examples,
                                   num_threads=2)

    # Read a batch
    parsed_example = tf.parse_example(batch, features={'neighbors': tf.FixedLenFeature([], tf.string),
                                                       'points': tf.FixedLenFeature([], tf.string),
                                                       'label': tf.FixedLenFeature([], tf.string)})

    # Decode point cloud
    idx = seq_to_idx(CONFIG.CHANNELS)

    points_raw = tf.decode_raw(parsed_example['points'], tf.float32)
    points = tf.reshape(points_raw, [batch_size, CONFIG.IMAGE_HEIGHT * CONFIG.IMAGE_WIDTH, 1, 5])
    points = tf.gather(points, seq_to_idx(CONFIG.CHANNELS), axis=3)

    neighbors_raw = tf.decode_raw(parsed_example['neighbors'], tf.float32)
    neighbors = tf.reshape(neighbors_raw, [batch_size, CONFIG.IMAGE_HEIGHT * CONFIG.IMAGE_WIDTH, CONFIG.N_LEN, 5])
    neighbors = tf.gather(neighbors, seq_to_idx(CONFIG.CHANNELS), axis=3)

    # Decode label
    label_raw = tf.decode_raw(parsed_example['label'], tf.float32)
    label = tf.reshape(label_raw, [batch_size, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, CONFIG.N_CLASSES + 2])

    return points, neighbors, label


############################ TRAINING MANAGER ############################

# Displays configuration
def print_config():
    print("\n----------- RIU-NET CONFIGURATION -----------")
    print("input channels     : {}".format(CONFIG.CHANNELS.upper()))
    print("input dims         : {}x{}x{}".format(CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, CONFIG.IMAGE_DEPTH))
    print("pointnet embeddings: {}".format("yes" if CONFIG.POINTNET == True else "no"))
    print("focal loss         : {}".format("yes" if CONFIG.FOCAL_LOSS == True else "no"))
    print(
        "# of parameters    : {}".format(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))
    print("---------------------------------------------\n")


# Compute the average example processing time
def time_to_speed(batch_time, batch_size):
    return round(float(batch_size) / batch_time, 2)


# Pretty obvious
def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# Returns last saved checkpoint index
def get_last_checkpoint(output_model):
    files = glob.glob(output_model + "-*.index")
    checkpoints = [-1]
    for checkpoint in files:
        checkpoint = checkpoint.replace(output_model + "-", "")
        checkpoint = checkpoint.replace(".index", "")
        checkpoints.append(int(checkpoint))

    return max(checkpoints)


# Computes current learning rate given the decay settings
def get_learning_rate(iteration, start_rate, decay_interval, decay_value):
    rate = start_rate * (decay_value ** math.floor(iteration / decay_interval))
    return rate


def image_preprocessing(x):
    print("lol")
    return x


def mask_preprocessing(x):
    print("ddd")
    return x


def train():
    for dataset in ["train", "val"]:
        if dataset == "val":
            validation_file_list = open("./data/semantic_val.txt", "r")
            validation_file_list = [line.split(',') for line in validation_file_list.readlines()]

        else:
            train_file_list = open("./data/semantic_train.txt", "r")
            train_file_list = [line.replace("\n", "") for line in train_file_list.readlines()]

    # Datasets
    partition = {'train': train_file_list, 'validation': validation_file_list}

    # Generators
    training_generator = DataGenerator(list_file_names=partition['train'],
                                       dim=(64, 512, 1, 5),
                                       batch_size=config['batch_size'],
                                       n_classes=6,
                                       shuffle=True)

    validation_generator = DataGenerator(list_file_names=partition['validation'],
                                         dim=(64, 512, 1, 5),
                                         batch_size=config['batch_size'],
                                         n_classes=6,
                                         shuffle=True)

    model = MiniNet3D(num_classes=4, input_dim=(2048, 16, 11)).model

    optimizer = tf.keras.optimizers.Adam(learning_rate=config["learning_rate"])

    train_on_batch = tf.function(_train_on_batch)
    validate_on_batch = tf.function(_validate_on_batch)

    # start training
    for epoch in range(config['epochs']):
        print(f'Epoch_number: {epoch}')
        for idx in range(training_generator.__len__()):
            input_final, x, y = training_generator.__getitem__(idx)

            softmax, loss = train_on_batch(x=[input_final, x], y=y, model=model, optimizer=optimizer)

            print(loss)
    print("lol")


if __name__ == "__main__":
    train()
