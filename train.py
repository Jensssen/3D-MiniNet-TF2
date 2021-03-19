# -*- coding: utf-8 -*-
"""
 Script to train 3D-MiniNet with Tensorflow 2.x
File:
    train.py

Authors: soe
Date:
    26.02.21

"""

import argparse
import glob
import os
import time

import numpy as np
import tensorflow as tf

from config.config import config
from datasets.semanticKitti import DataGenerator
from model.mininet3d import MiniNet3D
from train_utils import train_on_batch, validate_on_batch

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging

logger = logging.getLogger("tensorflow")
logger.setLevel(logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument("--use_pretrained_weights", default=False, required=False,
                    help="Set to True if you want to load pretrained weights from ./saved_model/weights.h5")
args = parser.parse_args()


class MeanIoU(object):
    """Mean intersection over union (mIoU) metric.
    Intersection over union (IoU) is a common evaluation metric for semantic
    segmentation. The predictions are first accumulated in a confusion matrix
    and the IoU is computed from it as follows:
        IoU = true_positive / (true_positive + false_positive + false_negative).
    The mean IoU is the mean of IoU between all classes.
    Keyword arguments:
        num_classes (int): number of classes in the classification problem.
    """

    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes

    def mean_iou(self, y_true, y_pred):
        """The metric function to be passed to the model.
        Args:
            y_true (tensor): True labels.
            y_pred (tensor): Predictions of the same shape as y_true.
        Returns:
            The mean intersection over union as a tensor.
        """
        # Wraps _mean_iou function and uses it as a TensorFlow op.
        # Takes numpy arrays as its arguments and returns numpy arrays as
        # its outputs.
        return tf.py_function(self._mean_iou, [y_true, y_pred], tf.float32)

    def _mean_iou(self, y_true, y_pred):
        """Computes the mean intesection over union using numpy.
        Args:
            y_true (tensor): True labels.
            y_pred (tensor): Predictions of the same shape as y_true.
        Returns:
            The mean intersection over union (np.float32).
        """
        # Compute the confusion matrix to get the number of true positives,
        # false positives, and false negatives
        # Convert predictions and target from categorical to integer format
        target = np.argmax(y_true, axis=-1).ravel()
        predicted = np.argmax(y_pred, axis=-1).ravel()

        # Trick for bincounting 2 arrays together
        x = predicted + self.num_classes * target
        bincount_2d = np.bincount(
            x.astype(np.int32), minlength=self.num_classes ** 2
        )
        assert bincount_2d.size == self.num_classes ** 2
        conf = bincount_2d.reshape(
            (self.num_classes, self.num_classes)
        )

        # Compute the IoU and mean IoU from the confusion matrix
        true_positive = np.diag(conf)
        false_positive = np.sum(conf, 0) - true_positive
        false_negative = np.sum(conf, 1) - true_positive

        # Just in case we get a division by 0, ignore/hide the error and
        # set the value to 1 since we predicted 0 pixels for that class and
        # and the batch has 0 pixels for that same class
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = true_positive / (true_positive + false_positive + false_negative)
        iou[np.isnan(iou)] = 1

        return iou


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
    print(f"input channels     : {config['channels'].upper()}")
    print(f"input dims         : {config.get('img_height')}x{config.get('img_width')}x{len(config.get('channels'))}")
    print(f"focal loss         : {'yes' if config.get('focal_loss') else 'no'}")
    print("---------------------------------------------\n")


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


def image_preprocessing(x):
    return x


def mask_preprocessing(x):
    print("ddd")
    return x


def train():
    validation_file_list = open("./data/semantic_val.txt", "r")
    validation_file_list = [line.replace("\n", "") for line in validation_file_list.readlines()]

    train_file_list = open("./data/semantic_train.txt", "r")
    train_file_list = [line.replace("\n", "") for line in train_file_list.readlines()]

    # Datasets
    partition = {'train': train_file_list, 'validation': validation_file_list}

    # Generators
    training_generator = DataGenerator(list_file_names=partition['train'],
                                       dim=(64, 512, 1, 5),
                                       batch_size=config['batch_size'],
                                       n_classes=6,
                                       shuffle=True,
                                       data_split="train")

    validation_generator = DataGenerator(list_file_names=partition['validation'],
                                         dim=(64, 512, 1, 5),
                                         batch_size=config['batch_size'],
                                         n_classes=6,
                                         shuffle=False,
                                         data_split="val")

    lr_method = tf.keras.optimizers.schedules.ExponentialDecay(
        config.get('learning_rate'), config.get('lr_decay_interval'), config.get('lr_decay_value'), staircase=True,
        name="exponential_decay_lr"
    )

    model = MiniNet3D(num_classes=config['n_classes'], input_dim=(2048, 16, 11)).model

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_method)

    if args.use_pretrained_weights == "True":
        model.load_weights('./saved_model/weights.h5')

    train_summary_writer = tf.summary.create_file_writer(os.path.join(config['tensorboard_log'], "train"))
    val_summary_writer = tf.summary.create_file_writer(os.path.join(config['tensorboard_log'], "val"))

    tb_train_loss = tf.keras.metrics.Mean(name='loss')
    tb_val_loss = tf.keras.metrics.Mean(name='loss')

    print_config()
    best_validation_loss = 1000
    # start training
    for epoch in range(config['epochs']):
        logger.info(f'Epoch_number: {epoch}')

        tb_train_loss.reset_states()
        tb_val_loss.reset_states()

        for batch_idx, (input_final, x, y) in enumerate(training_generator, 1):
            # points_data, label_data = augmentation(points_data, y)
            softmax, loss = train_on_batch(x=[input_final, x], y=y, model=model, optimizer=optimizer)
            tb_train_loss.update_state(loss.numpy())

            if batch_idx % 200 == 0:
                logger.info(
                    f"[Training] Epoch: {epoch}, "
                    f"Iteration: {batch_idx * config.get('batch_size') + epoch * training_generator.__len__()}, "
                    f"loss: {loss.numpy()}, lr: {optimizer.learning_rate(optimizer.iterations).numpy()}")

        logger.info(
            f"Mean epoch loss: {tb_train_loss.result().numpy()}"
        )
        with train_summary_writer.as_default():
            tf.summary.scalar("Loss", tb_train_loss.result(), step=epoch)
            tf.summary.scalar("Learning Rate", optimizer.learning_rate(optimizer.iterations).numpy(), step=epoch)

        for batch_idx, (input_final, x, y) in enumerate(validation_generator, 1):
            softmax, loss = validate_on_batch(x=[input_final, x], y=y, model=model)
            tb_val_loss.update_state(loss.numpy())

            if batch_idx % 200 == 0:
                logger.info(
                    f"[Validation] Epoch: {epoch}, "
                    f"Iteration: {batch_idx * config.get('batch_size') + epoch * training_generator.__len__()}, "
                    f"loss: {loss.numpy()}, lr: {optimizer.learning_rate(optimizer.iterations).numpy()}")

        logger.info(
            f"Mean epoch loss: {tb_val_loss.result().numpy()}"
        )
        with val_summary_writer.as_default():
            tf.summary.scalar("Loss", tb_val_loss.result(), step=epoch)

        # print(f"Validation loss: {epoch_val_loss / batch_idx}")
        # if epoch_val_loss / batch_idx < best_validation_loss:
        #     best_validation_loss = epoch_val_loss / batch_idx
        # tf.saved_model.save(model, "./saved_model")
        # model.save_weights('./saved_model/weights.h5')
        # for idx in range(0, 50):
        #     input_final, x, y = validation_generator.__getitem__(idx)
        # softmax, loss = train_on_batch(x=[input_final, x], y=y, model=model, optimizer=optimizer)
        # for batch_id, image in enumerate(softmax):
        #     gt = y[batch_id, :, :, 0] * 255
        # input = x[batch_id, :, :, 3] * 255
        # argmax = np.argmax(image.numpy(), axis=-1) * 85
        # image = np.concatenate((input, gt, argmax), axis=0)
        # cv2.imwrite(f"./saved_model/{idx}_{batch_id}.png", image)


if __name__ == "__main__":
    train()
