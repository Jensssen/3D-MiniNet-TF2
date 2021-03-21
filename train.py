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

import cv2
import numpy as np
import tensorflow as tf

from augmentation import augmentation
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
            x, y = augmentation(x, y)
            softmax, loss = train_on_batch(x=[input_final, x], y=y, model=model, optimizer=optimizer)
            tb_train_loss.update_state(loss.numpy())

            if batch_idx % 200 == 0:
                logger.info(
                    f"[Training] Epoch: {epoch}, "
                    f"Iteration: {batch_idx * config.get('batch_size') + epoch * training_generator.__len__()}, "
                    f"loss: {tb_train_loss.result().numpy()}, lr: {optimizer.learning_rate(optimizer.iterations).numpy()}")

        logger.info(
            f"Mean train epoch {epoch} loss: {tb_train_loss.result().numpy()}"
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
                    f"loss: {tb_val_loss.result().numpy()}, lr: {optimizer.learning_rate(optimizer.iterations).numpy()}")

        logger.info(
            f"Mean validation epoch {epoch} loss: {tb_val_loss.result().numpy()}"
        )
        with val_summary_writer.as_default():
            tf.summary.scalar("Loss", tb_val_loss.result(), step=epoch)

        if tb_val_loss.result().numpy() < best_validation_loss:
            best_validation_loss = tb_val_loss.result().numpy()
            tf.saved_model.save(model, "./saved_model")
            model.save_weights('./saved_model/weights.h5')
            if not os.path.exists("./validation_results"):
                os.makedirs("./validation_results")
            for idx in range(0, 50):
                input_final, x, y = validation_generator.__getitem__(idx)
                softmax, loss = validate_on_batch(x=[input_final, x], y=y, model=model)
                for batch_id, image in enumerate(softmax):
                    gt = y[batch_id, :, :, 0] * 255
                    input = x[batch_id, :, :, 3] * 255
                    argmax = np.argmax(image.numpy(), axis=-1) * 85
                    image = np.concatenate((input, gt, argmax), axis=0)
                    cv2.imwrite(f"./validation_results/{idx}_{batch_id}.png", image)


if __name__ == "__main__":
    train()
