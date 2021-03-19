# -*- coding: utf-8 -*-
"""
 
File:
    train_utils.py

Authors: soe
Date:
    28.02.21

"""

from typing import Tuple, List

import numpy as np
import tensorflow as tf

from config.config import config


@tf.function
def train_on_batch(x: List[np.ndarray], y: np.ndarray, model: tf.keras.Model, optimizer: tf.keras.optimizers) -> Tuple[
        tf.Tensor, tf.Tensor]:
    """

    Args:
        x: input images, 4-D Tensor of shape `[batch, height, width, num_channels]
        y: input labels, 4-D Tensor of shape `[batch, height, width, num_output_class]
        model: tensorflow model
        optimizer: use ONLY tf.keras.optimizers

    Returns:
        logits: 4-D Tensor of shape `[batch, height, width, num_output_class]
        loss:  scalar loss value

    """
    with tf.GradientTape() as tape:
        model_output = model(x, training=True)
        loss = u_net_loss(model_output[0], y)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return model_output[1], loss


@tf.function
def validate_on_batch(x: List[np.ndarray], y: np.ndarray, model: tf.keras.Model) -> Tuple[tf.Tensor, tf.Tensor]:
    """

    Args:
        x: input images, 4-D Tensor of shape `[batch, height, width, num_channels] from the validation data set
        y: tensor of true targets, 4-D Tensor of shape `[batch, height, width, num_output_class]`
        model: tensorflow model

    Returns:
        logits: 4-D Tensor of shape `[batch, height, width, num_output_class]
        loss:  scalar loss value

    """
    model_output = model(x, training=False)
    loss = u_net_loss(model_output[0], y)

    return model_output[1], loss


# Returns slices of a tensor
def slice_tensor(x, start, end=None):
    if end < 0:
        y = x[..., start:]
    else:
        if end is None:
            end = start
        y = x[..., start:end + 1]

    return y


def u_net_loss(pred, label):
    # Retrieve mask on last channel of the label
    mask = slice_tensor(label, config["n_classes"] + 1, -1)
    dist = slice_tensor(label, config["n_classes"], config["n_classes"])
    label = slice_tensor(label, 0, config["n_classes"] - 1)

    weight_norm = 2.0 * 3.0 ** 2.0

    weights_ce = 0.1 + 1.0 * tf.exp(- dist / weight_norm)
    weights_ce = weights_ce * mask

    # Compute the cross entropy
    if config["focal_loss"]:
        with tf.name_scope('focal_loss'):
            gamma = 2.
            pred_softmax = tf.nn.softmax(pred)
            cross_entropy = tf.multiply(label, -tf.math.log(pred_softmax))
            weights_fl = tf.multiply(label, tf.pow(tf.subtract(1., pred_softmax), gamma))
            loss = tf.reduce_sum(weights_ce * weights_fl * cross_entropy) / tf.reduce_sum(weights_ce)

    else:
        with tf.name_scope('loss'):

            cross_entropy = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=pred,
                                                                                 name="cross_entropy")
            loss = tf.reduce_sum(tf.reshape(cross_entropy,
                                            [config["batch_size"], config["img_height"], config["img_width"],
                                             1]) * weights_ce) / tf.reduce_sum(weights_ce)

    # Compute average precision
    with tf.name_scope('average_precision'):
        softmax_pred = tf.nn.softmax(pred)

        argmax_pred = tf.math.argmax(softmax_pred, axis=3)
        mask_bin = tf.squeeze(tf.math.greater(mask, 0))
        for c in range(1, config["n_classes"]):
            p = tf.math.equal(argmax_pred, c)
            l = tf.squeeze(tf.math.equal(slice_tensor(label, c, c), 1.0))

            intersection = tf.logical_and(p, l)
            union = tf.logical_or(p, l)

            iou = tf.reduce_sum(tf.cast(tf.logical_and(intersection, mask_bin), tf.float32)) / (
                    tf.reduce_sum(tf.cast(tf.logical_and(union, mask_bin), tf.float32)) + 0.00000001)
            print(c, iou)

    return loss
