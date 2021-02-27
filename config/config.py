# -*- coding: utf-8 -*-
"""
 
File:
    config.py

Authors: soe
Date:
    26.02.21

"""

data_config = {
    "train": "../boundingbox/avod/rangeimage/pnl_train.txt",
    "validation": "../boundingbox/avod/rangeimage/pnl_val.txt",
    "tfrecord_train": "data/pnl_train.tfrecord",
    "tfrecord_val": "data/pnl_val.tfrecord",
    "augmentation": ["original"],
    "n_size": [3, 3],
    "channels": "xyzdr",
    "pointnet": True
}

network_config = {
    "n_classes": 4,
    "img_width": 512,
    "img_height": 64
}

train_config = {
    "unet_depth": 5,
    "batch_size": 8,
    "learning_rate": 0.0003,
    "lr_decay_interval": 1000000,
    "lr_decay_value": 0.1,
    "focal_loss": True,
    "num_iterations": 1000000,
    "val_interval": 10,
    "path": "training_pnl2/",
    "logs": "logs/",
    "model": "model.ckpt",
    "save_interval": 2000,
    "output_path": "validation/"
}
