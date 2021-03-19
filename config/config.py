# -*- coding: utf-8 -*-
"""
 
File:
    config.py

Authors: soe
Date:
    26.02.21

"""

config = {
    "train": "../boundingbox/avod/rangeimage/pnl_train.txt",
    "validation": "../boundingbox/avod/rangeimage/pnl_val.txt",
    "train_file_list_name": "./data/train",
    "val_file_list_name": "./data/val",
    "augmentation": ["original"],
    "n_size": [3, 3],
    "channels": "xyzdr",
    "n_classes": 4,
    "img_width": 512,
    "img_height": 64,
    "epochs": 20,
    "unet_depth": 5,
    "batch_size": 4,
    "learning_rate": 0.0003,
    "lr_decay_interval": 1000000,
    "lr_decay_value": 0.1,
    "focal_loss": True,
    "num_iterations": 1000000,
    "val_interval": 10,
    "path": "training_pnl2/",
    "tensorboard_log": "./tensorboard_log/",
    "model": "model.ckpt",
    "save_interval": 2000,
    "output_path": "validation/"
}
