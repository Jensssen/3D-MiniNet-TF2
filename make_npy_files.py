import os

import cv2
import numpy as np
import tensorflow as tf

# imports Settings manager
import data_loader
from config.config import data_config

# SQUEEZESEG DATASET
# Channels description:
# 0: X
# 1: Y
# 2: Z
# 3: REFLECTANCE
# 4: DEPTH
# 5: LABEL

semantic_base = "/home/IBEO.AS/soe/Downloads/lidar_2d/"


# Generates npy files for training
def make_npy_files():
    # Creates one folder for train and one for test (train and val)
    for dataset in ["train", "val"]:

        # Get path
        dataset_output = data_config["train_file_list_name"] if dataset == "train" else data_config[
            "val_file_list_name"]

        if not os.path.exists(dataset_output):
            os.makedirs(dataset_output)
            os.makedirs(os.path.join(dataset_output, "neighbors"))
            os.makedirs(os.path.join(dataset_output, "points"))
            os.makedirs(os.path.join(dataset_output, "labels"))

        file_list_name = open(dataset_output + ".txt", "w")

        if dataset == "val":
            file_list = open("./data/semantic_val.txt", "r")
        else:
            file_list = open("./data/semantic_train.txt", "r")

        # Going through each example
        line_num = 1
        for file in file_list:

            augmentation_list = ["normal"] if dataset is "val" else data_config["augmentation"]

            # Augmentation settings
            for aug_type in augmentation_list:

                print("[{}] >> Processing file \"{}\" ({}), with augmentation : {}".format(dataset, file[:-1], line_num,
                                                                                           aug_type))

                # Load labels
                data = np.load(os.path.join(semantic_base, file[:-1] + ".npy"))

                mask = data[:, :, 0] != 0

                # data = data_loader.interp_data(data[:,:,0:5], mask)

                p, n = data_loader.pointnetize(data[:, :, 0:5], n_size=data_config["n_size"])
                groundtruth = data_loader.apply_mask(data[:, :, 5], mask)

                # Compute weigthed mask
                contours = np.zeros((mask.shape[0], mask.shape[1]), dtype=bool)

                if np.amax(groundtruth) > data_config["n_classes"] - 1:
                    print("[WARNING] There are more classes than expected !")

                for c in range(1, int(np.amax(groundtruth)) + 1):
                    channel = (groundtruth == c).astype(np.float32)
                    gt_dilate = cv2.dilate(channel, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
                    gt_dilate = gt_dilate - channel
                    contours = np.logical_or(contours, gt_dilate == 1.0)

                contours = contours.astype(np.float32) * mask

                dist = cv2.distanceTransform((1 - contours).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

                # Create output label for training
                label = np.zeros((groundtruth.shape[0], groundtruth.shape[1], data_config["n_classes"] + 2))
                for y in range(groundtruth.shape[0]):
                    for x in range(groundtruth.shape[1]):
                        label[y, x, int(groundtruth[y, x])] = 1.0

                label[:, :, data_config["n_classes"]] = dist
                label[:, :, data_config["n_classes"] + 1] = mask

                np.save(os.path.join(os.path.join(dataset_output, "neighbors"), file[:-1] + ".npy"), n)
                np.save(os.path.join(os.path.join(dataset_output, "points"), file[:-1] + ".npy"), p)
                np.save(os.path.join(os.path.join(dataset_output, "labels"), file[:-1] + ".npy"), label)

                file_list_name.write(semantic_base + file[:-1] + ".npy\n")

            line_num += 1

        print("Process finished, stored {} entries in \"{}\"".format(line_num - 1, dataset_output))

        file_list_name.close()

    print("All files created.")


if __name__ == "__main__":
    make_npy_files()
