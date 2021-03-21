import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from config.config import config
from train_utils import validate_on_batch

sys.path.append('./')
from datasets.semanticKitti import DataGenerator

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# Compute scores for a single image
def compute_iou_per_class(pred, label, mask, n_class):
    pred = np.argmax(pred[..., 0:n_class], axis=2) * mask
    label = label * mask

    ious = np.zeros(n_class)
    tps = np.zeros(n_class)
    fns = np.zeros(n_class)
    fps = np.zeros(n_class)

    for cls_id in range(n_class):
        tp = np.sum(pred[label == cls_id] == cls_id)
        fp = np.sum(label[pred == cls_id] != cls_id)
        fn = np.sum(pred[label == cls_id] != cls_id)

        ious[cls_id] = tp / (tp + fn + fp + 0.00000001)
        tps[cls_id] = tp
        fps[cls_id] = fp
        fns[cls_id] = fn

    return ious, tps, fps, fns


# Create a colored image with depth or label colors
def label_to_img(label_sm, depth, mask):
    img = np.zeros((label_sm.shape[0], label_sm.shape[1], 3))

    colors = np.array([[0, 0, 0], [78, 205, 196], [199, 244, 100], [255, 107, 107]])

    label = np.argmax(label_sm, axis=2)
    label = np.where(mask == 1, label, 0)

    for y in range(0, label.shape[0]):
        for x in range(0, label.shape[1]):
            if label[y, x] == 0:
                img[y, x, :] = [depth[y, x] * 255.0, depth[y, x] * 255.0, depth[y, x] * 255.0]
            else:
                img[y, x, :] = colors[label[y, x], :]

    return img / 255.0


# Export pointcloud with colored labels
def label_to_xyz(label_sm, data, mask, file):
    colors = np.array([[100, 100, 100], [78, 205, 196], [199, 244, 100], [255, 107, 107]])

    ys, xs = np.where(mask == 1)
    label = np.argmax(label_sm, axis=2)

    file = open(file, "w")
    for p in range(0, ys.shape[0]):
        x = xs[p]
        y = ys[p]
        l = label[y, x]
        file.write("{} {} {} {} {} {}\n".format(data[y, x, 0], data[y, x, 1], data[y, x, 2], colors[l, 0], colors[l, 1],
                                                colors[l, 2]))

    file.close()


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


# Run test routine
def test(saved_model, display=False):
    validation_file_list = open("./data/semantic_val.txt", "r")
    validation_file_list = [line.replace("\n", "") for line in validation_file_list.readlines()]

    # Create output dir if needed
    if not os.path.exists("./test_results"):
        os.makedirs("./test_results")

    print(f"Processing dataset file \"{11111}\" for saved_model {saved_model}:")

    model = tf.saved_model.load(saved_model)
    validation_generator = DataGenerator(list_file_names=validation_file_list,
                                         dim=(64, 512, 1, 5),
                                         batch_size=1,
                                         n_classes=6,
                                         shuffle=False,
                                         data_split="val")

    tps_sum = 0
    fns_sum = 0
    fps_sum = 0

    for batch_idx, (input_final, x, y) in enumerate(validation_generator, 1):
        softmax, loss = validate_on_batch(x=[input_final, x], y=y, model=model)

        softmax = softmax.numpy()[0]
        groundtruth = np.argmax(y[0, :, :, 0:config.get('n_classes')], axis=2)
        mask = y[0, :, :, config.get('n_classes') + 1] == 1

        if display and batch_idx % 20 == 0:
            plt.subplot(4, 1, 1)
            plt.imshow(x[0, :, :, 3] * mask)
            plt.title("Reflectance (for visualization)")
            plt.subplot(4, 1, 2)
            plt.imshow(softmax[:, :, 1] * mask)
            plt.title("Car prob")
            plt.subplot(4, 1, 3)
            plt.imshow(np.argmax(softmax, axis=2) * mask)
            plt.title("Prediction")
            plt.subplot(4, 1, 4)
            plt.imshow(groundtruth)
            plt.title("Label")
            #plt.show()
            plt.savefig(f'./test_results/{batch_idx}.png')

        iou, tps, fps, fns = compute_iou_per_class(softmax, groundtruth, mask, config.get('n_classes'))

        tps_sum += tps
        fns_sum += fns
        fps_sum += fps
        print(batch_idx)

    ious = tps_sum / (tps_sum + fns_sum + fps_sum + 0.000000001)
    pr = tps_sum / (tps_sum + fps_sum + 0.000000001)
    re = tps_sum / (tps_sum + fns_sum + 0.000000001)

    for i in range(1, config.get('n_classes')):
        print(
            "\tPixel-seg class {}: Precision: {:.3f}, Recall: {:.3f}, IoU: {:.3f}".format(i, pr[i], re[i], ious[i]))


if __name__ == "__main__":
    test(saved_model="./saved_model", display=False)
