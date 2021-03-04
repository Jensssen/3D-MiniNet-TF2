import numpy as np
import tensorflow as tf


class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_file_names, batch_size=32, dim=(32, 32, 32), n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_file_names = list_file_names
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_file_names) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_file_names_temp = [self.list_file_names[k] for k in indexes]

        # Generate data
        final_input, x, y = self.__data_generation(list_file_names_temp)

        return final_input, x, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_file_names))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_file_names_temp):
        # Initialization
        x = np.empty((self.batch_size, *self.dim), dtype=np.float)
        y = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_classes), dtype=np.float)

        # Generate data
        for i, ID in enumerate(list_file_names_temp):
            # Store sample

            x[i,] = np.load('./data/train/points/' + ID + '.npy')

            # Store class
            y[i,] = np.load('./data/train/labels/' + ID + '.npy')

        x = np.float32(x)
        y = np.float32(y)

        x = x[:, :, :, 0, :]
        mask = y[:, :, :, 5]
        mask = np.expand_dims(mask, axis=-1)

        point_groups = tf.image.extract_patches(x, sizes=[1, 4, 4, 1], strides=[1, 4, 4, 1],
                                                rates=[1, 1, 1, 1], padding='SAME')
        point_groups = tf.reshape(point_groups, (-1, 16, 128, 16, 5))

        mask_groups = tf.image.extract_patches(mask, sizes=[1, 4, 4, 1], strides=[1, 4, 4, 1],
                                               rates=[1, 1, 1, 1], padding='SAME')
        mask_groups = tf.reshape(mask_groups, (-1, 16, 128, 16, 1))

        # Get the mean point (taking apart non-valid points
        point_groups_sumF = tf.reduce_sum(point_groups, axis=3)  # (b , h, w, feat)
        mask_groups_sumF = tf.reduce_sum(mask_groups, axis=3)
        point_groups_mean = point_groups_sumF / mask_groups_sumF
        # point_groups_mean: (?, 16, 128, 5):  (b , h, w, feat)
        point_groups_mean = tf.expand_dims(point_groups_mean, 3)
        point_groups_mean = tf.tile(point_groups_mean, [1, 1, 1, 16, 1])
        is_nan = tf.math.is_nan(point_groups_mean)
        point_groups_mean = tf.where(is_nan, x=tf.zeros_like(point_groups_mean), y=point_groups_mean)

        # substract mean point to points
        relative_points = point_groups - point_groups_mean

        mask_groups_tile = tf.tile(mask_groups, [1, 1, 1, 1, 5])
        relative_points = tf.where(tf.cast(mask_groups_tile, dtype=tf.bool), x=relative_points,
                                   y=tf.zeros_like(relative_points))

        # relative_points: (?, 16, 128, 16, 5):  (b , h, w,  neighb, feat)
        xyz_rel = relative_points[:, :, :, :, 0:3]
        relative_distance = tf.expand_dims(tf.norm(xyz_rel, ord='euclidean', axis=-1), axis=-1)
        input_final = tf.concat([point_groups, relative_points, relative_distance], axis=-1)
        # input_final: (?, 16, 128, 16, 11):  (b , h, w, neighb, feat)
        # neighbours: (?, 32768, 8, 5):  (b, points, neigbours, features)

        input_final = tf.reshape(input_final, (-1, 16 * 128, 16, 11))  # (?, 16*128, 16, 11)

        return input_final, x, y
