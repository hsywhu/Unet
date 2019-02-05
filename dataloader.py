import os
from os.path import isdir, exists, abspath, join

import numpy as np
# import
from PIL import Image

class DataLoader():
    def __init__(self, root_dir='data', batch_size=2, test_percent=.1):
        self.batch_size = batch_size
        self.test_percent = test_percent

        self.root_dir = abspath(root_dir)
        self.data_dir = join(self.root_dir, 'scans')
        self.labels_dir = join(self.root_dir, 'labels')

        self.files = os.listdir(self.data_dir)

        self.data_files = [join(self.data_dir, f) for f in self.files]
        self.label_files = [join(self.labels_dir, f) for f in self.files]

    def __iter__(self):
        n_train = self.n_train()

        if self.mode == 'train':
            current = 0
            endId = n_train
        elif self.mode == 'test':
            current = n_train
            endId = len(self.data_files)

        while current < endId:
            # todo: load images and labels
            # hint: scale images between 0 and 1
            # hint: if training takes too long or memory overflow, reduce image size!
            img = Image.open(self.data_files[current])
            img = img.resize((388, 388))
            data_image = np.array(img, dtype=np.float32)

            # seamless tiling input image
            data_image_ = np.zeros((572, 572), dtype=np.float32)
            data_image_[int((data_image_.shape[0] - data_image.shape[0]) / 2): int(data_image_.shape[0] - (data_image_.shape[0] - data_image.shape[0]) / 2), int((data_image_.shape[1] - data_image.shape[1]) / 2): int(data_image_.shape[1] - (data_image_.shape[1] - data_image.shape[1]) / 2)] = data_image
            for i in range(int((data_image_.shape[0] - data_image.shape[0]) / 2)):
                for j in range(int((data_image_.shape[1] - data_image.shape[1]) / 2)):
                    data_image_[i, j] = data_image_[(data_image_.shape[0] - data_image.shape[0]) - i - 1, (data_image_.shape[1] - data_image.shape[1]) - j - 1]
                    data_image_[data_image_.shape[0] - 1 - i, j] = data_image_[int(data_image_.shape[0] - (data_image_.shape[0] - data_image.shape[0]) + i), (data_image_.shape[1] - data_image.shape[1]) - j - 1]
                    data_image_[i, data_image_.shape[1] - 1 - j] = data_image_[(data_image_.shape[0] - data_image.shape[0]) - i - 1, int(data_image_.shape[1] - (data_image_.shape[1] - data_image.shape[1]) + j)]
                    data_image_[data_image_.shape[0] - 1 - i, data_image_.shape[1] - 1 - j] = data_image_[data_image_.shape[0] - (data_image_.shape[0] - data_image.shape[0]) + i, data_image_.shape[1] - (data_image_.shape[1] - data_image.shape[1]) + j]
            for i in range(data_image.shape[0]):
                for j in range(int((data_image_.shape[0] - data_image.shape[0]) / 2)):
                    data_image_[int((data_image_.shape[0] - data_image.shape[0]) / 2 + i), j] = data_image_[int((data_image_.shape[0] - data_image.shape[0]) / 2 + i), data_image_.shape[0] - data_image.shape[0] - j - 1]
                    data_image_[int((data_image_.shape[0] - data_image.shape[0]) / 2 + i), data_image_.shape[0] - j - 1] = data_image_[int((data_image_.shape[0] - data_image.shape[0]) / 2 + i), data_image_.shape[0] - (data_image_.shape[0] - data_image.shape[0]) + j]
                    data_image_[j, int((data_image_.shape[0] - data_image.shape[0]) / 2 + i)] = data_image_[data_image_.shape[0] - data_image.shape[0] - j - 1, int((data_image_.shape[0] - data_image.shape[0]) / 2 + i)]
                    data_image_[data_image_.shape[0] - j - 1, int((data_image_.shape[0] - data_image.shape[0]) / 2 + i)] = data_image_[data_image_.shape[0] - (data_image_.shape[0] - data_image.shape[0]) + j, int((data_image_.shape[0] - data_image.shape[0]) / 2 + i)]

            # temp_image = Image.fromarray(data_image_)
            # temp_image.show()
            label = Image.open(self.label_files[current])
            label = label.resize((388, 388))
            label_image = np.array(label, dtype=np.float32)
            data_image /= 255
            current += 1
            yield (data_image_, label_image)

    def setMode(self, mode):
        self.mode = mode

    def n_train(self):
        data_length = len(self.data_files)
        return np.int_(data_length - np.floor(data_length * self.test_percent))

    # def __applyDataAugmentation(self, img, label):


    # def horizontalFlip(self, img, label):
