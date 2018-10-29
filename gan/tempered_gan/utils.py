# -*- coding:utf-8-*-
import numpy as np
import os
import math
import scipy.misc


def get_out_size(source_size, stride):
    """
    :param source_size:
    :param stride:
    :return: the wanted size
    """
    return int(math.ceil(float(source_size) / float(stride)))


def resize_img(img, resize_h, resize_w):
    """
    :param img:source image
    :param resize_h:target image height
    :param resize_w:target image width
    :return:target image
    """
    return scipy.misc.imresize(img, [resize_h, resize_w])


def load_img(folder, input_size, batch_size):
    """
    :param folder: the path to dataset, ex:'/home/datasets/data'
    :param input_size: the return image size, if need the image will be resized, ex:[96, 96, 3]
    :param batch_size: the batch size
    :return: the dataset in input_size shape: dataset, the numbers of batch: chunk_size
    """
    image_files = os.listdir(folder)
    dataset = np.ndarray(
        shape=[len(image_files), ] + input_size,
        dtype=np.float32
    )
    num_images = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        im = scipy.misc.imread(image_file).astype(np.float)
        if im.shape != input_size:
            im = resize_img(im, input_size[0], input_size[1])
        image_data = (np.array(im).astype(float) / (255.0 / 2.0) - 1)
        dataset[num_images, :, :, :] = np.reshape(image_data, newshape=input_size)
        num_images = num_images + 1
        # print(num_images)
        if num_images == 51200:
            break
    dataset = dataset[0:num_images, :, :, :]
    chunk_size = int(math.ceil(float(num_images) / float(batch_size)))
    print('Chunk_size:', chunk_size)
    print('Full dataset tensor:', dataset.shape)
    return dataset, chunk_size
