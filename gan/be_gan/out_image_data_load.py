# -*- coding: utf-8 -*-

import os
import math
import numpy as np
from PIL import Image


def list_out_image_files(out_image_path, batch_size):
    files = []
    # os.walk get the tree like file system's path dir files recurrently
    for(dirpath, dirnames, filenames) in os.walk(out_image_path):
        if len(dirnames) == 0:
            file = [str(dirpath) + "/" + str(f) for f in filenames]
            files.extend(file)
    chunk_size = int(math.ceil(float(len(files)) / float(batch_size)))
    print('list out image over!')
    print('length: ', len(files))
    print('chunk_size: ', chunk_size)
    return files, chunk_size


def load_image_batch(files_list, batch_shape, batch_size, index):
    x_batch = np.zeros([batch_size] + batch_shape, dtype=np.float32)
    for j, img_p in enumerate(files_list[index:index + batch_size]):
        try:
            im = Image.open(img_p).convert('RGB').resize((batch_shape[0], batch_shape[1]), Image.BILINEAR)
            # im.save('/home/ziyangcheng/datasets/celebA/' + str(j) + '.png', 'png')
            x_batch[j] = (np.reshape(np.array(list(im.getdata())), newshape=batch_shape).astype(np.float32) / (
                        255.0 / 2.0) - 1)
        except ValueError:
            print('read image error just ignore it', img_p)
    # print('x_batch shape', x_batch.shape)
    return x_batch  # -1~1


def load_image_batch_zero(files_list, batch_shape, batch_size, index):
    x_batch = np.zeros([batch_size] + batch_shape, dtype=np.float32)
    for j, img_p in enumerate(files_list[index:index + batch_size]):
        try:
            im = Image.open(img_p).convert('RGB').resize((batch_shape[0], batch_shape[1]), Image.BILINEAR)
            # im.save('/home/ziyangcheng/datasets/celebA/' + str(j) + '.png', 'png')
            x_batch[j] = np.reshape(np.array(list(im.getdata())), newshape=batch_shape).astype(np.float32) / 255.0
        except ValueError:
            print('read image error just ignore it', img_p)
    # print('x_batch shape', x_batch.shape)
    return x_batch  # 0~1


if __name__ == '__main__':
    path = '/home/ziyangcheng/datasets/lsun/church_outdoor_train/'
    celebA_folder = '/home/ziyangcheng/datasets/celebA/img_align_celeba'
    f, c = list_out_image_files(celebA_folder, 3)
    load_image_batch(f, [96, 96, 3], 3, 0)
