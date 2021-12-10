# -*- coding: UTF-8 -*-
# @Project: Progress 
# @File: Split 
# @Author: Henry Ng 
# @Date: 2021/12/06 14:03

import numpy as np
from config import Config
from skimage import io
from Debug import Debug


class Split(object):

    def __init__(self, segment_labels, ori_path):
        self.source_file = self.open_image(ori_path)
        self.data = segment_labels
        self.image_height = self.data.shape[0]  # 图片高度 341
        self.image_width = self.data.shape[1]  # 图片宽度 266
        self.current_color = -1
        self.color_dictionary = {}

    @staticmethod
    def open_image(path):
        rgb = io.imread(path)
        return rgb

    def add_color(self, position: (int, int)):
        if self.current_color in self.color_dictionary:
            self.color_dictionary[self.current_color].append(position)
        else:
            self.color_dictionary[self.current_color] = [position]

    def traversal(self, i: int, path_head: str, is_val: bool):
        """
        调用函数
        :param i: 第i张图片
        :param path_head: 路径头部，具体到real/fake
        :param is_val: 是否为验证集（用于控制文件形式存储还是文件夹形式存储）
        :return: NULL
        """
        for row in range(self.image_height):
            for col in range(self.image_width):
                self.current_color = self.data[row][col]
                self.add_color((row, col))
        for color in self.color_dictionary.keys():
            if is_val:
                path = Config.path_exist(path_head + str(i)) + "/{}.jpg".format(color)
            else:
                path = Config.path_exist(path_head) + "{}_{}.jpg".format(i, color)
            self.save_image(path, set(self.color_dictionary[color]))

    def save_image(self, path, color):
        """
        按照要求处理图片，然后进行保存
        :param path: 图片保存路径
        :param color: 当前颜色序号
        :return:
        """
        data_copy = np.copy(self.source_file)

        for row in range(self.image_height):
            for col in range(self.image_width):
                if (row, col) in color:
                    continue
                else:
                    data_copy[row][col] = (0, 0, 0)

        io.imsave(path, data_copy, check_contrast=False)

