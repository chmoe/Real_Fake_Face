# -*- coding: UTF-8 -*-
# @Project: Progress 
# @File: Split 
# @Author: Henry Ng 
# @Date: 2021/12/06 14:03

import cv2
import numpy as np
from config import Config
from skimage import io


class Split(object):

    def __init__(self, file, ori_path):
        self.source_file = self.open_image(ori_path)
        # self.fake_or_real = fake_or_real
        self.input_is_str = isinstance(file, str)
        self.data = self.open_image(file) if self.input_is_str else file
        self.image_height = self.data.shape[0]  # 图片高度 341
        self.image_width = self.data.shape[1]  # 图片宽度 266
        self.img_ndarray = np.full((self.image_height, self.image_width), '0000000')  # 建立一个图片大小的二维数组，用0初始化

        self.last_color = "#"  # 存储上一个颜色（在循环过程中为当前颜色）
        self.color_set = set()  # 创建一个集合用于不重复存储当前颜色的位置

        # 当前颜色范围
        self.most_left = self.image_width
        self.most_right = 0
        self.most_top = self.image_height
        self.most_bottom = 0

    @staticmethod
    def open_image(path):
        """
        打开图片
        Return:
            高(row), 宽(col), 颜色([rgb])
        """
        bgr = cv2.imread(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # BGR转换为RGB
        return rgb

    @staticmethod
    def rgb2hex(rgb):
        hex_color = '#'
        for i in rgb:
            num = int(i)
            # 将R、G、B分别转化为16进制拼接转换并大写  hex() 函数用于将10进制整数转换成16进制，以字符串形式表示
            hex_color += str(hex(num))[-2:].replace('x', '0').upper()
        return hex_color

    def traversal(self, i: int, path_head: str, is_val: bool):
        """
        调用函数
        :param i: 第i张图片
        :param path_head: 路径头部，具体到real/fake
        :param is_val: 是否为验证集（用于控制文件形式存储还是文件夹形式存储）
        :return: NULL
        """
        counter = 0
        for row in range(self.image_height):
            for col in range(self.image_width):
                if self.img_ndarray[row, col] != '0000000':  # 当之前有过标注时
                    pass
                else:  # 当没有标注时
                    self.last_color = self.rgb2hex(self.data[row, col])  # 保存当前的hex颜色值
                    self.img_ndarray[row, col] = self.last_color  # 在ndarray中标记颜色的值

                    self.get_connected(row, col)  # 遍历找到当前点的连通区域

                    points = self.move_rectangle()  # 上下左右

                    # print("point: \n", point)
                    if points:
                        if is_val:
                            path = Config.path_exist(path_head + str(i)) + "/{}.jpg".format(counter)
                        else:
                            path = Config.path_exist(path_head) + "{}_{}.jpg".format(i, counter)
                        self.save_image(path, points)
                        counter += 1

    def save_image(self, path, point):
        """
        按照要求处理图片，然后进行保存
        :param path: 图片保存路径
        :param point: 上下左右的限定范围
        :return:
        """
        data_copy = np.copy(self.source_file)

        for row in range(self.image_height):
            for col in range(self.image_width):
                if [row, col] in point:
                    continue
                else:
                    data_copy[row][col] = (0, 0, 0)
        if self.input_is_str:
            cv2.imwrite(path, data_copy[:, :, (2, 1, 0)])
        else:
            io.imsave(path, data_copy)

    def move_rectangle(self):
        """
        寻找符合当前颜色的像素点
        :return: 返回 上下左右
        """

        current_color_list = []  # 存有当前颜色的数组

        for row in range(self.most_top, min(self.most_bottom + 1, self.image_height + 1)):
            for col in range(self.most_left, min(self.most_right + 1, self.image_width + 1)):
                if self.img_ndarray[row, col] == self.last_color:
                    current_color_list.append([row, col])

        return current_color_list

    def get_connected(self, _row, _col):
        """
        在图片中从当前点寻找连通域，并且在ndarray中标注颜色
        :param _row: 坐标x
        :param _col: 坐标y
        :return:
        """
        self.color_set = set()  # 创建一个集合用于不重复存储当前颜色的位置
        self.color_set.add((_row, _col))  # 将当前的内容添加到set中

        self.most_left = self.image_width
        self.most_right = 0
        self.most_top = self.image_height
        self.most_bottom = 0

        while 0 != len(self.color_set):
            # print("循环中, len(set): {}".format(len(self.color_set)))
            [row, col] = self.color_set.pop()  # 从里面随机找一个，并且删除
            # 依次遍历上下左右
            if row - 1 >= 0:
                self.set_color(row - 1, col)
            if row + 1 <= self.image_height - 1:
                self.set_color(row + 1, col)
            if col - 1 >= 0:
                self.set_color(row, col - 1)
            if col + 1 <= self.image_width - 1:
                self.set_color(row, col + 1)

    def set_color(self, row, col):
        if self.rgb2hex(self.data[row, col]) == self.last_color:  # 属于当前颜色
            # print("data_col: {}, last_color:{}".format(self.rgb2hex(source_file[row, col]), self.last_color))
            if self.img_ndarray[row, col] != self.last_color:  # 并且没有标记
                # print("没有标记")
                self.img_ndarray[row, col] = self.last_color
                self.color_set.add((row, col))
                if row > self.most_bottom:
                    self.most_bottom = row
                if row < self.most_top:
                    self.most_top = row
                if col < self.most_left:
                    self.most_left = col
                if col > self.most_right:
                    self.most_right = col
