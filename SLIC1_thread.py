# -*- coding: UTF-8 -*-
# @Project: 5.Real_Fake_Face
# @File: SLIC1
# @Author: Henry Ng 
# @Date: 2021-11-30 10:29

import math
from skimage import io, color
import numpy as np
from tqdm import trange
import cv2
import os

class Cluster(object):
    cluster_index = 1

    def __init__(self, h, w, l=0, a=0, b=0):
        self.update(h, w, l, a, b)
        self.pixels = []  # 超像素对应的像素点
        self.no = self.cluster_index
        Cluster.cluster_index += 1

    def update(self, h, w, l, a, b):
        self.h = h
        self.w = w
        self.l = l
        self.a = a
        self.b = b

    def __str__(self):
        return "{},{}:{} {} {} ".format(self.h, self.w, self.l, self.a, self.b)

    def __repr__(self):
        return self.__str__()


class SLICProcessor(object):
    @staticmethod
    def open_image(path):
        """
        打开图片
        Return:
            高(row), 宽(col), 颜色([lab])
        """
        rgb = io.imread(path)
        lab_arr = color.rgb2lab(rgb)
        return lab_arr  # shape=(341, 266, 3)

    @staticmethod
    def point2int(arr):
        pass

    @staticmethod
    def save_lab_image(path, lab_arr):
        """
        将图片从lab转换回rgb，然后保存到指定的path
        :param path:
        :param lab_arr:
        :return:
        """
        rgb_arr = color.lab2rgb(lab_arr)
        io.imsave(path, rgb_arr)

    def make_cluster(self, h, w):
        h = int(h)
        w = int(w)
        # data是指原本的图片文件
        return Cluster(h, w,
                       self.data[h][w][0],  # l
                       self.data[h][w][1],  # a
                       self.data[h][w][2])  # b

    def __init__(self, filename, K, M):
        self.K = K  # 分割为K个S*S的大小
        self.M = M  # 需要从M地方调整使之成为正方形
        self.number = os.path.splitext(os.path.split(filename)[1])[0] # 图像编号
        self.type = filename.split('/')[-2] # 图像类型
        self.data = self.open_image(filename)
        self.image_height = self.data.shape[0]  # 图片高度 341
        self.image_width = self.data.shape[1]  # 图片宽度 266
        self.N = self.image_height * self.image_width  # 图片像素 90706
        self.S = int(math.sqrt(self.N / self.K))  # N个像素平均分为k个超像素，每个超像素边长（相邻种子距离，即步长）

        self.clusters = []  # 超像素的位置
        self.label = {}  # 每个像素属于哪个超像素
        # dis: 像素到超像素的距离
        self.dis = np.full((self.image_height, self.image_width), np.inf)

    # RGB格式颜色转换为16进制颜色格式
    @staticmethod
    def rgb2hex(rgb):
        print("转换")
        hexcolor = '#'
        for i in rgb:
            print("i:", i)
            num = int(i) if i % 1 == 0 else int(i * 255)
            print("num", num)
            # 将R、G、B分别转化为16进制拼接转换并大写  hex() 函数用于将10进制整数转换成16进制，以字符串形式表示
            hexcolor += str(hex(num))[-2:].replace('x', '0').upper()
        return hexcolor

    def init_clusters(self):
        """
        初始化超像素
        :return: self
        """
        h = self.S / 2
        w = self.S / 2
        while h < self.image_height:  # 不超越高度的范围
            while w < self.image_width:  # 不超越宽度的范围
                self.clusters.append(self.make_cluster(h, w))  # 保存每一个超像素
                w += self.S  # 向右移动s宽度
            w = self.S / 2  # 横向从头开始
            h += self.S  # 纵向向下移动一个s宽度

    def get_gradient(self, h, w):
        """
        获取梯度
        :param h:
        :param w:
        :return:
        """
        if w + 1 >= self.image_width:
            w = self.image_width - 2
        if h + 1 >= self.image_height:
            h = self.image_height - 2

        gradient = self.data[h + 1][w + 1][0] - self.data[h][w][0] + \
                   self.data[h + 1][w + 1][1] - self.data[h][w][1] + \
                   self.data[h + 1][w + 1][2] - self.data[h][w][2]
        return gradient

    def move_clusters(self):
        # 找到梯度最小的超像素
        for cluster in self.clusters:  # 遍历所有超像素
            cluster_gradient = self.get_gradient(cluster.h, cluster.w)
            # 梯度下降
            for dh in range(-1, 2):
                for dw in range(-1, 2):
                    _h = cluster.h + dh
                    _w = cluster.w + dw
                    new_gradient = self.get_gradient(_h, _w)
                    if new_gradient < cluster_gradient:  # 找到梯度更小的
                        cluster.update(_h, _w, self.data[_h][_w][0], self.data[_h][_w][1],
                                       self.data[_h][_w][2])  # 更新超像素位置
                        cluster_gradient = new_gradient  # 暂存新的超像素

    def assignment(self):
        # 为每个像素分配超像素（保存在超像素中）
        for cluster in self.clusters:  # 遍历超像素
            for h in range(cluster.h - 2 * self.S, cluster.h + 2 * self.S):  # 遍历超像素周围2S大小的区域
                if h < 0 or h >= self.image_height:  # 超出高度范围
                    continue
                for w in range(cluster.w - 2 * self.S, cluster.w + 2 * self.S):  # 遍历超像素周围2S大小的区域
                    if w < 0 or w >= self.image_width:  # 超出宽度范围
                        continue

                    # 计算度量
                    L, A, B = self.data[h][w]  # 暂存当前坐标的lab值
                    Dc = math.sqrt(  # 颜色距离
                        math.pow(L - cluster.l, 2) +
                        math.pow(A - cluster.a, 2) +
                        math.pow(B - cluster.b, 2))
                    Ds = math.sqrt(  # 距离距离
                        math.pow(h - cluster.h, 2) +
                        math.pow(w - cluster.w, 2))
                    # 距离度量
                    D = math.sqrt(math.pow(Dc / self.M, 2) + math.pow(Ds / self.S, 2))

                    if D < self.dis[h][w]:  # 找到距离更小的（2S的度量，原来不属于当前超像素的也会被算在其中）
                        if (h, w) not in self.label:  # 当前点没有记录超像素
                            self.label[(h, w)] = cluster  # 将当前点的超像素设置为当前超像素
                            cluster.pixels.append((h, w))  # 存储待更新超像素的像素点（后续需要专门的位置去进行更新）
                        else:  # 当前点有记录超像素
                            self.label[(h, w)].pixels.remove((h, w))  # 从原来的点的超像素的记录中移除当前点
                            self.label[(h, w)] = cluster  # 将当前点的超像素更新为现在的超像素
                            cluster.pixels.append((h, w))  # 记录当前的点
                        self.dis[h][w] = D

    def update_cluster(self):
        """
        将超像素的lab值取得属于当前超像素的像素最中间的限速的lab值
        :return:
        """
        for cluster in self.clusters:  # 遍历超像素
            sum_h = sum_w = number = 0
            for p in cluster.pixels:  # 遍历超像素的保存的内容
                sum_h += p[0]  #
                sum_w += p[1]
                number += 1
            _h = int(sum_h / number)
            _w = int(sum_w / number)
            # 将超像素的lab值取得属于当前超像素的像素最中间的限速的lab只
            cluster.update(_h, _w, self.data[_h][_w][0], self.data[_h][_w][1], self.data[_h][_w][2])

    def save_current_image(self, name):
        image_arr = np.copy(self.data)  # 当前图像
        for cluster in self.clusters:  # 遍历所有超像素
            # 修改图像的lab值为其对应的超像素的lab值
            for p in cluster.pixels:  # 遍历每个超像素对应像素点
                image_arr[p[0]][p[1]][0] = cluster.l
                image_arr[p[0]][p[1]][1] = cluster.a
                image_arr[p[0]][p[1]][2] = cluster.b
            # # 清空图像中超像素的lab值
            # image_arr[cluster.h][cluster.w][0] = 0
            # image_arr[cluster.h][cluster.w][1] = 0
            # image_arr[cluster.h][cluster.w][2] = 0
        self.save_lab_image(name, image_arr)  # 将图片存储

    def iterate_10times(self):
        self.init_clusters()  # 初始化超像素
        self.move_clusters()
        for _ in trange(10):
            self.assignment()
            self.update_cluster()
        name = path_exist('./SLIC_result/' + self.type) + '/{}.png'.format(self.number)
        self.save_current_image(name)

def path_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])
def get_image_file_list(path):
    image_filenames = [os.path.join(path, x) for x in os.listdir(path) if is_image_file(x)]
    return image_filenames

train_file_path = "./result/"
fake_image_filenames = get_image_file_list(train_file_path + "fake")
real_image_filenames = get_image_file_list(train_file_path + "real")


from multiprocessing.pool import ThreadPool

def process(item):
    # print(name + "Image: Processing({}/{})".format(i + 1, file_count))
    inumber = os.path.splitext(os.path.split(item)[1])[0] # 图像编号
    itype = item.split('/')[-2] # 图像类型
    if os.path.isfile('./SLIC_result/' + itype + '/{}.png'.format(inumber)):
        return
    p = SLICProcessor(item, 40, 10)
    p.iterate_10times()

def run(list_name):
    pool = ThreadPool()
    pool.map(process, list_name)
    pool.close()
    pool.join()

if __name__ == '__main__':
    run(fake_image_filenames)
    run(real_image_filenames)
    

# 参考

# https://blog.csdn.net/electech6/article/details/45509779

# https://www.jianshu.com/p/f2bc9dbbd9b2

# https://www.kawabangga.com/posts/1923
