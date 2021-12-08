# -*- coding: UTF-8 -*-
# @Project: Progress 
# @File: SLIC 
# @Author: Henry Ng 
# @Date: 2021/12/06 13:00
import math

from config import Config
from Debug import Debug
from pypeline.Split import Split
from skimage.segmentation import slic, mark_boundaries
from skimage import io, color
from tqdm import tqdm


path = Config.original_face_path


class SLIC(object):

    # @staticmethod
    def __init__(self, k: int, m: int, validation_split: float):
        self.K = k
        self.M = m
        self.validation_split = validation_split
        self.fake_image_filenames = Config.get_image_file_list(path + Config.name_fake)
        self.real_image_filenames = Config.get_image_file_list(path + Config.name_real)

    @staticmethod
    def process_inner(img, segment, compactness):
        rgb = io.imread(img)
        segments = slic(rgb, n_segments=segment, compactness=compactness, )
        superpixels = color.label2rgb(segments, rgb, kind='avg')
        return superpixels

    def run(self, list_name: str, real_or_fake: str):
        """
        用于内部调用和处理，最终进行保存
        :param list_name: 需要处理的文件路径列表
        :param real_or_fake: 是真还是假
        :return:
        """
        file_count = len(list_name)  # 文件数
        switch_count = math.ceil(self.validation_split * file_count)  # 分割训练集和验证集
        path_train = Config.path(Config.slic_result_path, self.K, Config.name_train, real_or_fake)
        path_validation = Config.path(Config.slic_result_path, self.K, Config.name_validation, real_or_fake)
        # 训练集
        for i in tqdm(range(switch_count)):
            arr = self.process_inner(list_name[i], self.K, self.M)
            Split(arr, list_name[i]).traversal(i, path_train, False)
        # 验证集
        for i in tqdm(range(file_count - switch_count + 1)):
            arr = self.process_inner(list_name[i + switch_count], self.K, self.M)
            Split(arr, list_name[i + switch_count]).traversal(i, path_validation, True)

    def process(self):
        Debug.info("SLIC_Processing: {} images with {} slices".format(Config.name_fake, self.K))
        self.run(self.fake_image_filenames, Config.name_fake)
        Debug.info("SLIC_Processing: {} images with {} slices".format(Config.name_real, self.K))
        self.run(self.real_image_filenames, Config.name_real)

