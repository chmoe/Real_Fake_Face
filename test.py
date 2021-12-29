from config import Config
from Debug import Debug
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from pypeline.VGG import VGG
from pypeline.XCeption import XCeption
from pypeline.Resnet import Resnet
import numpy as np
from tqdm import tqdm
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class test(object):
    def __init__(self, k: int, frozen_layer: int = Config.frozen_layer_vgg16, net_name: str = Config.name_vgg16):
        self.K = k
        self.net_name = net_name
        self.label = {'fake': 0, 'real': 1}
        self.model = None
        self.generator = None
        self.image_width = 300
        self.image_height = 300
        self.batch_size = 32
        self.frozen_layer = frozen_layer

    def generate_validation(self, batch_size: int = 32, target_size: (int, int) = (256, 256)):
        head_data_path = Config.path(Config.slic_result_path, self.K, Config.name_validation)
        real_path = Config.path(head_data_path, Config.name_real)
        fake_path = Config.path(head_data_path, Config.name_fake)
        real_list = Config.get_child_folder(real_path)  # 存储真图片的文件夹 每张图片一个文件夹
        fake_list = Config.get_child_folder(fake_path)
        image_list = real_list + fake_list  # 所有图片的子文件夹
        boundary = len(real_list)
        max_len = len(image_list)

        steps = 0
        finish_flag = False
        while True:
            x_list = []  # 存储图片的数组，每次返回后清零 （len/batch, 20, 300, 300, 3）
            y_list = []
            for i in range(steps, min(steps + batch_size, max_len)):
                tmp_list = []  # （20, 300, 300, 3)
                for pic in Config.get_image_file_list(image_list[i]):  # 遍历每一个子文件夹（20）
                    tmp_list.append(np.array(image.load_img(
                        path=pic,
                        target_size=target_size
                    )))
                x_list.append(np.array(tmp_list))  # （20, 300, 300, 3)
                y_list.append(self.label[Config.name_real] if i < boundary else self.label[Config.name_fake])  # 1

            steps += batch_size
            if steps >= max_len:
                steps = 0
                finish_flag = True
            # (n, 20, 300, 300, 3)  (n, 1)
            yield np.array(x_list), np.array(y_list), finish_flag
            finish_flag = False

    def get_model(self) -> tf.keras.models.Model:
        shape = (self.image_width, self.image_height, 3)
        model = Model()
        if Config.name_vgg16 == self.net_name:
            return VGG(shape=shape, frozen_layer=self.frozen_layer).model()
        elif Config.name_resnet == self.net_name:
            return Resnet(shape=shape, frozen_layer=self.frozen_layer).model()
        elif Config.name_xception == self.net_name:
            return XCeption(shape=shape, frozen_layer=self.frozen_layer).model()
        model.load_weights(Config.path_exist(Config.model_path) + '{}_{}_fine-tuning.h5'.format(self.K, self.net_name))

        return model


    def calculation(self):
        self.model = self.get_model()
        data = {'TP':0, 'TN':0, 'FP':0, 'FN':0, 'ALL':0}
        value = next(self.generator)

        while not value[2]:  # 没有完成一个循环
            # Debug.info('进入循环')
            for i in range(len(value[0])):
                if value[1][i] == self.label['real']:  # 真实图片
                    predict_result = self.model.predict(value[0][i])
                    print(i, predict_result)
                    for res in predict_result:
                        if res >= 0.5:
                            data['TP'] += 1
                        else:
                            data['FP'] += 1
                        data['ALL'] += 1
                elif value[1][i] == self.label['fake']:  # fake图片
                    predict_result = self.model.predict(value[0][i])
                    for res in predict_result:
                        if res < 0.5:
                            data['TN'] += 1
                        else:
                            data['FN'] += 1
                        data['ALL'] += 1
                # Debug.info('加一')
            value = next(self.generator)

        return data

    def main(self):
        Debug.info('正在计算{}的内容'.format(self.K))
        self.generator = self.generate_validation(
                        batch_size=self.batch_size,
                        target_size=(self.image_width, self.image_height)
                    )
        data = self.calculation()
        self.save_history(self.K, data, Config.path_exist(Config.history_path) + 'validation_acc(calc).txt')

    @staticmethod
    def save_history(k ,history, result_file):
        if not os.path.exists(result_file):
            with open(result_file, 'w+') as fp:
                fp.write("K\tTP\tTN\tFP\tFN\tAcc\tPre\tRec\n")

        with open(result_file, "a") as fp:
            fp.write("%d\t%d\t%d\t%d\t%d\t%f\t%f\t%f\n" % (
                k, 
                history['TP'], 
                history['TN'], 
                history['FP'], 
                history['FN'],
                (history['TP'] + history['TN'])/(history['TP'] + history['TN'] + history['FP'] + history['FN']),
                history['TP']/(history['TP'] + history['FP']),
                history['TP']/(history['TP'] + history['FN']),
            ))



if __name__ == "__main__":
    Debug.info('今から検証を開始いたします')
    Debug.info('ソラは高性能ですから！')
    # for i in tqdm(range(20, 25)):
    #     test(i, frozen_layer=Config.frozen_layer_vgg16, net_name=Config.name_vgg16).main()
    #     test(i, frozen_layer=Config.frozen_layer_resnet, net_name=Config.name_resnet).main()
    #     test(i, frozen_layer=Config.frozen_layer_xception, net_name=Config.name_xception).main()
    for i in tqdm(range(40, 43)):
        test(i, frozen_layer=Config.frozen_layer_vgg16, net_name=Config.name_vgg16).main()
        test(i, frozen_layer=Config.frozen_layer_resnet, net_name=Config.name_resnet).main()
        test(i, frozen_layer=Config.frozen_layer_xception, net_name=Config.name_xception).main()
    for i in tqdm(range(43, 52)):
        test(i, frozen_layer=Config.frozen_layer_xception, net_name=Config.name_xception).main()