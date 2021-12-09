# -*- coding: UTF-8 -*-
# @Project: Progress 
# @File: main 
# @Author: Henry Ng 
# @Date: 2021/12/06 11:46

from config import Config
from Debug import Debug
import os
import sys
from pypeline import *

sys.path.append('./pypeline/')


def net_run(i_control: int):
    Net(k=i_control, frozen_layer=Config.frozen_layer_vgg16, net_name=Config.name_vgg16).run()
    Net(k=i_control, frozen_layer=Config.frozen_layer_resnet, net_name=Config.name_resnet).run()
    Net(k=i_control, frozen_layer=Config.frozen_layer_xception, net_name=Config.name_xception).run()


class Main(object):
    def __init__(self):
        self.m = 10
        self.validation_split = 0.8

    def loop(self):
        for i in range(20, 65):
            SLIC(k=i, m=self.m, validation_split=self.validation_split).process()
            net_run(i)


if __name__ == "__main__":
    Debug.info('プログラミングは今から実行します')
    Debug.info('ソラは高性能ですから！')
    Main().loop()
