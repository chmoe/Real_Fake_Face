# -*- coding: UTF-8 -*-
# @Project: Progress 
# @File: Debug 
# @Author: Henry Ng 
# @Date: 2021/12/06 13:32
import os
import time


class Debug(object):
    @staticmethod
    def default(info_type, info_str):
        print('[' + info_type + ']' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + info_str)

    @staticmethod
    def info(*info_str):
        tmp_str = ''
        for i in info_str:
            tmp_str += str(i)
            tmp_str += ', '
        Debug.default('info', tmp_str)


