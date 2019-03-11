'''
显示密度图
'''
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

density_map = np.loadtxt(open(r'.\data\original\shanghaitech\part_A_final\test_data\ground_truth_csv\IMG_11.csv', 'rb'),
                         delimiter=',', skiprows=0)
# np.savetxt('new.csv', my_matrix, delimiter = ',') #将数组或者矩阵存储为csv文件

# 保存密度图
# density_map = 255 * density_map / np.max(density_map)
# cv2.imwrite(os.path.join('.', 'test1.png'), density_map)

density_map_1=density_map[1:200,1:200]
density_zero=np.zeros_like(density_map)
# 显示密度图
plt.imshow(density_map)
# plt.imshow(density_map_1)
# plt.imshow(density_zero)
plt.show()


# '''
# 分析模型内容
# '''
#
# from src.crowd_count import CrowdCounter
# import h5py
# import numpy as np
# import torch
#
# net = CrowdCounter()
# fname = r'.\src\mcnn_shtechA_0.h5'
#
# h5f = h5py.File(fname, mode='r')
# for k, v in net.state_dict().items():
#     print(k)
#     print(np.asarray(h5f[k]))
#     param = torch.from_numpy(np.asarray(h5f[k]))
#     v.copy_(param)
#
#
# '''
# 测试文件复制
# '''
# import os
# import shutil
# shutil.copyfile(r'./saved_models/mcnn_shtechA_0.h5',r'.\src\mcnn_shtechA_0.h5')


# '''
# 分析ground_truth.csv
# '''
# import numpy as np
#
# density_map = np.loadtxt(open(r'.\data\original\shanghaitech\part_A_final\test_data\ground_truth_csv\IMG_1.csv', 'rb'),
#                          delimiter=',', skiprows=0)
# count = np.sum(density_map)
# print(density_map)  # density_map是图片的灰度图经过高斯函数处理之后的结果，将该数组求和即可得到人群数量。
# print(count)


# '''
# 分析数据导入过程
# '''
# from src.data_loader import ImageDataLoader
#
# train_path = './data/formatted_trainval/shanghaitech_part_A_patches_9/train'
# train_gt_path = './data/formatted_trainval/shanghaitech_part_A_patches_9/train_den'
# val_path = './data/formatted_trainval/shanghaitech_part_A_patches_9/val'
# val_gt_path = './data/formatted_trainval/shanghaitech_part_A_patches_9/val_den'
#
# data_loader = ImageDataLoader(train_path, train_gt_path, shuffle=True, gt_downsample=True, pre_load=True)
# data_loader_val = ImageDataLoader(val_path, val_gt_path, shuffle=False, gt_downsample=True, pre_load=True)


# '''
# 分析模型参数
# '''
#
# from src.crowd_count import CrowdCounter
# from src import network
#
# net = CrowdCounter()
# network.load_net(r'.\final_models\mcnn_shtechA_660.h5', net)
# net.cuda()
# net.train()
#
# params = list(net.parameters())
# # print(params)
# print(params[0])
# print(params[0].shape)  # 第一个卷积核的W：（16,1,9,9）
# print(params[1].shape)  # 第一个卷积核的b:  (16,)
# print(params[2].shape)  # 第一个卷积核的W：（32,16,7,7）


