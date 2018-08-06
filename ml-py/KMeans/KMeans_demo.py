"""
机器学习实战--KMeans--demo
date：2018-7-16
author: 王建坤
"""
import numpy as np


# 加载数据集
def load_dataset(filename):
    data_list = []
    fr = open(filename)
    for line in fr.readlines():
        cur_line = line.strip().split('\t')
        map_line = list(map(float, cur_line))
        data_list.append(map_line)
    return data_list


# 计算欧式距离，
def calc_dist(vec1, vec2):
    # 转为 np.array 才能正确的计算
    return np.sum((np.array(vec1) - np.array(vec2))**2)**0.5


# 随机选取k个中心向量
def create_rand_center(data_set, k):
    n = data_set.shape[1]
    # 创建矩阵存放簇的中心向量
    centers = np.mat(np.zeros((k, n)))
    for j in range(n):
        j_min = min(data_mat[:, j])
        j_max = max(data_mat[:, j])
        j_range = float(j_max - j_min)
        centers[:, j] = j_min + j_range*np.random.rand(k, 1)
    return centers


# KMeans算法
def k_means(data_set, k):
    m = data_set.shape[0]
    # 存放各个样本的簇标记，及该样本与簇中心的距离
    cluster_result = np.mat(np.zeros((m, 2)))
    # 获得初始化的簇中心
    center = create_rand_center(data_set, k)
    # 簇中心的变化标志
    center_change = True
    while center_change:
        center_change = False
        # 对每个样本循环
        for i in range(m):
            min_dist = float('inf')
            min_index = -1
            # 对每个类循环
            for j in range(k):
                # 计算样本与簇中心的距离
                center_dist = calc_dist(center[j, :], data_set[i, :])
                # 如果样本与簇中心的距离小于与其它簇中心的最小距离，则重置最小距离和改变该样本的簇标记
                if center_dist < min_dist:
                    min_dist = center_dist
                    min_index = j
            # 如果簇标记发生改变，更新簇标记、距离和簇中心变化标志
            if cluster_result[i, 0] != min_index:
                cluster_result[i, 0] = min_index
                cluster_result[i, 1] = min_dist**2
                center_change = True
        # print(center)
        # 更新每个簇的中心向量
        for c in range(k):
            # 得到该簇的所有样本集。np.nonzero()用于找出不等于0的所有下标
            k_mat = data_set[np.nonzero(cluster_result[:, 0] == c)[0]]
            # 平均该簇样本集得到新的簇中心向量
            center[c, :] = np.mean(k_mat, axis=0)
    return center, cluster_result


if __name__ == '__main__':
    # 加载数据集
    data_mat = np.mat(load_dataset('testSet.txt'))
    # 使用KMeans算法
    centers, data_cluster = k_means(data_mat, 4)
    print(centers)
