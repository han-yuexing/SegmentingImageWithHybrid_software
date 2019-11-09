# coding: utf-8
# 测试用例

from sklearn import manifold, cluster
import numpy
import matplotlib
import matplotlib.pyplot as plt

# 归一化
def max_min_normalization(data_value):
    data_shape = data_value.shape
    data_rows = data_shape[0]
    data_cols = data_shape[1]
    data_col_max_values = data_value.max(axis=0)
    data_col_min_values = data_value.min(axis=0)

    for i in xrange(0, data_rows, 1):
        for j in xrange(0, data_cols, 1):
            data_value[i][j] = \
                (data_value[i][j] - data_col_min_values[j]) / \
                (data_col_max_values[j] - data_col_min_values[j])


a = numpy.loadtxt('results\\hogHist.txt')
n_neighbors = 12
n_components = 1
Y = manifold.Isomap(n_neighbors, n_components).fit_transform(a)
max_min_normalization(Y)


b = numpy.loadtxt('results\\mean.txt')
b = b[:, numpy.newaxis]  # 转换为二维数组
max_min_normalization(b)

c = numpy.concatenate((Y, b), axis=1)

var = numpy.loadtxt('results\\kmeansvar.txt')
n_clusters = int(var[0])

# kmeans
kmeans = cluster.KMeans(n_clusters, random_state=0).fit(c)
labels = kmeans.labels_

numpy.savetxt("results\\kmeans_labels.txt", labels, fmt='%d')
