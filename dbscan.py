# coding: utf-8
# 测试用例

from sklearn import manifold, cluster
import numpy
# import matplotlib  # 画图用
# import matplotlib.pyplot as plt  # 画图用

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


a = numpy.loadtxt('hogHist.txt')
n_neighbors = 12
n_components = 1
Y = manifold.Isomap(n_neighbors, n_components).fit_transform(a)
max_min_normalization(Y)


b = numpy.loadtxt('results\\mean.txt')
b = b[:, numpy.newaxis]  # 转换为二维数组
max_min_normalization(b)

c = numpy.concatenate((Y, b), axis=1)

var = numpy.loadtxt('results\\dbvar.txt')
eps = var[0]
min_samples = var[1]


# dbscan
db = cluster.DBSCAN(eps, min_samples).fit(c)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

f = open("results\\dbscan_nlabels.txt", 'wb')
f.write(str(n_clusters_))
f.close()
numpy.savetxt("results\\dbscan_labels.txt", labels, fmt='%d')

# 画图
# axes = plt.subplot(111)
# type1_x = []
# type1_y = []
# type2_x = []
# type2_y = []
# type3_x = []
# type3_y = []
# type4_x = []
# type4_y = []
# print 'range(len(labels)):'
# print (len(labels))
# for i in range(len(labels)):
#     if labels[i] == 0:
#         type1_x.append(c[i][0])
#         type1_y.append(c[i][1])
#     if labels[i] == 1:
#         type2_x.append(c[i][0])
#         type2_y.append(c[i][1])
#     if labels[i] == 2:
#         type3_x.append(c[i][0])
#         type3_y.append(c[i][1])
#     if labels[i] == -1:
#         type4_x.append(c[i][0])
#         type4_y.append(c[i][1])
# type1 = axes.scatter(type1_x, type1_y, s=40, c='red')
# type2 = axes.scatter(type2_x, type2_y, s=40, c='green')
# type3 = axes.scatter(type3_x, type3_y, s=40, c='blue')
# type4 = axes.scatter(type4_x, type4_y, s=40, c='black')
#
# plt.xlabel('x1')
# plt.ylabel('x2')
# axes.legend((type1, type2, type3, type4), ('0', '1', '2', '3'), loc=1)
# plt.show()

