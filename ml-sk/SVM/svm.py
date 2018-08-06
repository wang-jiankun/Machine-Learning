"""
Scikit-Learn 库使用--svm
date：2018-7-19
author: 王建坤
"""
import numpy as np


# 加载鸢尾花数据集
def load_dataset():
    from sklearn import datasets
    iris = datasets.load_iris()
    # print(iris)
    # 使用第3和第4个特征
    X = iris['data'][:, (2, 3)]
    # bool类型转为数值型
    y = (iris['target'] == 2).astype(np.float64)
    return X, y


# 线性SVM二分类器
def linear_svm_classify(X, y):
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
    # 特征缩放
    scale = StandardScaler()
    X_scaled = scale.fit_transform(X.astype(np.float64))
    # C是损失函数项的系数，称为惩罚系数
    svm_clf = LinearSVC(C=1, loss='hinge')
    svm_clf.fit(X_scaled, y)
    # 预测
    predict = svm_clf.predict([[5.5, 1.7]])
    print('线性SVM二分类器预测为：', predict)


# 非线性SVM二分类器、
def nonlinear_svm_classify(X, y):
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.svm import LinearSVC
    # 增加多项式特征
    poly_features = PolynomialFeatures(degree=3)
    X_poly = poly_features.fit_transform(X.astype(np.float64))
    # 特征缩放
    scale = StandardScaler()
    X_scaled = scale.fit_transform(X_poly)
    # 需要指定损失函数
    poly_svm_clf = LinearSVC(C=10, loss='hinge')
    poly_svm_clf.fit(X_scaled, y)
    # 预测，先把样本特征向量转为包含多项式特征得向量
    sample_poly = poly_features.fit_transform([[5.5, 1.7]])
    predict = poly_svm_clf.predict(sample_poly)
    print('非线性SVM二分类器预测为：', predict)


# 多项式核SVM二分类器
def kernel_poly_svm_classify(X, y):
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    # 特征缩放
    scale = StandardScaler()
    X_scaled = scale.fit_transform(X)
    # 使用多项式核，
    kernel_poly_svm_clf = SVC(kernel="poly", degree=3, coef0=1, C=5)
    kernel_poly_svm_clf.fit(X_scaled, y)
    # 预测，特征向量不用转换
    predict = kernel_poly_svm_clf.predict([[5.5, 1.7]])
    print('核SVM二分类器预测为：', predict)


# 高斯RBF核SVM二分类器
def kernel_rbf_svm_classify(X, y):
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    # 特征缩放
    scale = StandardScaler()
    X_scaled = scale.fit_transform(X)
    # 增大gamma和减小C都会加大拟合
    kernel_rbf_svm_clf = SVC(kernel="rbf", gamma=5, C=1)
    kernel_rbf_svm_clf.fit(X_scaled, y)
    # 预测，特征向量不用缩放
    predict = kernel_rbf_svm_clf.predict([[5.5, 1.7]])
    print('高斯RBF核SVM二分类器预测为：', predict)


if __name__ == '__main__':
    # 加载数据集
    X, y = load_dataset()
    # 线性SVM二分类器
    # linear_svm_classify(X, y)
    # 非线性SVM二分类器
    # nonlinear_svm_classify(X, y)
    # 多项式核SVM二分类器
    kernel_poly_svm_classify(X, y)
    # 高斯rbf核SVM二分类器
    kernel_rbf_svm_classify(X, y)
