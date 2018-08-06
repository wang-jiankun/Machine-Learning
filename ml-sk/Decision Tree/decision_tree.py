"""
Scikit-Learn 库使用--决策树
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
    y = iris['target']
    return X, y, iris


# 决策树分类器
def tree_classify(X, y):
    from sklearn.tree import DecisionTreeClassifier
    tree_clf = DecisionTreeClassifier(max_depth=2)
    tree_clf.fit(X, y)
    print(tree_clf.tree_)
    return tree_clf


# 绘制决策树图
def draw_tree(model, iris):
    from sklearn.tree import export_graphviz
    export_graphviz(
        model,
        out_file="iris_tree.dot",
        feature_names=iris.feature_names[2:],
        class_names=iris.target_names,
        rounded=True,
        filled=True
    )
    # import pydotplus
    # dot_data = export_graphviz(model, out_file=None)
    # graph = pydotplus.graph_from_dot_data(dot_data)
    # graph.write_pdf("iris.pdf")


# 预测
def tree_predict(model, sample):
    # 预测类别
    predict = model.predict(sample)
    # 属于各类别的概率
    predict_prob = model.predict_proba(sample)
    print('决策树预测类别为：', predict, '属于各类别的概率为：', predict_prob)


if __name__ == '__main__':
    # 加载数据集
    X, y, iris = load_dataset()
    # 创建决策树分类器
    tree_clf = tree_classify(X, y)
    # 绘制决策树
    # draw_tree(tree_clf, iris)
    # 预测
    tree_predict(tree_clf, [[5, 1.5]])