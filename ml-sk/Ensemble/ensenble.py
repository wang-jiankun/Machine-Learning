"""
Scikit-Learn 库使用--集成学习--bagging（有放回采样）、随机森林、Adaboost
date：2018-7-19
author: 王建坤
"""


# 加载数据集
def load_dataset():
    from sklearn.datasets import make_moons
    X, y = make_moons()
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test


# bagging 集成分类
def bagging_classify(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import BaggingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500, bootstrap=True, n_jobs=-1, oob_score=True)
    bag_clf.fit(X_train, y_train)
    # 样本在各个类上的概率
    # print(bag_clf.oob_decision_function_)
    # bagging分类器在Out-of-Bag上的精度
    print('bagging分类器在Out-of-Bag上的精度为', bag_clf.oob_score_)
    # bagging分类器在测试集上的精度
    y_pred = bag_clf.predict(X_test)
    print('bagging分类器在测试集上的精度为', accuracy_score(y_test, y_pred))


# 随机森林分类器
def randomforest_classify(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
    rnd_clf.fit(X_train, y_train)
    y_pred_rf = rnd_clf.predict(X_test)
    print('随机森林精度为', accuracy_score(y_test, y_pred_rf))


# Adaboost分类器
def adaboost_classify(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    # 200个决策桩组成
    ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200,
                                 algorithm="SAMME.R", learning_rate=0.5)
    ada_clf.fit(X_train, y_train)
    y_pred_ada = ada_clf.predict(X_test)
    print('Adaboost分类器精度为', accuracy_score(y_test, y_pred_ada))


if __name__ == '__main__':
    # 加载数据集
    X_train, X_test, y_train, y_test = load_dataset()
    # bagging 集成分类器
    bagging_classify(X_train, X_test, y_train, y_test)
    # 随机森林分类器
    randomforest_classify(X_train, X_test, y_train, y_test)
    # Adaboost分类器
    adaboost_classify(X_train, X_test, y_train, y_test)