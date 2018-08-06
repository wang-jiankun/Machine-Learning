"""
Scikit-Learn 库使用--模型的训练方法--线性回归、Logistic 回归和 Softmax回归
date：2018-7-18
author: 王建坤
"""
import numpy as np
from matplotlib import pyplot as plt


# 创建线性回归数据集
def create_dataset():
    X = 2 * np.random.rand(100, 1)
    # 结果加上高斯噪声
    y = 4 + 3*X + np.random.randn(100, 1)
    return X, y


# 线性回归解析法：使用正态方程求解，直接得到全局最优解
def linear_regression_analysis(X, y):
    # 特征向量为参数b添加值为1的特征
    X_b = np.c_[np.ones((100, 1)), X]
    # 用正态方程解得全局最优解
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    print("线性回归解析解为：", theta_best)
    # 预测
    sample = np.array([[0], [2]])
    sample_b = np.c_[np.ones((2, 1)), sample]
    predict = sample_b.dot(theta_best)
    # print('解析解方程预测为：', predict)
    # 绘制线性回归模型图像
    plt.plot(sample, predict, 'r-')
    plt.plot(X, y, 'b.')
    plt.axis([0, 2, 0, 15])
    plt.show()
    return X_b


# 使用sk-learn的线性回归模型，默认使用解析法
def linear_regression_sk(X, y):
    from sklearn.linear_model import LinearRegression
    # 创建线性回归模型实例
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    print('sk-learn线性回归解析解：', 'b：', lin_reg.intercept_, 'w：', lin_reg.coef_)


# 线性回归批量梯度下降法（batch gradient descent）
def linear_regression_batch_gd(X_b, y):
    # 学习率不变、迭代次数和样本数
    learning_rate = 0.1
    max_iterations = 1000
    m = 100
    # 随机初始值
    theta = np.random.randn(2, 1)
    # 开始迭代
    for n in range(max_iterations):
        gradients = 2/m * X_b.T.dot(X_b.dot(theta)-y)
        theta = theta - learning_rate*gradients
    print('线性回归批量梯度下降法解：', theta)


# 线性回归随机梯度下降法（stochastic gradient descent）
def linear_regression_stochastic_gd(X_b, y):
    # epoch次数，样本数
    n_epochs = 50
    m = 100
    theta = np.random.randn(2, 1)
    for epoch in range(n_epochs):
        for i in range(m):
            random_index = np.random.randint(m)
            xi = X_b[random_index:random_index+1]
            yi = y[random_index:random_index+1]
            gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
            learning_rate = 1.0/(epoch*m + i + 10)
            theta = theta - learning_rate*gradients
    print('线性回归随机梯度下降法解：', theta)


# sk-learn 线性回归随机梯度下降
def linear_regression_stochastic_gd_sk(X, y):
    from sklearn.linear_model import SGDRegressor
    sgd_reg = SGDRegressor(n_iter=50, penalty=None, eta0=0.1)
    sgd_reg.fit(X, y.ravel())
    print('sk-learn线性回归随机梯度下降法解：',  'b：', sgd_reg.intercept_, 'w：', sgd_reg.coef_)


# 创建线性回归数据集
def create_dataset_poly():
    m = 100
    X1 = 6 * np.random.rand(m, 1) - 3
    y1 = 0.5 * X1 ** 2 + X1 + 2 + np.random.randn(m, 1)
    return X1, y1


# 多项式回归
def polynomial_regression(X, y):
    # 添加二次特征
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    lin_reg_poly = LinearRegression()
    lin_reg_poly.fit(X_poly, y)
    print('多项式回归解：', 'b：', lin_reg_poly.intercept_, 'w：', lin_reg_poly.coef_)
    return lin_reg_poly


# 绘制关于训练集规模的学习曲线
def plot_learning_curves(model, X, y):
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.show()


# 岭回归，l2正则化，封闭方程求解
def ridge_regression_analysis(X, y):
    from sklearn.linear_model import Ridge
    ridge_reg = Ridge(alpha=1, solver="cholesky")
    ridge_reg.fit(X, y)
    print('岭回归解：', 'b：', ridge_reg.intercept_, 'w：', ridge_reg.coef_)


# Lasso 回归，l2正则化，封闭方程求解
def lasso_regression_analysis(X, y):
    from sklearn.linear_model import Lasso
    lasso_reg = Lasso(alpha=0.1)
    lasso_reg.fit(X, y)
    print('Lasso 回归解：', 'b：', lasso_reg.intercept_, 'w：', lasso_reg.coef_)


# l2,l1正则化，梯度下降求解
def regularization_regression_gd(X, y):
    from sklearn.linear_model import SGDRegressor
    # l1正则化把 penalty="l2" 改为 penalty="l1"
    sgd_reg = SGDRegressor(penalty="l2")
    sgd_reg.fit(X, y.ravel())
    print('l2梯度下降法解：', 'b：', sgd_reg.intercept_, 'w：', sgd_reg.coef_)


# 弹性网路正则化，即l1、l2混合正则化
def elasticnet_regression_gd(X, y):
    from sklearn.linear_model import ElasticNet
    # l1_ratio 指的就是混合率, 即l1正则化占的比例
    elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
    elastic_net.fit(X, y)
    print('弹性网络解：', 'b：', elastic_net.intercept_, 'w：', elastic_net.coef_)


# 早期停止法（Early Stopping）
def early_stoping(X, y):
    from sklearn.base import clone
    from sklearn.linear_model import SGDRegressor
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    # 当warm_start=True时，调用fit()方法后，训练会从停下来的地方继续，而不是从头重新开始。
    sgd_reg = SGDRegressor(max_iter=1, warm_start=True, penalty=None, learning_rate="constant", eta0=0.0005)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    minimum_val_error = float("inf")
    best_epoch = None
    best_model = None
    for epoch in range(1000):
        sgd_reg.fit(X_train,  y_train.ravel())
        y_val_predict = sgd_reg.predict(X_val)
        val_error = mean_squared_error(y_val_predict, y_val)
        if val_error < minimum_val_error:
            minimum_val_error = val_error
            best_epoch = epoch
            best_model = clone(sgd_reg)
    print('stopping in:', best_epoch)


# 加载鸢尾花数据集
def load_dataset_flower():
    from sklearn import datasets
    iris = datasets.load_iris()
    # X_f = iris['data']
    # y_f = iris['target']
    # print('加载鸢尾花数据集成功：', iris)
    return iris


# logistic 回归
def logistic_classify(iris):
    from sklearn.linear_model import LogisticRegression
    X = iris["data"][:, 3:]  # petal width
    y = (iris["target"] == 2).astype(np.int)
    log_reg = LogisticRegression()
    log_reg.fit(X, y)
    # 绘图
    X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
    y_proba = log_reg.predict_proba(X_new)
    plt.plot(X_new, y_proba[:, 1], "g-", label="Iris-Virginica")
    plt.plot(X_new, y_proba[:, 0], "b--", label="Not Iris-Virginica")
    plt.show()


# softmax 回归多分类
def softmax_classify(iris):
    from sklearn.linear_model import LogisticRegression
    # 划分数据集
    X = iris["data"][:, (2, 3)]  # petal length, petal width
    y = iris["target"]
    # 创建 softmax 回归实例
    softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10)
    softmax_reg.fit(X, y)
    # 预测
    predict = softmax_reg.predict([[5, 2]])
    predict_pro = softmax_reg.predict_proba([[5, 2]])
    print('softmax回归预测为：', predict, '各类概率为', predict_pro)


if __name__ == '__main__':
    # 获得线性回归数据集
    X, y = create_dataset()
    # 线性回归解析法
    # X_b = linear_regression_analysis(X, y)
    # sk-learn线性回归解
    # linear_regression_sk(X, y)
    # 线性回归批量梯度下降法
    # linear_regression_batch_gd(X_b, y)
    # 线性回归随机梯度下降法
    # linear_regression_stochastic_gd(X_b, y)
    # sk-learn线性回归随机梯度下降法
    # linear_regression_stochastic_gd_sk(X, y)
    # 获得多项式回归数据集
    # X1, y1 = create_dataset_poly()
    # 多项式回归解
    # lin_reg_poly = polynomial_regression(X1, y1)
    # 获得关于训练集规模的学习曲线
    # plot_learning_curves(lin_reg_poly, X1, y1)
    # 岭回归，l2正则化
    # ridge_regression_analysis(X, y)
    # lasso回归，l1正则化
    # lasso_regression_analysis(X, y)
    # 梯度下降法的正则化
    # regularization_regression_gd(X, y)
    # 弹性网络
    # elasticnet_regression_gd(X, y)
    # 早期停止
    # early_stoping(X1, y1)
    # 加载花的数据集
    iris = load_dataset_flower()
    # logistic 回归二分类
    logistic_classify(iris)
    # softmax 多分类
    softmax_classify(iris)
