# main.py
# Author: [sixwaaaay](https://github.com/sixwaaaay)
import functools
import operator
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, linear_kernel, laplacian_kernel
import gradio as gr
import kernels

# 用于组合的函数
func = namedtuple('func', ['func', 'param', 'weight'])

# 使用的核函数，参数，权重
funcs = [
    func(rbf_kernel, [0.5], 1),
    func(polynomial_kernel, [3, 0.5, 1], 1),
    func(linear_kernel, [], 1),
    func(laplacian_kernel, [0.5], 1),
    func(kernels.rbf(0.5), [], 1),  # 自行实现的 RBF 核函数
]


def svm_classification(
        use_gaussian: bool,
        use_poly: bool,
        use_linear: bool,
        use_laplacian: bool,
        use_custom: bool,
        use_multiply: bool,
):
    """

    :param use_gaussian: 使用高斯核
    :param use_poly: 使用多项式核
    :param use_linear: 使用线性核
    :param use_laplacian: 使用拉普莱斯核
    :param use_custom: 使用自定义核
    :param use_multiply:
    :return:
    """
    selected = [use_gaussian, use_linear, use_poly, use_laplacian, use_custom]
    selected_funcs = [funcs[i] for i in range(len(selected)) if selected[i]]
    op = operator.mul if use_multiply else operator.add

    # 自定义的组合核函数
    def kernel(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return functools.reduce(
            op,
            (f.func(X, Y, *f.param) * f.weight for f in selected_funcs)
        )

    return svm_with_custom_kernel(kernel)


def svm_with_custom_kernel(kernel_function):
    # 加载数据
    iris = datasets.load_iris()
    X, Y = iris.data[:, :2], iris.target
    # 定义使用自定义核函数的分类器并进行训练
    clf = svm.SVC(kernel=kernel_function)
    clf.fit(X, Y)
    # 预测，绘制决策面等
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired, edgecolors="k")
    plt.title("3-Class classification SVM with use_custom kernel")
    plt.axis("tight")
    return plt, clf.score(X, Y)


iface = gr.Interface(fn=svm_classification,
                     inputs=["checkbox", "checkbox", "checkbox", "checkbox", "checkbox",
                             "checkbox"],
                     outputs=['plot', "number"],
                     title="组合核函数",
                     description="核函数组合").launch()
