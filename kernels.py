# kernels.py
# Author: [sixwaaaay](https://github.com/sixwaaaay)


import numpy as np


# 高斯
def rbf(gamma=0.5):  # 高阶函数
    coef = -1 / (2 * gamma ** 2)

    def kernel_function(X: np.ndarray, Y: np.ndarray):
        return np.exp(coef * np.square(X[:, None] - Y).sum(axis=2))

    # 先增维，再计算，最后再降维
    return kernel_function


# 不那么高效的方法 rbf 实现
def naive_rbf(gamma=0.5):  # 仅示意
    coef = -1 / (2 * gamma ** 2)

    def kernel_function(X: np.ndarray, Y: np.ndarray):
        def naive_rbf_compute(va, vb):
            return np.exp(coef * np.square(va - vb).sum())

        dot = np.zeros((X.shape[0], Y.shape[0]))
        for i in range(X.shape[0]):
            for j in range(Y.shape[0]):
                dot[i, j] = naive_rbf_compute(X[i], Y[j])
        return dot

    return kernel_function


# 多项式
def polynomial(coef0=1, degree=2):
    def kernel_function(X, Y):
        return (coef0 + X @ Y.T) ** degree

    return kernel_function


# 线性
def linear():
    return lambda X, Y: np.dot(X, Y.T)


if __name__ == '__main__':
    from timeit import timeit
    X = np.random.randn(80, 4)
    naive = naive_rbf(1)
    vec = rbf(1)
    setup = 'from __main__ import X, vec, naive;import numpy as np'
    num = 1000
    t1 = timeit('naive(X, X)', setup=setup, number=num)
    t2 = timeit('vec(X, X)', setup=setup, number=num)
    print('Speed difference: {:0.3f}x'.format(t1 / t2))
