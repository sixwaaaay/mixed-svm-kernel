
# SVM 的核函数组合运用

# 演示程序运行
```bash
python main.py
```

注意安装依赖


# SVM、SVC 的核函数方法签名
SVM、SVM 允许自行定义核函数，对输入输出的参数说明如下
```python
def my_kernel(X: np.ndarry, Y: np.ndarry)->np.ndarry:
```
补充：

- X 的形状为 **(N, K)**
- Y 的形状为** (M,K)**
- 返回的形状**要求**为** (N, M)**

<br />
## 一个简单的示例 
假设数据集的特征数量为 2 
```python
def naive_kernel(X: np.ndarry, Y: np.ndarry)->np.ndarry:
    """
                 (2  0)
    k(X, Y) = X  (    ) Y.T
                 (0  1)
    """
    M = np.array([[2, 0], [0, 1.0]])
    return np.dot(np.dot(X, M), Y.T)
```
# 已有核函数
位于`sklearn.metrics.pairwise` 下

例如：

- 高斯核 `rbf_kernel`
- 多项式核 `polynomial_kernel`
- 线性核 `linear_kernel`
- 拉普拉斯核 `laplacian_kernel`

等等
# 核的组合使用
以使用乘法聚合多个函数为例
```python
# 用于组合的函数
func = namedtuple('func', ['func', 'param', 'weight'])# 函数对象，参数列表，所占权重

# 使用的核函数，参数，权重
funcs = [
    func(rbf_kernel, [0.5], 1),
    func(polynomial_kernel, [3, 0.5, 1], 1),
    func(linear_kernel, [], 1),
    func(laplacian_kernel, [0.5], 1),
]

# 自定义的组合核函数
def kernel(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return functools.reduce(
        operator.mul, 
        (f.func(X, Y, *f.param) * f.weight for f in funcs)
    )
```

以经典的 IRIS 数据集分类为例，运用该核函数进行分类操作

```python
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
plt.plot()
```
如果运行代码可以看到相应的分类图

此处，展示一下更多的组合状况，比如说

- 乘法或者加法作为聚合操作
- 任选几个函数

[WEB 演示](http://127.0.0.1:7860/)
# 自定义核函数
有时候可能需要自行定义核函数的实现

一般来说都是定义的运算都类似于矩阵乘法的运算
```python
A: np.ndarray  # (n x k)
B: np.ndarray  # (m x k)
result: np.ndarray  # (n x m)
for i in range(A.shape[0]):
    for j in range(B.shape[0]):
        result[i, j] = np.dot(A[i], B[j])
        # for k in range(A.shape[1]):
        #    result[i, j] += A[i, k] * B[j, k]
# 对于矩阵乘法，可以优化成 A @ B.T
```
类似地，RBF 是在循环内部求向量差的范数（欧几里得距离）再做指数运算

但是这样就慢了！

这里有时候可以利用 Numpy 等矩阵运算的广播机制，向量化整个操作。<br />例如矩阵乘法这种**对应位相乘**再相加可以通过如下方式实现
```python
(A[:,None] * B).sum(axis=2)
```
通过这种向量化的优化，可以带来相当大的性能提升。
## 实现RBF函数
使用 for 循环实现
```python
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
```

向量化实现

```python
# 高斯
def rbf(gamma=0.5):  # 高阶函数
    coef = -1 / (2 * gamma ** 2)

    def kernel_function(X: np.ndarray, Y: np.ndarray):
        return np.exp(coef * np.square(X[:, None] - Y).sum(axis=2))

    # 先增维，再计算，最后再降维
    return kernel_function
```

性能测试
```python
    from timeit import timeit
    X = np.random.randn(80, 4)
    naive = naive_rbf(1)
    vec = rbf(1)
    setup = 'from __main__ import X, vec, naive;import numpy as np'
    num = 1000
    t1 = timeit('naive(X, X)', setup=setup, number=num)
    t2 = timeit('vec(X, X)', setup=setup, number=num)
    print('Speed difference: {:0.3f}x'.format(t1 / t2))
```
以R7 4800U 8核16线程机器测试，结果为

Speed difference: **167.814x**
# 总结

- 可以使用Python函数作为核函数
- 对于自定义的操作可以有向量化的实现

# 参考

1. [Using Python functions as kernels](https://scikit-learn.org/stable/modules/svm.html#using-python-functions-as-kernels)
1. [kernel function](https://zh.wikipedia.org/wiki/%E5%BE%84%E5%90%91%E5%9F%BA%E5%87%BD%E6%95%B0%E6%A0%B8)
1. [numpy 广播机制](https://numpy.org.cn/article/basics/python_numpy_tutorial.html#%E5%B9%BF%E6%92%AD-broadcasting)
