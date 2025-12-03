from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.measure import marching_cubes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 加载Iris数据集
iris = load_iris()
X = iris.data[:, [1,2,3]]  # 三个特征
y = iris.target

# 只保留前两类
mask = y < 2
X = X[mask]
y = y[mask]

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

classifiers = [
    ("Logistic Regression", LogisticRegression(max_iter=200)),
    ("Decision Tree", DecisionTreeClassifier(max_depth=4)),
    ("SVM (RBF Kernel)", SVC(kernel="rbf", C=1.0, gamma='auto', probability=True))
]

fig = plt.figure(figsize=(18, 6))
n_grid = 30  # 网格分辨率

for idx, (name, model) in enumerate(classifiers):
    model.fit(X_train, y_train)
    ax = fig.add_subplot(1, 3, idx + 1, projection='3d')

    # 原始数据点
    ax.scatter(X[y == 0, 0], X[y == 0, 1], X[y == 0, 2], c='yellow', edgecolors='k', s=40, label='Class 0')
    ax.scatter(X[y == 1, 0], X[y == 1, 1], X[y == 1, 2], c='green', edgecolors='k', s=40, label='Class 1')

    # 构建三维网格
    x_lin = np.linspace(X[:, 0].min()-0.3, X[:, 0].max()+0.3, n_grid)
    y_lin = np.linspace(X[:, 1].min()-0.3, X[:, 1].max()+0.3, n_grid)
    z_lin = np.linspace(X[:, 2].min()-0.3, X[:, 2].max()+0.3, n_grid)
    xx, yy, zz = np.meshgrid(x_lin, y_lin, z_lin, indexing='ij')
    grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

    #预测概率
    probs = model.predict_proba(grid_points)[:, 1]
    prob_volume = probs.reshape(xx.shape)

    #marching_cubes算法，提取三维中的等值面并转换为实际坐标
    verts, faces, normals, values = marching_cubes(prob_volume, level=0.5,
                                               spacing=(x_lin[1]-x_lin[0], y_lin[1]-y_lin[0], z_lin[1]-z_lin[0]))
    #将所有顶点坐标平移到实际数据的起点
    verts[:, 0] += x_lin[0]
    verts[:, 1] += y_lin[0]
    verts[:, 2] += z_lin[0]

    #绘制三维决策边界
    mesh = Poly3DCollection(verts[faces], alpha=0.4)
    mesh.set_facecolor('cyan')
    ax.add_collection3d(mesh)

    ax.set_xlabel('Sepal Width')
    ax.set_ylabel('Petal Length')
    ax.set_zlabel('Petal Width')
    ax.set_title(name)
    ax.legend()
    
plt.subplots_adjust(wspace=0.3, hspace=0.4)
plt.show()