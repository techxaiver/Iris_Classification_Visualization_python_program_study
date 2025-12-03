from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors

# 1. 准备数据 (使用3个特征)
iris = load_iris()
# 特征索引: 0:Sepal Length, 2:Petal Length, 3:Petal Width
feature_indices = [0, 2, 3] 
feature_names = [iris.feature_names[i] for i in feature_indices]
X = iris.data[:, feature_indices] 
y = iris.target

# 将原始的 3 类数据合并为 2 类
y = np.where(y == 0, 0, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. 定义分类器
classifiers = [
    ("Logistic Regression", LogisticRegression(max_iter=200)),
    ("Decision Tree", DecisionTreeClassifier(max_depth=4)),
    ("SVM (RBF Kernel)", SVC(kernel="rbf", C=1.0, gamma='auto', probability=True))
]

# 计算各轴的范围
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
z_min, z_max = X[:, 2].min() - 1, X[:, 2].max() + 1

# 计算各轴的固定值（用于切片，这里取均值）
x_mean = X[:, 0].mean()
y_mean = X[:, 1].mean()
z_mean = X[:, 2].mean()

# 3. 开始绘图
fig = plt.figure(figsize=(20, 7))
colors = ['#0000FF', '#FF0000']  # 红色和蓝色对应两类

for i, (name, model) in enumerate(classifiers):
    print(f"Plotting {name}...")
    model.fit(X_train, y_train)
    ax = fig.add_subplot(1, 3, i + 1, projection='3d')
    # --- A. 绘制三个投影平面 (墙壁上的概率图) ---
    
    # 1. XY 平面 (Z轴底部)

    #生成二维网格坐标点
    xx_xy, yy_xy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
    
    #构造用于模型预测的输入数据x,y,z_mean
    input_xy = np.c_[xx_xy.ravel(), yy_xy.ravel(), np.full(xx_xy.size, z_mean)]

    #预测输入数据为类别1的概率，并恢复为何输入数据一样的二维数组
    probs_xy = model.predict_proba(input_xy)[:, 1].reshape(xx_xy.shape)

    #在XY平面绘制等高线填充图，投影到Z轴，填充每个网格点属于类别1的概率，剩余部分即为类别0
    ax.contourf(xx_xy, yy_xy, probs_xy, zdir='z', offset=z_min, cmap='coolwarm', alpha=0.4)

    # 2. XZ 平面 (Y轴背部)
    xx_xz, zz_xz = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(z_min, z_max, 50))
    input_xz = np.c_[xx_xz.ravel(), np.full(xx_xz.size, y_mean), zz_xz.ravel()]
    probs_xz = model.predict_proba(input_xz)[:, 1].reshape(xx_xz.shape)
    ax.contourf(xx_xz, probs_xz, zz_xz, zdir='y', offset=y_max, cmap='coolwarm', alpha=0.4)

    # 3. YZ 平面 (X轴左侧)
    yy_yz, zz_yz = np.meshgrid(np.linspace(y_min, y_max, 50), np.linspace(z_min, z_max, 50))
    input_yz = np.c_[np.full(yy_yz.size, x_mean), yy_yz.ravel(), zz_yz.ravel()]
    probs_yz = model.predict_proba(input_yz)[:, 1].reshape(yy_yz.shape)
    ax.contourf(probs_yz, yy_yz, zz_yz, zdir='x', offset=x_min, cmap='coolwarm', alpha=0.4)

    # --- B. 合成三维决策曲面 (空间内部的实体) ---
    
    # 生成密集的 3D 网格
    n_grid = 30 # 网格密度，越高越精细但计算越慢
    gx, gy, gz = np.meshgrid(np.linspace(x_min, x_max, n_grid),
                             np.linspace(y_min, y_max, n_grid),
                             np.linspace(z_min, z_max, n_grid))
    
    # 展平并预测整个空间的概率
    grid_points = np.c_[gx.ravel(), gy.ravel(), gz.ravel()]
    probs_vol = model.predict_proba(grid_points)[:, 1]
    
    # 核心逻辑：提取概率接近 0.5 的点 (即决策边界)
    # 0.05 是容差，决定了"曲面"的厚度
    boundary_mask = np.abs(probs_vol - 0.5) < 0.05
    
    if np.any(boundary_mask):
        ax.scatter(grid_points[boundary_mask, 0], 
                   grid_points[boundary_mask, 1], 
                   grid_points[boundary_mask, 2], 
                   c='black',      # 黑色表示边界
                   s=2,            # 点的大小
                   alpha=0.15,     # 透明度，越低越像烟雾/曲面
                   marker='o', 
                   zorder=1)

    # --- C. 绘制原始数据散点 ---
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=mcolors.ListedColormap(colors), s=30, edgecolors='k', alpha=1.0, zorder=10)

    # 设置标签和范围
    ax.set_title(name, fontsize=16, fontweight='bold')
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_zlabel(feature_names[2])
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    
    # 调整视角以便观察内部结构
    ax.view_init(elev=25, azim=-45)

plt.subplots_adjust(wspace=0.13, left=0.0, right=0.9, bottom=0.1, top=0.9)
plt.show()