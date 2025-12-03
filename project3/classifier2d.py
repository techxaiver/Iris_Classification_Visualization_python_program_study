from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier  # 新增：决策树
from sklearn.svm import SVC                      # 新增：支持向量机
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 加载Iris数据集
iris = load_iris()
X = iris.data[:, 2:]  # 选择后两个特征进行可视化
y = iris.target

# 划分数据集，70%为训练集，30%为测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#定义多个分类器
classifiers = [
    ("Logistic Regression", LogisticRegression(max_iter=200)),
    ("Decision Tree", DecisionTreeClassifier(max_depth=4)),
    ("SVM (RBF Kernel)", SVC(kernel="rbf", C=1.0, gamma='auto',probability=True))
]

# 可视化决策边界
xx, yy = np.meshgrid(np.arange(X[:, 0].min() - 1, X[:, 0].max() + 1, 0.1),
                     np.arange(X[:, 1].min() - 1, X[:, 1].max() + 1, 0.1))
#生成二维网格,稍稍延伸边界，xx存放第一个特征值，yy存放第二个


# 设置每个类别的固定颜色
class_colors = ['yellow', 'green', 'blue']  # 自定义颜色：黄、绿、蓝

#每一行对应一个分类器
fig, axs = plt.subplots(3, 4, figsize=(20, 16))

for row,(name,model) in enumerate(classifiers):#根据每个分类器绘图
    model.fit(X_train,y_train)
    # **1. 整体决策边界图**
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])#对每个点进行预测，输出为每个点属于的类型
    Z = Z.reshape(xx.shape)#根据xx的形状变化

    # 绘制决策边界
    ax = axs[row, 0]
    ax.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.colors.ListedColormap(class_colors))
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=50, cmap=mcolors.ListedColormap(class_colors))
    ax.set_ylabel(f'{name}\nPepal Width', fontsize=12, fontweight='bold')
    if row == 0: ax.set_title('Decision Boundaries')
    if row == 2: ax.set_xlabel('Petal Length')

     # **2. 计算每个类别的概率**
    probs = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])  # 计算概率
    probs = probs.reshape(xx.shape[0], xx.shape[1], 3)  # 重塑为 (网格行, 网格列, 类别数)

    # **每一类分类器的概率图**
    for i in range(3):  # i corresponds to each class
        ax = axs[row,i + 1]  # Use axs[1], axs[2], axs[3] for individual class probabilities
        class_prob=probs[:,:,i]

        # 对每个类别绘制概率图，并设置渐变色
        cmap = mcolors.LinearSegmentedColormap.from_list(
            f'class_{i}_colormap', ['white', class_colors[i]], N=256)
        contour = ax.contourf(xx, yy, class_prob, alpha=0.7, cmap=cmap)
    
        # 画数据点，按照预测的类别显示
        ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=50, cmap=mcolors.ListedColormap(class_colors), alpha=1)
    
        # 添加color bar
        fig.colorbar(contour, ax=ax)
    
        # 设置标题和标签
        ax.set_title(f'Class {i} Probability')
        ax.set_xlabel('Petal Length')
        ax.set_ylabel('Pepal Width')

# 调整布局
plt.subplots_adjust(wspace=0.3, hspace=0.4)  # 增加子图之间的水平和垂直间距
plt.show()

