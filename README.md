# Iris_Classification_Visualization_python_program_study
# 项目简介

本项目基于 Iris 数据集，包含多种机器学习分类器的二维和三维可视化分析，主要文件如下：

## 文件结构

- [data_preview.py](e:\python_program_study\实验3\project3\data_preview.py)：数据预览与特征可视化，展示箱线图和交互式散点图。
- [classifier2d.py](e:\python_program_study\实验3\project3\classifier2d.py)：二维决策边界与分类概率可视化，支持逻辑回归、决策树和支持向量机。
- [3dprobmap.py](e:\python_program_study\实验3\project3\3dprobmap.py)：三维概率分布与决策边界投影，支持二分类。
- [3dboundary.py](e:\python_program_study\实验3\project3\3dboundary.py)：三维决策边界等值面提取与可视化，支持二分类。

## 环境依赖

请确保已安装以下 Python 库：

```sh
pip install numpy pandas matplotlib seaborn scikit-learn plotly scikit-image
```

## 使用方法

1. **数据预览**  
   运行 [data_preview.py](e:\python_program_study\实验3\project3\data_preview.py) 可查看 Iris 数据集的特征分布和交互式散点图。

2. **二维分类器可视化**  
   运行 [classifier2d.py](e:\python_program_study\实验3\project3\classifier2d.py) 可比较不同分类器的决策边界和类别概率分布。

3. **三维概率分布与边界**  
   运行 [3dprobmap.py](e:\python_program_study\实验3\project3\3dprobmap.py) 或 [3dboundary.py](e:\python_program_study\实验3\project3\3dboundary.py) 可进行三维空间的决策边界和概率分布可视化。

## 说明

- 所有脚本均基于 Iris 数据集，部分脚本对类别数做了合并或筛选。
- 可视化结果会自动弹出窗口显示，部分脚本支持交互式操作。
- 推荐使用 VSCode 编辑器运行和调试。

---

如需进一步扩展或自定义特征、分类器，请参考各脚本内注释。
