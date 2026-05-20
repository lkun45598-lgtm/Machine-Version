# Machine Vision Experiments

华南农业大学 图像处理与机器视觉课程实验代码

---

## 目录结构

```
Machine-Version/
├── week1/          # 实验一：空间域点运算
├── week2/          # 实验二：空间域邻域滤波
├── week3/          # 实验三：频率域滤波
├── week4/          # 实验四：形态学处理
└── ...
```

每周实验内部按模块组织：

```
weekN/
├── src/            # 源代码（含 tests）
├── data/           # 输入图像与处理结果
├── report/         # 实验报告（md + docx）
└── reference/      # 课件、模板等参考资料
```

## 各周实验

| 周次 | 实验内容 | 主要方法 |
|---|---|---|
| Week 1 | 空间域点运算 | 灰度变换 / 灰度反转 / 阈值化（固定 + Otsu）/ 直方图均衡化 / 伪彩色映射 |
| Week 2 | 空间域邻域滤波 | 均值 / 中值 / 高斯滤波；Prewitt / Sobel / Laplacian / Canny 边缘检测；图像代数运算 |
| Week 3 | 频率域滤波 | 二维 FFT 幅度／相位谱；矩形高通/低通蒙版；空域 vs 频域对比；理想 / Butterworth / Gaussian 频域滤波 |
| Week 4 | 形态学处理 | 腐蚀 / 膨胀 / 开 / 闭；轮廓提取；形态学梯度 / 顶帽 / 黑帽；击中击不中变换 |

---

> 运行环境：Python 3.x，conda 环境 `pytorch`
> 依赖：opencv-python / numpy / matplotlib / scikit-image / python-docx
