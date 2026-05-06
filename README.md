# Machine Vision Experiments

华南农业大学 图像处理与机器视觉课程实验代码

---

## 目录结构

```
Machine-Version/
├── week1/          # 实验一：空间域点运算
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

---

> 运行环境：Python 3.x，conda 环境 `pytorch`
> 依赖：opencv-python / numpy / matplotlib / scikit-image / python-docx
