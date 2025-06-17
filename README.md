# 基于PyTorch的手写数字识别系统

## 项目说明
本项目基于PyTorch实现MNIST手写数字识别，包含完整的训练、测试和部署流程。支持CPU/GPU加速，提供图形界面交互。

### 模型架构
- 使用两层卷积网络（Conv2d + ReLU + MaxPool）
- 输出维度：`[batch_size, 10]`（对应数字0-9）
- 输入要求：灰度图尺寸 `[1, 28, 28]`（单通道）

### 技术规范
- Python 3.8+
- PyTorch 1.10+
- torchvision 0.11+

### 数据预处理
1. 图像需转换为浮点类型（`.float()`）
2. 归一化到 `[0, 1]` 范围
3. 三通道图像需转为灰度图（推荐使用 `transforms.Grayscale`）

### 常见问题
**错误1**: `Input type (unsigned char) and bias type (float) should be the same`
> **解决**: 确保输入张量已通过 `.float()` 转换

**错误2**: `mat1 and mat2 shapes cannot be multiplied`
> **解决**: 检查模型中的展平层（`x.view(...)`）输出尺寸是否匹配
