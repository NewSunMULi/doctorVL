# 医学图像智能分析系统

## 项目概述
本项目是一个医学图像智能分析系统，结合了视觉分割和多模态大语言模型技术，能够对医学影像进行分割分析并生成智能报告。

**适合新手的特点：**
- 详细的安装步骤
- 简单的运行指南
- 清晰的项目结构说明
- 常见问题解决方案

## 核心功能
1. **医学图像分割**：使用 UNet 和 Sam3 模型对医学影像进行分割
2. **智能分析**：结合 Qwen3-VL 大语言模型生成分析报告
3. **数据处理**：支持 NIfTI 格式医学影像的加载和处理

## 项目文件结构

```
doctorVL/
├── dataset/              # 数据集
│   ├── image/           # 医学图像
│   └── llm/             # 大模型相关数据
├── model/               # 模型存储
│   ├── Qwen/            # Qwen3 模型
│   └── sam3/            # Sam3 模型
├── scripts/             # 核心源代码
│   ├── dataset/         # 数据集处理
│   ├── img/             # 图像处理
│   │   ├── sam3/        # Sam3 模型实现
│   │   └── unet/        # UNet 模型实现
│   ├── llm/             # 大语言模型
│   │   └── Qwen/        # Qwen3 模型实现
│   ├── example/         # 示例代码
│   ├── api.py           # API 接口
│   └── model.py         # 主模型类
├── README.md            # 本文档
└── requirements.txt     # 依赖列表
```

## 环境配置

### 1. 安装 Python
确保你的电脑上安装了 Python 3.8 或更高版本。

### 2. 安装依赖

1. 首先安装 PyTorch 和相关库：
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

2. 安装项目依赖：
```bash
pip3 install -r requirements.txt
```

## 快速开始

### 方法 1：直接运行示例

1. 进入 scripts 目录：
```bash
cd scripts
```

2. 运行主程序：
```bash
python model.py
```

### 方法 2：使用数据集进行训练

1. 准备数据集：
   - 将医学影像放入 `dataset/image/` 目录
   - 创建 `dataset/llm/qa_pairs.json` 文件，格式如下：
   ```json
   [
     {
       "image": "path/to/image.nii.gz",
       "label": "path/to/label.nii.gz",
       "message": "请分析这个医学影像"
     }
   ]
   ```

2. 修改 `model.py` 中的数据集路径：
```python
dataset = DoctorDataset(root="../", data_path="dataset/llm/qa_pairs.json")
```

3. 运行训练：
```bash
python model.py
```

## 如何修改项目

### 1. 修改模型参数

在 `model.py` 文件中，你可以修改以下参数：

- `num_epochs`：训练轮数
- `batch_size`：批次大小
- `learning_rate`：学习率

示例：
```python
model.train(
    train_dataset=dataset,
    num_epochs=5,      # 减少训练轮数
    batch_size=2,       # 减小批次大小
    learning_rate=1e-5  # 减小学习率
)
```

### 2. 添加新的图像处理功能

在 `scripts/img/` 目录下添加新的处理函数，例如：

```python
# 在 imgProcess.py 中添加
def new_processing_function(img):
    """新的图像处理函数"""
    # 处理逻辑
    return processed_img
```

### 3. 修改大语言模型提示

在 `scripts/llm/Qwen/model.py` 中修改提示模板：

```python
def get_input_img(self, img):
    inputs = self.processor(images=img, text="请详细分析这个医学影像的异常情况", return_tensors="pt")
    return inputs
```

## 常见问题

### 1. 运行时出现内存不足错误
**解决方案**：减小 `batch_size` 参数，例如设置为 1 或 2。

### 2. 模型加载失败
**解决方案**：确保模型文件存在于正确的路径，检查 `model_path` 参数。

### 3. 数据加载错误
**解决方案**：检查 JSON 文件格式是否正确，确保图像路径存在。

## 项目扩展

### 1. 添加新的模型

在 `scripts/img/` 或 `scripts/llm/` 目录下创建新的模型类，继承现有的模型基类。

### 2. 集成新的数据集

修改 `scripts/dataset/dataset.py` 文件，添加对新数据集格式的支持。

### 3. 开发 Web 界面

使用 FastAPI 或 Flask 创建 Web 接口，在 `scripts/api.py` 中实现。

## 技术支持

如果遇到问题，可以：
1. 检查代码中的注释
2. 查看示例代码
3. 参考 PyTorch 和 Transformers 官方文档