# DoctorVL: 医学影像处理与分析系统

## 项目简介

DoctorVL 是一个集成了先进深度学习模型的医学影像处理与分析系统，主要功能包括：

- **医学影像分割**：使用 Sam3 模型对医学影像进行精确分割
- **文本生成与问答**：使用 Qwen3 大语言模型处理医学相关问题
- **医学影像分析**：结合分割结果和语言模型，对医学影像进行综合分析并回答问题

## 项目结构

```
doctorVL/
├── dataset/          # 数据集目录
│   ├── image/        # 医学影像数据
│   └── llm/          # 语言模型训练数据
├── img/              # 示例图像
├── log/              # 日志文件
├── model/            # 模型存储目录
│   ├── Qwen/         # Qwen3 模型
│   ├── sam3/         # Sam3 模型
│   └── sam3_lora/    # Sam3 LoRA 适配器
├── output/           # 输出目录
├── scripts/          # 脚本目录
│   ├── dataset/      # 数据集处理脚本
│   ├── example/      # 示例脚本
│   ├── img/          # 图像处理脚本
│   └── llm/          # 语言模型脚本
├── LICENSE           # 许可证文件
├── README.md         # 项目说明文档
├── docker-compose.yml # Docker 配置文件
├── main.py           # FastAPI 应用入口
├── model_download.py # 模型下载脚本
└── requirements.txt  # 依赖库文件
```

## 安装指南

### 1. 克隆项目

```bash
git clone <项目仓库地址>
cd doctorVL
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 下载模型

运行模型下载脚本以获取所需的预训练模型：

```bash
python model_download.py
```

## 快速开始

### 启动 API 服务

```bash
python main.py
```

服务将在默认端口（通常是 8000）上运行，您可以通过访问 `http://localhost:8000` 来查看 API 文档。

## API 文档

### 1. 分割医学影像

**端点**: `/segment`

**方法**: `POST`

**参数**:
- `file`: 上传的图像文件
- `text`: 文本提示，用于引导分割（默认值："tumor"）

**返回**:
- `status`: 操作状态
- `mask`: 分割结果的 base64 编码

### 2. 生成文本回答

**端点**: `/generate`

**方法**: `POST`

**参数**:
- `messages`: 输入消息列表

**返回**:
- `status`: 操作状态
- `response`: 生成的文本回答

### 3. 分析医学影像并回答问题

**端点**: `/analyze`

**方法**: `POST`

**参数**:
- `file`: 上传的图像文件
- `question`: 问题文本
- `text`: 文本提示，用于引导分割（默认值："tumor"）

**返回**:
- `status`: 操作状态
- `response`: 分析结果的文本回答

## 核心功能

### 1. 医学影像分割

使用 Sam3 模型对医学影像进行分割，支持通过文本提示引导分割过程，可用于肿瘤、器官等医学结构的识别和标注。

### 2. 医学影像分析

结合分割结果和 Qwen3 大语言模型，对医学影像进行综合分析，能够回答与影像相关的问题，如病变位置、大小、性质等。

### 3. 模型训练

支持对 Sam3 和 Qwen3 模型进行微调，以适应特定的医学影像数据集。

## 技术栈

- **Python 3.10+**
- **FastAPI** - API 框架
- **PyTorch** - 深度学习框架
- **Transformers** - 预训练模型库
- **ModelScope** - 模型管理平台
- **PEFT** - 参数高效微调
- **NumPy, Pillow** - 数据处理库

## 示例使用

### 分割医学影像

```python
import requests
from PIL import Image
import io

# 读取图像
img = Image.open("path/to/image.jpg")

# 准备请求数据
buffer = io.BytesIO()
img.save(buffer, format="JPEG")
buffer.seek(0)

# 发送请求
response = requests.post(
    "http://localhost:8000/segment",
    files={"file": buffer},
    data={"text": "tumor"}
)

# 处理响应
result = response.json()
if result["status"] == "success":
    # 处理分割结果
    pass
```

### 分析医学影像

```python
import requests
from PIL import Image
import io

# 读取图像
img = Image.open("path/to/image.jpg")

# 准备请求数据
buffer = io.BytesIO()
img.save(buffer, format="JPEG")
buffer.seek(0)

# 发送请求
response = requests.post(
    "http://localhost:8000/analyze",
    files={"file": buffer},
    data={
        "question": "这个影像中是否有肿瘤？如果有，它的位置在哪里？",
        "text": "tumor"
    }
)

# 处理响应
result = response.json()
if result["status"] == "success":
    print(result["response"])
```

## 模型说明

### Sam3 模型

Sam3 是一个先进的医学影像分割模型，能够根据文本提示对医学影像进行精确分割。项目中使用了预训练的 Sam3 模型，并支持通过 LoRA 适配器进行微调。

### Qwen3 模型

Qwen3 是一个功能强大的大语言模型，能够理解医学相关的问题并生成准确的回答。项目中使用了 Qwen3-VL-2B-Instruct 版本，支持处理图像和文本输入。

## 许可证

本项目采用 MIT 许可证，详见 LICENSE 文件。

## 联系方式

如有任何问题或建议，请通过以下方式联系我们：

- 电子邮件：<contact@example.com>
- GitHub：<https://github.com/example/doctorVL>
