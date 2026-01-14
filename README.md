# 项目使用
在Pycharm新建项目后，将该项目压缩包解压至你刚建好的项目文件夹下，并创建一个空文件夹model<br>

项目结构:
```
.venv
.idea
dataset
    你的数据集
model
    你的模型文件
output
    你的输出文件
scripts
    api_model.py
    example.py
    example2.py
    main.py
    model.py
    model_download.py
```

# 依赖安装
使用 pip 安装相应依赖:
```shell
pip install -r requirements.txt
```

# 模型下载
运行model_download.py文件
```shell
cd scripts
python model_download.py
```

# 模型调试
模型测试请在model.py文件下的特定部分测试
```python
if __name__ == "__main__":
    # 此处可编写调试代码
    pass
```

# 启动后端服务
运行main.py文件
```shell
cd ./scripts
python main.py
```
打开API文档，测试接口功能：
http://127.0.0.1:8000/docs

# 注意事项：
## 1. 模型输出为流式输出（和豆包一样），但无法在文档http://127.0.0.1:8000/docs中体现，需自行使用requests库实现
## 2. 模型在有图片输入的情况下大约3-5分钟出结果，反正只需1-2min（RTX 4060），CPU时间x2