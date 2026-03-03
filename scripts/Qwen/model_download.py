#模型下载
from modelscope import snapshot_download

if __name__ == '__main__':
    model_dir = snapshot_download('Qwen/Qwen3-VL-2B-Instruct', local_dir="../model/qwen/Qwen3-VL-2B-Instruct")