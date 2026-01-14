#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen3-VL-2B-Instruct', local_dir="../model/Qwen/Qwen3-VL-2B-Instruct")