"""
模型文件
"""
from typing import List
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from .model import QWen3Doctor

class Message(BaseModel):
    text: str
    img: List[str] = []
    size: int = 256

def load_model(path: str = ""):
    return QWen3Doctor(path)

model = APIRouter()

model_qw = load_model("./model/Qwen/Qwen3-VL-2B-Instruct") # 模型文件路径

def stream_output(message: List[dict], num: int = 256):
    stream = model_qw.get_text_stream()
    model_qw(message, stream, num)
    for msg in stream:
        yield f"{msg}"

@model.post("/output", description="发送输入，获取模型输出")
async def output(content: Message):
    content = content.model_dump()
    msg = [{
        "role": "user",
        "content": [
            {
                "type": "text", "text": content["text"],
            }
        ]
    }]
    if len(content['img']) > 0:
        for i in content['img']:
            msg[0]['content'].append({
                "type": "Image",
                "Image": i
            })
    return StreamingResponse(
        stream_output(msg, content["size"]),
        media_type="text/event-stream",
    )

@model.get("/device", description="获取模型所在的设备")
async def device():
    return {
        "device": model_qw.get_device(),
    }
