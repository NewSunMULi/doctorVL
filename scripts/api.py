from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from PIL import Image
import io
import base64
import numpy as np
import torch
import torch.nn.functional as F
from model import DoctorVL

# 初始化 FastAPI 应用
app = FastAPI(title="DoctorVL API", description="医学影像处理与分析 API")

# 加载模型
model = DoctorVL("../model")

@app.post("/segment")
async def segment_image(
    file: UploadFile = File(...),
    text: str = Form("tumor")
):
    """
    对医学影像进行分割
    
    Args:
        file: 上传的图像文件
        text: 文本提示，用于引导分割
        
    Returns:
        分割结果的 JSON 响应
    """
    try:
        # 读取图像
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        
        # 进行分割
        mask = model.segment(img, text)
        
        # 处理分割结果
        # 调整大小并阈值化
        pred_masks = F.interpolate(mask, size=img.size[::-1], mode='bilinear', align_corners=False)
        pred_masks = torch.sigmoid(pred_masks) > 0.5
        pred_masks = pred_masks.float()
        
        # 转换为 numpy 数组
        mask_np = pred_masks[0][0].numpy()
        
        # 将掩码转换为 base64 编码
        mask_img = Image.fromarray((mask_np * 255).astype(np.uint8))
        buffer = io.BytesIO()
        mask_img.save(buffer, format="PNG")
        mask_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return JSONResponse({
            "status": "success",
            "mask": mask_base64
        })
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.post("/generate")
async def generate_text(
    messages: list = Form(...)
):
    """
    生成文本回答
    
    Args:
        messages: 输入消息列表
        
    Returns:
        生成的文本回答
    """
    try:
        # 生成回答
        response = model.generate(messages)
        
        return JSONResponse({
            "status": "success",
            "response": response
        })
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    question: str = Form(...),
    text: str = Form("tumor")
):
    """
    分析医学影像并回答问题
    
    Args:
        file: 上传的图像文件
        question: 问题文本
        text: 文本提示，用于引导分割
        
    Returns:
        分析结果的 JSON 响应
    """
    try:
        # 读取图像
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        
        # 分析图像
        response = model.analyze(img, question, text)
        
        return JSONResponse({
            "status": "success",
            "response": response
        })
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.get("/")
async def root():
    """
    根路径
    
    Returns:
        API 信息
    """
    return {
        "message": "Welcome to DoctorVL API",
        "endpoints": {
            "/segment": "分割医学影像",
            "/generate": "生成文本回答",
            "/analyze": "分析医学影像并回答问题"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
