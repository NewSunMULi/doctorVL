from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

import uvicorn as uv

app = FastAPI(title="API", openapi_url="/deer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 请替换为你的前端实际地址和端口
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法，包括 OPTIONS
    allow_headers=["*"],  # 允许所有头部
)

if __name__ == "__main__":
    uv.run(app)