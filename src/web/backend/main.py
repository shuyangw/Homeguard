from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from pathlib import Path

# Add project root to sys.path to allow imports from src
import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.web.backend.api.router import router

app = FastAPI(title="Homeguard Backtesting API")

# Configure CORS
origins = [
    "http://localhost:5173",  # Vite dev server
    "http://127.0.0.1:5173",
    "*" # Development only
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("src.web.backend.main:app", host="0.0.0.0", port=8000, reload=True)
