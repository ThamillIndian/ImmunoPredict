from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import router, init_pipeline
from .database import init_db
import os

app = FastAPI(title="ImmunoPredict API", version="1.0.0")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup event
@app.on_event("startup")
def startup_event():
    # 1. Initialize Database
    init_db()
    
    # 2. Initialize Hybrid Pipeline
    # Using relative paths from project root
    config_path = 'backend/config.yaml'
    model_path = 'backend/artifacts/stage2/encoder_best.pth'
    
    if os.path.exists(config_path) and os.path.exists(model_path):
        init_pipeline(config_path, model_path)
        print("Hybrid Model Pipeline initialized successfully.")
    else:
        print(f"Warning: Model or Config missing. Prediction endpoint will fail.")
        print(f"Config: {os.path.exists(config_path)}, Model: {os.path.exists(model_path)}")

@app.get("/")
def read_root():
    return {"status": "online", "message": "ImmunoPredict API is running"}

app.include_router(router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
