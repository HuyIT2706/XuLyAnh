from fastapi import FastAPI, UploadFile, File
from backend.predictor import SkinDetector
from backend.schemas import PredictionResult
from typing import List
import torch

app = FastAPI(title="Skin Disease Detection API")

# Khởi tạo detector (nên để global để tránh load model nhiều lần)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
detector = SkinDetector(model_path="models/best_model.pth", device=device)

@app.post("/predict", response_model=List[PredictionResult])
async def predict_api(file: UploadFile = File(...)):
    image_bytes = await file.read()
    predictions = detector.predict(image_bytes)
    return predictions

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)