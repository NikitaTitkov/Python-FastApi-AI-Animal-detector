from fastapi import FastAPI, UploadFile, HTTPException, status
from fastapi.responses import FileResponse
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO
import shutil
import uvicorn
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

repo_id = "AlBeBack/animals_detection_yolov8"
filename = "yolov8_trained_v2.pt"

weights_path = hf_hub_download(repo_id=repo_id, filename=filename)

UPLOAD_DIR = Path("./uploads")
PROCESSED_DIR = Path("./processed")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def draw_bbox_on_image(image_path, model, confidence_threshold=0.5):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = model(image_path)

    boxes = results[0].boxes
    classes = []
    
    for box in boxes:
        x_min, y_min, x_max, y_max = box.xyxy[0]
        confidence = box.conf[0]
        class_id = int(box.cls[0])

        if confidence > confidence_threshold: 
            classes.append(class_id)
            color = (0, 255, 0) if class_id == 1 else (0, 0, 255)
            cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)

            label = f"Class: {class_id}, Conf: {confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, label, (int(x_min), int(y_min) - 10), font, 0.9, color, 2, cv2.LINE_AA)

    processed_image_path = PROCESSED_DIR / f"processed_{Path(image_path).name}"
    cv2.imwrite(str(processed_image_path), image)
    
    return processed_image_path, classes


@app.post("/upload")
async def upload_file(file: UploadFile):
    upload_path = UPLOAD_DIR / file.filename
    with upload_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    model = YOLO(weights_path)

    # Обработка изображения с рамками
    processed_image_path, classes = draw_bbox_on_image(str(upload_path), model)

    # Классификация по найденным классам
    if any(c == 1 for c in classes):  
        return FileResponse(processed_image_path, media_type="image/jpeg", filename=file.filename, status_code=status.HTTP_200_OK)
    elif classes:
        return FileResponse(processed_image_path, media_type="image/jpeg", filename=file.filename, status_code=status.HTTP_201_CREATED)
    else:
        return FileResponse(processed_image_path, media_type="image/jpeg", filename=file.filename, status_code=status.HTTP_202_ACCEPTED)

if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)

