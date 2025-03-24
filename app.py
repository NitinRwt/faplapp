from fastapi import FastAPI, File, UploadFile, Form, HTTPException
import cv2
import numpy as np
import os
from ultralytics import YOLO
from PIL import Image
import io
from ocr import detect_and_crop as ocr_detect_and_crop, detect_final_classes
from analog import detect_and_crop as analog_detect_and_crop, get_meter_reading

app = FastAPI()

# Load model paths dynamically from environment variables or use defaults
ocr_model_box_path = os.getenv("OCR_MODEL_BOX", "Models/res_temp_box.pt")
ocr_model_ocr_path = os.getenv("OCR_MODEL_OCR", "Models/res_temp_ocr.pt")
analog_model_box_path = os.getenv("ANALOG_MODEL_BOX", "Models/analog_box.pt")
analog_model_best_path = os.getenv("ANALOG_MODEL_BEST", "Models/best.pt")

# Load detection models
try:
    # OCR models for CR_TEST
    model_1 = YOLO(ocr_model_box_path)
    model_2 = YOLO(ocr_model_ocr_path)
    # Analog models for DC_TEST
    model_3 = YOLO(analog_model_box_path)
    model_4 = YOLO(analog_model_best_path)
except Exception as e:
    print(f"Error loading models: {str(e)}")
    raise

def process_analog_meter(file_bytes):
    """Process an analog meter image (DC_TEST) and return structured response."""
    try:
        # Convert file bytes to a cv2 image
        nparr = np.frombuffer(file_bytes, np.uint8)
        image_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image_cv is None:
            raise HTTPException(status_code=400, detail="Invalid image data for analog processing")
            
        # Crop meter region using analog detection (note: analog_detect_and_crop now accepts an image array)
        cropped_image, error = analog_detect_and_crop(image_cv, model_3)
        if error:
            raise HTTPException(status_code=400, detail=error)
        
        meter_reading = get_meter_reading(cropped_image, model_4)
        if isinstance(meter_reading, str):
            raise HTTPException(status_code=400, detail=meter_reading)
        
        return {
            "overall_confidence_score": 0.90,
            "kv": [
                {
                    "keyName": "meter_reading_kV",
                    "keyValue": str(meter_reading),
                    "conf": 0.90
                }
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing analog meter: {str(e)}")

def process_res_temp(file_bytes):
    """Process an OCR image (CR_TEST) and return structured response."""
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        cropped_regions = ocr_detect_and_crop(model_1, image)
        if not cropped_regions:
            raise HTTPException(status_code=400, detail="No regions detected in the image")
        
        final_classes_dict = detect_final_classes(model_2, cropped_regions)
        if not final_classes_dict:
            raise HTTPException(status_code=400, detail="No classes detected in the image")

        kv_list = []
        for class_name, values in final_classes_dict.items():
            detected_value = "".join(values)
            kv_list.append({
                "keyName": class_name,
                "keyValue": detected_value,
                "conf": 0.85  # Adjust confidence as needed
            })

        return {
            "overall_confidence_score": 0.85,
            "kv": kv_list
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing OCR data: {str(e)}")

@app.post("/detect/")
async def detect(
    file: UploadFile = File(...),
    test_type: str = Form(...)  # "CR_TEST" for OCR; "DC_TEST" for analog meter processing.
):
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload PNG or JPEG images")

    file_bytes = await file.read()

    if test_type == "CR_TEST":
        return process_res_temp(file_bytes)
    elif test_type == "DC_TEST":
        return process_analog_meter(file_bytes)
    else:
        raise HTTPException(status_code=400, detail="Invalid test_type. Choose 'CR_TEST' or 'DC_TEST'.")

@app.get("/")
def health_check():
    return {"status": "healthy", "version": "2.0"}
