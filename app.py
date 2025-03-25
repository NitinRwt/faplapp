from fastapi import FastAPI, File, UploadFile, Form, HTTPException
import cv2
import numpy as np
import os
from ultralytics import YOLO
from PIL import Image
import io
import easyocr
from ocr import detect_and_crop as ocr_detect_and_crop, detect_final_classes
from analog import detect_and_crop as analog_detect_and_crop, get_meter_reading
from Remaining_test import draw_obb

app = FastAPI()

try:
    res_temp_box = YOLO("Models/res_temp_box.pt")
    res_temp_ocr = YOLO("Models/res_temp_ocr.pt")
    analog_box = YOLO("Models/analog_box_v2.pt")
    analog_reading = YOLO("Models/analog_reading.pt")
    remaining_test_model = YOLO("Models/Remaining_tests_model.pt")
except Exception as e:
    print(f"Error loading models: {str(e)}")
    raise

reader = easyocr.Reader(['en'])

def process_analog_meter(file_bytes):
    try:
        image_cv = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image_cv is None:
            raise HTTPException(status_code=400, detail="Invalid image data for analog processing")
        cropped_image, error = analog_detect_and_crop(image_cv, analog_box)
        if error:
            raise HTTPException(status_code=400, detail=error)
        meter_reading = get_meter_reading(cropped_image, analog_reading)
        return {"overall_confidence_score": 0.90, "DC_TEST": [{"keyName": "meter_reading_kV", "keyValue": str(meter_reading), "conf": 0.90}]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing analog meter: {str(e)}")

def process_res_temp(file_bytes):
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        cropped_regions = ocr_detect_and_crop(res_temp_box, image)
        final_classes_dict = detect_final_classes(res_temp_ocr, cropped_regions)
        kv_list = [{"keyName": k, "keyValue": "".join(v), "conf": 0.85} for k, v in final_classes_dict.items()]
        return {"overall_confidence_score": 0.85, "CR_TEST": kv_list}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing OCR data: {str(e)}")

def process_remaining_test(file_bytes, expected_classes):
    try:
        image_cv = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        results = remaining_test_model(image_cv)
        detected_classes = []
        extracted_texts = []
        for r in results:
            if r.obb is not None:
                image_cv, texts = draw_obb(image_cv, r.obb)
                extracted_texts.extend(texts)
                for i, class_id in enumerate(r.obb.cls.cpu().numpy()):
                    detected_classes.append(r.names[int(class_id)])
        
        # Check if at least one expected class is found
        matching_classes = [cls for cls in detected_classes if cls in expected_classes]
        if not matching_classes:
            raise HTTPException(status_code=400, detail=f"No required classes detected. Expected one of: {expected_classes}, but got: {detected_classes}")
        
        # Create key-value pairs for matching classes
        kv_list = []
        text_index = 0
        for cls in detected_classes:
            if cls in expected_classes:
                text_value = extracted_texts[text_index] if text_index < len(extracted_texts) else "N/A"
                kv_list.append({"keyName": cls, "keyValue": text_value, "conf": 0.85})
                text_index += 1
        
        return {"overall_confidence_score": 0.85, "Result": kv_list}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing test: {str(e)}")

@app.post("/detect/")
async def detect(file: UploadFile = File(...), test_type: str = Form(...)):
    file_bytes = await file.read()
    if test_type == "CR_TEST":
        return process_res_temp(file_bytes)
    elif test_type == "DC_TEST":
        return process_analog_meter(file_bytes)
    elif test_type == "PD_TEST":
        return process_remaining_test(file_bytes, expected_classes=["UVolt", "qCValue"])
    elif test_type == "HV_TEST":
        return process_remaining_test(file_bytes, expected_classes=["kV", "TimeLeft", "q(IEC) value"])
    else:
        raise HTTPException(status_code=400, detail="Invalid test_type. Choose 'CR_TEST', 'DC_TEST', 'PD_TEST', or 'HV_TEST'")

@app.get("/")
def health_check():
    return {"status": "healthy", "version": "v1.6"}
