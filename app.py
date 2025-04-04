from fastapi import FastAPI, File, UploadFile, Form, HTTPException
import cv2
import numpy as np
import os
from ultralytics import YOLO
from PIL import Image
import io
import easyocr
from ocr import detect_and_crop as ocr_detect_and_crop, detect_final_classes
from Remaining_test import draw_obb  
from analog import crop_region, calculate_meter_reading, get_center_point

app = FastAPI()

try:
    res_temp_box = YOLO("Models/res_temp_box.pt")
    res_temp_ocr = YOLO("Models/res_temp_ocr.pt")
    analog_box = YOLO("Models/analog_box_v2.pt")
    analog_reading = YOLO("Models/analog_reading_v2.pt")
    remaining_test_model = YOLO("Models/Remaining_tests_model.pt")
    new_apparatus_model = YOLO("Models/New_Apparatus_model.pt")
except Exception as e:
    print(f"Error loading models: {str(e)}")
    raise

reader = easyocr.Reader(['en'])


def process_res_temp(file_bytes):
    try:
        # Try to process using both models and select the best result
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        
        # For OCR model processing (original res_temp approach)
        cropped_regions = ocr_detect_and_crop(res_temp_box, image)
        final_classes_dict = detect_final_classes(res_temp_ocr, cropped_regions)
        
        # Convert image for apparatus model
        image_cv = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        # Process with new apparatus model
        apparatus_results = new_apparatus_model(image_cv)
        apparatus_data = {}
        
        # Extract text using apparatus model
        for r in apparatus_results:
            if r.obb is not None:
                _, extracted_texts = draw_obb(image_cv.copy(), r.obb)
                for i, class_id in enumerate(r.obb.cls.cpu().numpy()):
                    class_name = r.names[int(class_id)]
                    if i < len(extracted_texts) and extracted_texts[i]:
                        apparatus_data[class_name] = extracted_texts[i]
        
        # Combine results from both models
        final_data = {**final_classes_dict, **apparatus_data}
        
        # Convert to key-value list format
        kv_list = [{"keyName": k, "keyValue": "".join(v) if isinstance(v, list) else v, 
                    "actualValue": "".join(v) if isinstance(v, list) else v, 
                    "confidenceScore": 0.85} for k, v in final_data.items()]
        
        return {"ocs": 0.85, "extractions": kv_list}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing data: {str(e)}")

def process_remaining_test(file_bytes, expected_classes):
    try:
        image_cv = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image_cv is None:
            raise HTTPException(status_code=400, detail="Invalid image data for processing")
        
        # Run inference using the remaining tests model
        results = remaining_test_model(image_cv)
        
        extracted_data = {}
        confidence_score = 0.85
        
        # Process results and extract text from detected regions
        for r in results:
            if r.obb is not None:
                # Use the draw_obb function from Remaining_test.py to extract text
                _, extracted_texts = draw_obb(image_cv.copy(), r.obb)
                
                # Match the extracted texts with their class names
                for i, class_id in enumerate(r.obb.cls.cpu().numpy()):
                    class_name = r.names[int(class_id)]
                    
                    # Only process classes that we expect for this test type
                    if class_name in expected_classes and i < len(extracted_texts) and extracted_texts[i]:
                        # Store the detected text with its class name
                        extracted_data[class_name] = extracted_texts[i]
        
        # Format response
        kv_list = [{"keyName": k, "keyValue": v, "actualValue": v, "confidenceScore": confidence_score} 
                  for k, v in extracted_data.items()]
        
        # Determine test type based on expected classes
        test_type = "extractions" if "UVolt" in expected_classes else "extractions"
        
        # If no data was extracted
        if not kv_list:
            raise HTTPException(status_code=400, detail=f"No data extracted for the expected classes: {expected_classes}")
            
        return {"ocs": confidence_score, test_type: kv_list}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing test data: {str(e)}")

def process_analog_meter(file_bytes):
    try:
        # Convert bytes to OpenCV image
        image_cv = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image_cv is None:
            raise HTTPException(status_code=400, detail="Invalid image data for analog meter processing")
        
        # Detect the meter using analog_box model
        results = analog_box(image_cv)
        
        cropped_meter = None
        for r in results:
            if hasattr(r, "obb") and r.obb is not None:
                cropped_meter = crop_region(image_cv, r.obb)
                if cropped_meter is not None:
                    break
        
        if cropped_meter is None:
            raise HTTPException(status_code=400, detail="No analog meter detected in image")
        
        # Process the cropped meter image to read the value
        meter_results = analog_reading(cropped_meter)
        
        needle_corners = None
        number_positions = []
        
        # Process detection results
        for r in meter_results:
            if hasattr(r, "obb") and r.obb is not None:
                boxes = r.obb.xyxyxyxy.cpu().numpy()
                classes = r.obb.cls.cpu().numpy()
                
                for box, class_id in zip(boxes, classes):
                    class_name = r.names[int(class_id)]
                    center = get_center_point(box)
                    
                    if class_name.lower() == "needle":
                        needle_corners = box.reshape(4, 2)
                    elif class_name.isdigit() or class_name in ["0", "5", "10", "15", "20", "25", "30"] or class_name.lower() == "numbers":
                        number_positions.append((0, center))
        
        # Calculate meter reading if needle and numbers are detected
        if needle_corners is not None and number_positions:
            reading, method = calculate_meter_reading(needle_corners, number_positions)
            if reading is not None:
                # Format response to match API structure
                kv_list = [
                    {"keyName": "MeterReading", "keyValue": str(reading), 
                     "actualValue": str(reading), "confidenceScore": 0.85}
                ]
                return {"ocs": 0.85, "extractions": kv_list}
            else:
                raise HTTPException(status_code=400, detail="Needle position is out of range")
        else:
            missing = []
            if needle_corners is None:
                missing.append("needle")
            if not number_positions:
                missing.append("numbers")
            raise HTTPException(status_code=400, detail=f"Could not detect {' and '.join(missing)} in analog meter")
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing analog meter: {str(e)}")

@app.post("/detect/")
async def detect(file: UploadFile = File(...), test_type: str = Form(...)):
    file_bytes = await file.read()
    if test_type == "CONDUCTOR_RESISTANCE_TEST":
        return process_res_temp(file_bytes)
    elif test_type == "DC_TEST":
        return process_analog_meter(file_bytes)
    elif test_type == "PARTIAL_DISCHARGE_TEST":
        return process_remaining_test(file_bytes, expected_classes=["UVolt", "qCValue"])
    elif test_type == "HIGH_VOLTAGE_TEST":
        return process_remaining_test(file_bytes, expected_classes=["kV", "TimeLeft", "q(IEC) value"])
    else:
        raise HTTPException(status_code=400, detail="Invalid test_type. Choose 'CONDUCTOR_RESISTANCE_TEST', 'DC_TEST', 'PARTIAL_DISCHARGE_TEST', or 'HIGH_VOLTAGE_TEST'")

@app.get("/")
def health_check():
    return {"status": "healthy", "version": "v2.1"}
