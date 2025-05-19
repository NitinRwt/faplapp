from fastapi import FastAPI, File, UploadFile, Form, HTTPException
import cv2
import numpy as np
import os
from ultralytics import YOLO
from PIL import Image
import io
import easyocr
from ocr import crop_regions, detect_and_process
from Remaining_test import draw_obb  
from analog import crop_region, calculate_meter_reading, get_center_point

app = FastAPI()

try:
    res_temp_box = YOLO("Models/res_temp_box.pt")
    res_temp_ocr = YOLO("Models/temp_ocr.pt")
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
        
        # For OCR model processing (using new ocr.py logic)
        ocr_results = detect_and_process(res_temp_box, res_temp_ocr, reader, image)
        
        # Convert ocr_results to dictionary format
        final_classes_dict = {}
        for class_name, value in ocr_results:
            final_classes_dict[class_name] = value
        
        # Convert image for apparatus model
        image_cv = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        # Rest of your existing code remains the same
        # Process with new apparatus model
        apparatus_results = new_apparatus_model(image_cv)
        apparatus_data = {}
        confidence_scores = {}
        
        # Extract text using apparatus model
        for r in apparatus_results:
            if r.obb is not None:
                # Get confidence scores for detections
                confidences = r.obb.conf.cpu().numpy() if hasattr(r.obb, 'conf') else None
                
                _, extracted_texts = draw_obb(image_cv.copy(), r.obb)
                for i, class_id in enumerate(r.obb.cls.cpu().numpy()):
                    class_name = r.names[int(class_id)]
                    if i < len(extracted_texts) and extracted_texts[i]:
                        apparatus_data[class_name] = extracted_texts[i]
                        
                        # Store confidence score if available
                        if confidences is not None and i < len(confidences):
                            confidence_scores[class_name] = float(confidences[i])
                        else:
                            confidence_scores[class_name] = 0.75  # Default fallback
        
        # Combine results from both models
        final_data = {**final_classes_dict, **apparatus_data}
        
        # Calculate overall confidence (average of available scores)
        overall_confidence = 0.0
        if confidence_scores:
            overall_confidence = sum(confidence_scores.values()) / len(confidence_scores)
        else:
            overall_confidence = 0.75  # Default if no scores available
            
        # Round overall confidence to 2 decimal places
        overall_confidence = round(overall_confidence, 2)
        
        # Convert to key-value list format with individual confidence scores
        kv_list = []
        for k, v in final_data.items():
            # Use the confidence score if available, otherwise use default
            conf = round(confidence_scores.get(k, 0.75), 2)
            kv_list.append({
                "keyName": k, 
                "keyValue": "".join(v) if isinstance(v, list) else v, 
                "actualValue": "".join(v) if isinstance(v, list) else v, 
                "confidenceScore": conf
            })
        
        return {"ocs": overall_confidence, "extractions": kv_list}
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
        confidence_scores = {}
        
        # Process results and extract text from detected regions
        for r in results:
            if r.obb is not None:
                # Get confidence scores from the detections
                confidences = r.obb.conf.cpu().numpy() if hasattr(r.obb, 'conf') else None
                
                # Use the draw_obb function from Remaining_test.py to extract text
                _, extracted_texts = draw_obb(image_cv.copy(), r.obb)
                
                # Match the extracted texts with their class names
                for i, class_id in enumerate(r.obb.cls.cpu().numpy()):
                    class_name = r.names[int(class_id)]
                    
                    # Only process classes that we expect for this test type
                    if class_name in expected_classes and i < len(extracted_texts) and extracted_texts[i]:
                        # Store the detected text with its class name
                        extracted_data[class_name] = extracted_texts[i]
                        
                        # Store confidence score if available
                        if confidences is not None and i < len(confidences):
                            confidence_scores[class_name] = float(confidences[i])
                        else:
                            confidence_scores[class_name] = 0.75  # Default fallback
        
        # Calculate overall confidence score (average of individual scores)
        overall_confidence = 0.0
        if confidence_scores:
            overall_confidence = sum(confidence_scores.values()) / len(confidence_scores)
        else:
            overall_confidence = 0.75  # Default if no scores available
            
        # Round overall confidence to 2 decimal places
        overall_confidence = round(overall_confidence, 2)
        
        # Format response with individual rounded confidence scores
        kv_list = []
        for k, v in extracted_data.items():
            conf = round(confidence_scores.get(k, 0.75), 2)
            kv_list.append({
                "keyName": k, 
                "keyValue": v, 
                "actualValue": v, 
                "confidenceScore": conf
            })
        
        # Determine test type based on expected classes
        test_type = "extractions"
        
        # If no data was extracted
        if not kv_list:
            raise HTTPException(status_code=400, detail=f"No data extracted for the expected classes: {expected_classes}")
            
        return {"ocs": overall_confidence, test_type: kv_list}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing test data: {str(e)}")

def process_dc_test(file_bytes):
    """
    Implements the DC_TEST pipeline using functions from analog.py.
    It decodes the image, ensures consistent color format, detects and crops the meter
    region using the analog_box model, and then uses the analog_reading model along with
    calculate_meter_reading and get_center_point to compute the meter reading.
    """
    try:
        # Decode file bytes into a CV image (BGR)
        image_cv = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image_cv is None:
            raise HTTPException(status_code=400, detail="Invalid image data for DC_TEST")
        
        results = analog_box(image_cv)
        cropped_meter = None
        for r in results:
            if hasattr(r, "obb") and r.obb is not None:
                cropped_meter = crop_region(image_cv, r.obb)
                if cropped_meter is not None:
                    break
        
        if cropped_meter is None:
            raise HTTPException(status_code=400, detail="No analog meter detected in image")
        
        meter_results = analog_reading(cropped_meter)
        needle_corners = None
        needle_corners = None
        number_positions = []
        needle_confidence = 0
        number_confidences = []
        
        for r in meter_results:
            if hasattr(r, "obb") and r.obb is not None:
                boxes = r.obb.xyxyxyxy.cpu().numpy()
                classes = r.obb.cls.cpu().numpy()
                confidences = r.obb.conf.cpu().numpy()  # Get confidence scores
                
                for box, class_id, conf in zip(boxes, classes, confidences):
                    class_name = r.names[int(class_id)]
                    center = get_center_point(box)
                    
                    if class_name.lower() == "needle":
                        needle_corners = box.reshape(4, 2)
                        needle_confidence = float(conf)
                    elif (class_name.isdigit() or 
                          class_name in ["0", "5", "10", "15", "20", "25", "30"] or 
                          class_name.lower() == "numbers"):
                        number_positions.append((0, center))
                        number_confidences.append(float(conf))
        
        if needle_corners is not None and number_positions:
            reading, method = calculate_meter_reading(needle_corners, number_positions)
            
            overall_confidence = (2 * needle_confidence + sum(number_confidences)) / (2 + len(number_confidences))
            overall_confidence = round(overall_confidence, 2)
            
            reading = round(float(reading), 2)
            
            list = [{
                "keyName": "MeterReading",
                "keyValue": str(reading),
                "actualValue": str(reading),
                "confidenceScore": overall_confidence,
            }]
            
            return {
                "ocs": overall_confidence,
                "extractions": list
            }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing DC_TEST: {str(e)}")


@app.post("/detect/")
async def detect(file: UploadFile = File(...), test_type: str = Form(...)):
    file_bytes = await file.read()
    if test_type == "CONDUCTOR_RESISTANCE_TEST":
        return process_res_temp(file_bytes)
    elif test_type == "DC_TEST":
        return process_dc_test(file_bytes)
    elif test_type == "PARTIAL_DISCHARGE_TEST":
        return process_remaining_test(file_bytes, expected_classes=["UVolt", "qCValue"])
    elif test_type == "HIGH_VOLTAGE_TEST":
        return process_remaining_test(file_bytes, expected_classes=["kV", "TimeLeft", "q(IEC) value"])
    else:
        raise HTTPException(status_code=400, detail="Invalid test_type. Choose 'CONDUCTOR_RESISTANCE_TEST', 'DC_TEST', 'PARTIAL_DISCHARGE_TEST', or 'HIGH_VOLTAGE_TEST'")

@app.get("/")
def health_check():
    return {"status": "healthy", "version": "v2.1"}