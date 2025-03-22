from fastapi import FastAPI, File, UploadFile, Form, HTTPException
import cv2
import numpy as np
import shutil
from ultralytics import YOLO
from PIL import Image
import io
import os
from ocr import detect_and_crop as ocr_detect_and_crop, detect_final_classes
from analog import detect_and_crop as analog_detect_and_crop, get_meter_reading

app = FastAPI()

try:
    model_1 = YOLO("Models/res_temp_box.pt")
    model_2 = YOLO("Models/res_temp_ocr.pt")
    model_3 = YOLO("Models/analog_box.pt")
    model_4 = YOLO("Models/best.pt")
except Exception as e:
    print(f"Error loading models: {str(e)}")
    raise

TEST_VALUES = {
    "test1": {"analog meter"},
    "test2": {"res", "temp"},
}

def process_analog_meter(image_path):
    """Process analog meter using analog.py functions"""
    try:
        cropped_image, error = analog_detect_and_crop(image_path, model_3)
        if error:
            raise HTTPException(status_code=400, detail=error)
        
        meter_reading = get_meter_reading(cropped_image, model_4)
        if isinstance(meter_reading, str):
            raise HTTPException(status_code=400, detail=meter_reading)
        
        return {"meter_reading_kV": meter_reading}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Not Valid test is Selected !!: {str(e)}")

def process_res_temp(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        
        # Get cropped regions using first model
        cropped_regions = ocr_detect_and_crop(model_1, image)
        if not cropped_regions:
            raise HTTPException(status_code=400, detail="No regions detected in the image")
        
        # Get final classes using second model
        final_classes_dict = detect_final_classes(model_2, cropped_regions)
        if not final_classes_dict:
            raise HTTPException(status_code=400, detail="No classes detected in the image")
        
        result = {}
        # Process each class separately
        for class_name, values in final_classes_dict.items():
            detected_value = "".join(values)
            formatted_value = detected_value

            # Format for temperature readings
            if "°" in formatted_value:
                parts = formatted_value.split("°")
                if len(parts) > 2 and parts[0].isdigit() and parts[1].isdigit():
                    formatted_value = f"{parts[0]}.{parts[1]}°{parts[2]}"
                elif len(parts) == 2:
                    formatted_value = f"{parts[0]}°{parts[1]}"
            
            result[class_name] = formatted_value
    
        return {
            "detected_values": result
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error processing image: {str(e)}"
        )

@app.post("/detect/")
async def detect(
    file: UploadFile = File(...),
    test_case: str = Form(None),
    detection_type: str = Form(None)
):
    # Validate file
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload PNG or JPEG images")

    # Create temp directory if it doesn't exist
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Save uploaded file temporarily
    temp_path = os.path.join(temp_dir, f"temp_{file.filename}")
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process based on test case
        if test_case and test_case in TEST_VALUES:
            expected_classes = TEST_VALUES[test_case]
            if "analog meter" in expected_classes:
                result = process_analog_meter(temp_path)
                return {"detection_type": "analog_meter", "result": result}
            else:
                result = process_res_temp(temp_path)
                return {"detection_type": "res_temp", "result": result}
        
        # If no test case but detection_type is provided
        elif detection_type:
            if detection_type not in ["analog_meter", "res_temp"]:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid detection_type. Choose 'analog_meter' or 'res_temp'"
                )
            
            if detection_type == "analog_meter":
                result = process_analog_meter(temp_path)
            else:
                result = process_res_temp(temp_path)
            return {"detection_type": detection_type, "result": result}
        
        # If neither test_case nor detection_type is provided
        else:
            raise HTTPException(
                status_code=400,
                detail="Missing required field. Provide either 'test_case' ('test1' or 'test2') or 'detection_type' ('analog_meter' or 'res_temp')"
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/")
def health_check():
    return {"status": "healthy", "version": "1.0"}