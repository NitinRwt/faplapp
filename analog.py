import cv2
import numpy as np
from ultralytics import YOLO

def detect_and_crop(image_path, model_obb):
    results = model_obb(image_path, conf=0.2)
    obb_data = results[0].obb
    if obb_data is None:
        return None, "No objects detected"
    
    image = cv2.imread(image_path)
    for i, class_id in enumerate(obb_data.cls.cpu().numpy()):
        corners_flat = obb_data.xyxyxyxy.cpu().numpy()[i]
        corners = np.array(corners_flat).reshape(4, 2)
        x_min, y_min = np.min(corners, axis=0)
        x_max, y_max = np.max(corners, axis=0)
        cropped_image = image[int(y_min):int(y_max), int(x_min):int(x_max)]
        return cropped_image, None
    return None, "No meter detected"

def get_meter_reading(cropped_image, model):
    results = model(cropped_image, conf=0.2)
    obb_data = results[0].obb
    class_names = results[0].names

    if obb_data is None:
        return "No objects detected"

    number_positions = []
    needle_corners = None
    number_values = [0, 5, 10, 15, 20, 25, 30]
    
    for i, class_id in enumerate(obb_data.cls.cpu().numpy()):
        class_name = class_names[int(class_id)]
        if hasattr(obb_data, "xyxyxyxy") and obb_data.xyxyxyxy is not None:
            corners_flat = obb_data.xyxyxyxy.cpu().numpy()[i]
            corners = np.array(corners_flat).reshape(4, 2)
            
            if class_name.lower() == "needle":
                needle_corners = corners
            elif class_name.lower() == "numbers":
                center_x = np.mean(corners[:, 0])
                number_positions.append((corners, center_x))
    
    number_positions.sort(key=lambda x: x[1])
    labeled_numbers = [(corners, number_values[i], np.mean(corners[:, 0])) for i, (corners, _) in enumerate(number_positions)]
    
    interpolated_value = None
    if needle_corners is not None and labeled_numbers:
        needle_tip_avg_x = np.mean(needle_corners[2:4, 0])
        left_value, right_value = None, None
        left_center_x, right_center_x = None, None
        
        for corners, value, center_x in labeled_numbers:
            if center_x <= needle_tip_avg_x:
                left_value, left_center_x = value, center_x
            else:
                right_value, right_center_x = value, center_x
                break
        
        if left_value is None:
            interpolated_value = right_value
        elif right_value is None:
            interpolated_value = left_value
        else:
            ratio = (needle_tip_avg_x - left_center_x) / (right_center_x - left_center_x)
            interpolated_value = round(left_value + (ratio * (right_value - left_value)), 1)
    
    return interpolated_value

def main(image_path):
    model_3 = YOLO("Models/analog_box.pt")
    model_4 = YOLO("Models/best.pt")
    
    cropped_image, error = detect_and_crop(image_path, model_3)
    if error:
        print(error)
        return
    
    meter_reading = get_meter_reading(cropped_image, model_4)
    print(f"Meter Reading: {meter_reading} kV")

if __name__ == "__main__":
    image_path = "testimg/images/614.jpg"
    main(image_path)
