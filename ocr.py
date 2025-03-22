import sys
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image

def visualize_with_obb(image, obb):
    """Visualizes OBB bounding boxes on the image using Matplotlib."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image)

    if obb is not None:
        boxes = obb.xyxyxyxy.cpu().numpy()
        for box in boxes:
            pts = box.reshape(4, 2)  # Convert to (x, y) points
            ax.plot([pts[i][0] for i in [0,1,2,3,0]], 
                    [pts[i][1] for i in [0,1,2,3,0]], 
                    linestyle='-', linewidth=2, color='lime')
            

def crop_regions(image, obb, class_ids, names):
    """Crops regions based on OBB detection and returns (cropped image, x-coordinate, class_name)."""
    cropped_regions = []

    if obb is not None:
        boxes = obb.xyxyxyxy.cpu().numpy()
        for i, box in enumerate(boxes):
            pts = box.reshape(4, 2).astype(int)
            x_min, y_min = np.min(pts, axis=0)
            x_max, y_max = np.max(pts, axis=0)
            
            # Get class name for this box
            class_id = int(class_ids[i])
            class_name = names[class_id]

            cropped = image.crop((x_min, y_min, x_max, y_max))
            cropped_regions.append((cropped, x_min, class_name))  # Store class_name with cropped image
    
    return cropped_regions

def detect_and_crop(model_3, image):
    """Runs first detection model (OBB-based) and returns cropped regions with class info."""
    results = model_3(image)
    cropped_regions = []
    
    for r in results:
        visualize_with_obb(image, r.obb)  # Display first detection
        if r.obb is not None:
            cropped_regions = crop_regions(image, r.obb, r.obb.cls.cpu().numpy(), r.names)

    return cropped_regions

def detect_final_classes(model_4, cropped_regions):
    """Runs second detection model (OBB-based OCR) and returns detected classes by first class."""
    class_results = {}  # Dictionary to store results by class

    for cropped, x_min, class_name in cropped_regions:
        results = model_4(cropped)
        detected_data = []
        
        for r in results:
            if r.obb is not None:
                for i, class_id in enumerate(r.obb.cls.cpu().numpy()):
                    ocr_class_name = r.names[int(class_id)]
                    box_pts = r.obb.xyxyxyxy.cpu().numpy()[i].reshape(4, 2)
                    x_center = np.mean(box_pts[:, 0])

                    detected_data.append((ocr_class_name, x_center))

        # Sort detected characters by x-center (left to right)
        detected_data.sort(key=lambda x: x[1])
        
        # Process to place '.' after second digit if needed
        final_classes = [
            "." if cls == "dot" else "°" if cls == "degree" else cls  
            for cls, _ in detected_data
        ]
        
        # Store result by first detection class
        if class_name not in class_results:
            class_results[class_name] = []
        class_results[class_name] = final_classes

    return class_results

def main(model_path_3, model_path_4, image_path):
    """Main function to run both detections using OBB models and visualize results."""
    model_3 = YOLO(model_path_3)  
    model_4 = YOLO(model_path_4)  

    image = Image.open(image_path).convert("RGB")

    # First detection to identify and crop regions by class
    cropped_regions = detect_and_crop(model_3, image)
    
    # Second detection to read values from each cropped region
    class_results = detect_final_classes(model_4, cropped_regions)

    # Display results for each class separately
    print("Detection Results by Class:")
    for class_name, values in class_results.items():
        print(f"  {class_name}: {''.join(values)}")

if __name__ == "__main__":
    model_path_3 = "Models/res_temp_box.pt" 
    model_path_4 = "Models/res_temp_ocr.pt" 
    image_path = "res_temp/RT2.jpeg"  
    main(model_path_3, model_path_4, image_path)
