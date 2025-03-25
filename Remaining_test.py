import sys
import cv2
import numpy as np
import easyocr
from ultralytics import YOLO

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

def draw_obb(image, obb):
    boxes = obb.xyxyxyxy.cpu().numpy()
    extracted_texts = []

    for i, box in enumerate(boxes):
        pts = box.reshape(4, 2).astype(np.int32)
        
        # Draw the bounding box
        cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # Crop the detected region
        x_min, y_min = np.min(pts, axis=0)
        x_max, y_max = np.max(pts, axis=0)
        cropped_region = image[y_min:y_max, x_min:x_max]
        
        # Apply OCR on the cropped region
        if cropped_region.size > 0:
            text_results = reader.readtext(cropped_region)
            detected_text = " ".join([text[1] for text in text_results])
            extracted_texts.append(detected_text)

            # Put extracted text on the image
            cv2.putText(image, detected_text, (x_min, y_min - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    return image, extracted_texts

def main(model_path_3, image_path):
    # Load the YOLO OBB model for detection
    model_3 = YOLO(model_path_3)
    
    # Read the input image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read image at", image_path)
        sys.exit(1)
    
    # Run inference using model_3 for detection
    results = model_3(image)
    
    # Iterate over the results and draw OBB predictions
    for r in results:
        if r.obb is not None:
            image, extracted_texts = draw_obb(image, r.obb)
            for i, class_id in enumerate(r.obb.cls.cpu().numpy()):
                class_name = r.names[int(class_id)]
                print(f"Detected class ID: {class_id}, Class name: {class_name}")
            
            # Print extracted texts from OCR
            for idx, text in enumerate(extracted_texts):
                print(f"OCR Extracted Text {idx + 1}: {text}")

    # Display the resulting image with bounding boxes and text
    cv2.imshow("Detections with OCR", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model_path_3 = "Models/R2.pt"
    image_path = "images/i_107.png"
    main(model_path_3, image_path)
