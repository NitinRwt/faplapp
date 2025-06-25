import sys
import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR
from ultralytics import YOLO
import re
import matplotlib.pyplot as plt

# python detect.py Models/HV_PD_model.pt test_images/HV_PD/206.png
# Initialize RapidOCR reader
reader = RapidOCR()

def preprocess_cropped_region(cropped_bgr: np.ndarray, mag_ratio: float = 1.6, show_steps: bool = False) -> np.ndarray:
    
    # Handle empty inputs
    if cropped_bgr is None or cropped_bgr.size == 0:
        return None
        
    # 1) Convert to grayscale
    gray = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2GRAY)

    # 2) Upscale by mag_ratio
    h, w = gray.shape
    new_w = int(w * mag_ratio)
    new_h = int(h * mag_ratio)
    gray_up = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 3) Apply Otsu's threshold → binary
    _, thresh = cv2.threshold(
        gray_up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    
    # Show preprocessing steps if requested
    if show_steps:
        visualize_preprocessing_steps(cropped_bgr, gray, gray_up, thresh, mag_ratio)

    return thresh 

def visualize_preprocessing_steps(original, gray, upscaled, thresh, mag_ratio):
    """Visualize the preprocessing steps using matplotlib"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Step 1: Original image
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    axes[0].imshow(original_rgb)
    axes[0].set_title('1. Original')
    axes[0].axis('off')
    
    # Step 2: Grayscale
    axes[1].imshow(gray, cmap='gray')
    axes[1].set_title('2. Grayscale')
    axes[1].axis('off')
    
    # Step 3: Upscaled
    axes[2].imshow(upscaled, cmap='gray')
    axes[2].set_title(f'3. Upscaled {mag_ratio}x')
    axes[2].axis('off')
    
    # Step 4: Thresholded
    axes[3].imshow(thresh, cmap='gray')
    axes[3].set_title('4. Otsu Threshold')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.show()

def clean_extracted_text(text):
    """Clean extracted text by keeping digits and decimal points"""
    if not text or not text.strip():
        return ""
    
    # Extract digits and decimal points from the text
    cleaned = re.sub(r'[^\d.]', '', text)
    
    return cleaned

def manipulate_text(text):
    
    # Clean the text first to get only digits and decimal points
    cleaned_text = clean_extracted_text(text)
    
    if not cleaned_text:
        return ""
    
    # Check if decimal point is already present
    if '.' in cleaned_text:
        # Use the text as is (with existing decimal point)
        final_text = cleaned_text + 'pC'
    else:
        # Extract only digits
        digits = re.findall(r'\d', cleaned_text)
        if not digits:
            return ""
        
        # Join all digits
        digit_string = ''.join(digits)
        
        # Add decimal point: first digit, then dot, then remaining digits
        if len(digit_string) >= 2:
            formatted_text = digit_string[0] + '.' + digit_string[1:]
        else:
            formatted_text = '0.' + digit_string
        
        final_text = formatted_text + 'pC'
    
    return final_text

def draw_obb(image, obb):
    boxes = obb.xyxyxyxy.cpu().numpy()
    class_ids = obb.cls.cpu().numpy()
    extracted_texts = []
    
    # Create a copy for visualization
    vis_image = image.copy()

    for i, box in enumerate(boxes):
        pts = box.reshape(4, 2).astype(np.int32)
        
        # Draw the bounding box
        cv2.polylines(vis_image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # Crop the detected region
        x_min, y_min = np.min(pts, axis=0)
        x_max, y_max = np.max(pts, axis=0)
        cropped_region = image[y_min:y_max, x_min:x_max]
        
        # Apply OCR on the cropped region
        if cropped_region.size > 0:
            # First try with preprocessed image (show steps for first detection)
            show_preprocessing = (i == 0)  # Show preprocessing for first detection only
            preprocessed = preprocess_cropped_region(cropped_region, show_steps=show_preprocessing)
            if preprocessed is not None:
                # RapidOCR returns format: [(bbox, text, confidence), ...] or None
                preprocessed_results = reader(preprocessed)
                if preprocessed_results is not None and len(preprocessed_results) > 0:
                    preprocessed_text = " ".join([result[1] for result in preprocessed_results 
                                                if result is not None and len(result) > 1 and result[1] and result[1].strip()])
                else:
                    preprocessed_text = ""
            else:
                preprocessed_text = ""
                
            # Also try with original image
            original_results = reader(cropped_region)
            if original_results is not None and len(original_results) > 0:
                original_text = " ".join([result[1] for result in original_results 
                                        if result is not None and len(result) > 1 and result[1] and result[1].strip()])
            else:
                original_text = ""
            
            # Use the better result (more text or higher confidence)
            if len(preprocessed_text) > len(original_text):
                detected_text = preprocessed_text
            else:
                detected_text = original_text
                
            if detected_text:
                # Process text to add decimal point if not present and add pC suffix
                final_text = manipulate_text(detected_text)
                
                if final_text:  # Only proceed if valid text found
                    extracted_texts.append(final_text)
                    display_text = final_text
                    
                    # Put extracted text on the image
                    cv2.putText(vis_image, display_text, (x_min, y_min - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    
                    # Add box number for reference
                    cv2.putText(vis_image, f"#{i+1}", (x_min + 5, y_max - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    return vis_image, extracted_texts

def visualize_results_matplotlib(original_image, vis_image, extracted_texts):
    """Display results using matplotlib"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original image
    original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    axes[0].imshow(original_rgb)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Processed image with detections
    vis_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
    axes[1].imshow(vis_rgb)
    axes[1].set_title('Detected Regions with OCR')
    axes[1].axis('off')
    
    # Add extracted texts as subtitle
    if extracted_texts:
        text_str = "Extracted: " + ", ".join(extracted_texts)
        fig.suptitle(text_str, fontsize=12, y=0.02)
    
    plt.tight_layout()
    plt.show()

def main(model_path_3, image_path):
    # Load the YOLO OBB model for detection
    model_3 = YOLO(model_path_3)
    
    # Read the input image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read image at", image_path)
        sys.exit(1)
    
    # Create a copy of the original image
    original_image = image.copy()
    
    # Run inference using model_3 for detection
    results = model_3(image)
    
    all_extracted_texts = []
    
    # Iterate over the results and draw OBB predictions
    for r in results:
        if r.obb is not None:
            vis_image, extracted_texts = draw_obb(original_image, r.obb)
            all_extracted_texts.extend(extracted_texts)
            
            # Print extracted texts from OCR
            for idx, text in enumerate(extracted_texts):
                print(f" {text}")
    
    # Display the visualization if any detections were made
    if all_extracted_texts:        
        # Display using matplotlib
        visualize_results_matplotlib(original_image, vis_image, all_extracted_texts)        
    else:
        print("No text regions detected.")
    
    return vis_image if all_extracted_texts else original_image, all_extracted_texts

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python detect.py <model_path_3> <image_path>")
        sys.exit(1)
    
    model_path_3 = sys.argv[1]
    image_path = sys.argv[2]
    
    vis_image, extracted_texts = main(model_path_3, image_path)
