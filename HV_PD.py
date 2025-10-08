import sys
import cv2
import numpy as np
import re
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO
import easyocr
import torch

# Initialize TrOCR processor and model (printed text)
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
# Initialize EasyOCR reader
gpu_flag = torch.cuda.is_available()
reader = easyocr.Reader(['en'], gpu=gpu_flag)
# Move TrOCR model to GPU if available
device = "cuda" if gpu_flag else "cpu"
model.to(device)


def preprocess_cropped_region(cropped_bgr: np.ndarray, mag_ratio: float = 1.6) -> np.ndarray:
    if cropped_bgr is None or cropped_bgr.size == 0:
        return None
    gray = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    new_w = int(w * mag_ratio)
    new_h = int(h * mag_ratio)
    gray_up = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    _, thresh = cv2.threshold(gray_up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def clean_extracted_text(text: str) -> str:
    if not text or not text.strip():
        return ""
    return re.sub(r"[^0-9.]", "", text)


def manipulate_text(text: str) -> str:
    """
    Normalize OCR output to the format "xx.xx pC" (two decimal places).
    Handles missing decimal points, misrecognized 'pC', and ensures the suffix.
    """
    if not text or not text.strip():
        return ""
    # Remove spaces
    txt = text.replace(' ', '')
    # Remove common misreads of 'pC'
    txt = re.sub(r'(?i)p[0o]?c', '', txt)
    # Extract numeric part
    m = re.search(r"(\d+\.?\d*)", txt)
    if not m:
        return ""
    num = m.group(1)
    # Split integer and fraction
    if '.' in num:
        integer, frac = num.split('.', 1)
    else:
        # Infer two-digit fraction if missing
        if len(num) > 2:
            integer, frac = num[:-2], num[-2:]
        elif len(num) == 2:
            integer, frac = num[0], num[1] + '0'
        elif len(num) == 1:
            integer, frac = '0', num + '0'
        else:
            return ""
    # Ensure exactly two fraction digits
    if len(frac) < 2:
        frac = (frac + '00')[:2]
    else:
        frac = frac[:2]
    integer = integer or '0'
    return f"{integer}.{frac} pC"


def ocr_trocr(cropped_bgr: np.ndarray) -> str:
    image_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)
    pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


def ocr_easyocr(cropped_bgr: np.ndarray) -> str:
    results = reader.readtext(cropped_bgr)
    return " ".join([r[1] for r in results if r[1].strip()])


def draw_obb_with_classes(image: np.ndarray, obb, class_names) -> tuple:
    """Draw OBB function that uses class names to determine OCR method"""
    boxes = obb.xyxyxyxy.cpu().numpy()
    class_ids = obb.cls.cpu().numpy()
    vis_image = image.copy()
    extracted_texts = []

    for i, box in enumerate(boxes):
        pts = box.reshape(4, 2).astype(np.int32)
        cv2.polylines(vis_image, [pts], True, (0, 255, 0), 2)
        x_min, y_min = np.min(pts, axis=0)
        x_max, y_max = np.max(pts, axis=0)
        crop = image[y_min:y_max, x_min:x_max]
        if crop.size == 0:
            continue
        
        # Get class name from class ID
        class_id = int(class_ids[i])
        class_name = class_names[class_id]
        
        # Use TrOCR for qCValue, EasyOCR for all others
        text = ""
        try:
            if class_name == "qCValue":
                text = ocr_trocr(crop)
            else:
                text = ocr_easyocr(crop)
        except Exception:
            # Fallback: try the other OCR method
            try:
                if class_name == "qCValue":
                    text = ocr_easyocr(crop)
                else:
                    text = ocr_trocr(crop)
            except Exception:
                text = ""
        
        if text.strip():
            # Apply text processing based on class
            if class_name in ["qCValue", "q(IEC) value"]:
                norm = manipulate_text(text)
            else:
                # For other classes, return text as-is or with appropriate suffix
                norm = text.strip()
                if class_name == "kV" and not norm.lower().endswith('kv'):
                    norm += 'kV'
            
            if norm:
                extracted_texts.append(norm)
                cv2.putText(vis_image, norm, (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
                cv2.putText(vis_image, f"#{i+1}", (x_min+5, y_max-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2)
    return vis_image, extracted_texts

def draw_obb(image: np.ndarray, obb) -> tuple:
    """Legacy draw_obb function - uses TrOCR first, then EasyOCR fallback"""
    boxes = obb.xyxyxyxy.cpu().numpy()
    vis_image = image.copy()
    extracted_texts = []

    for i, box in enumerate(boxes):
        pts = box.reshape(4, 2).astype(np.int32)
        cv2.polylines(vis_image, [pts], True, (0, 255, 0), 2)
        x_min, y_min = np.min(pts, axis=0)
        x_max, y_max = np.max(pts, axis=0)
        crop = image[y_min:y_max, x_min:x_max]
        if crop.size == 0:
            continue
        # Try TrOCR first
        text = ""
        try:
            text = ocr_trocr(crop)
        except Exception:
            pass
        # Fallback to EasyOCR
        if not text.strip():
            text = ocr_easyocr(crop)
        if text.strip():
            norm = manipulate_text(text)
            if norm:
                extracted_texts.append(norm)
                cv2.putText(vis_image, norm, (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
                cv2.putText(vis_image, f"#{i+1}", (x_min+5, y_max-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2)
    return vis_image, extracted_texts


def main(model_path: str, image_path: str) -> tuple:
    model_det = YOLO(model_path)
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: could not read {image_path}")
        sys.exit(1)
    vis = img.copy()
    all_texts = []
    for res in model_det(img):
        if res.obb is not None:
            vis, texts = draw_obb(vis, res.obb)
            all_texts.extend(texts)
            for t in texts:
                print(t)
    return vis, all_texts

# Example:
# v, t = main('Models/HV_PD_model.pt', 'test_images/HV_PD/206.png')
# cv2.imshow('Result', v); cv2.waitKey(0); cv2.destroyAllWindows()
