import sys
import numpy as np
from ultralytics import YOLO
from PIL import Image
import easyocr

def crop_regions(image, obb):
    """Crops regions based on OBB detection and returns (cropped, class_id, x_min)."""
    regions = []
    if obb is not None:
        boxes = obb.xyxyxyxy.cpu().numpy()
        classes = obb.cls.cpu().numpy()
        confs = obb.conf.cpu().numpy()
        for box, cls, conf in zip(boxes, classes, confs):
            if conf < 0.6:
                continue
            pts = box.reshape(4, 2).astype(int)
            x_min, y_min = np.min(pts, axis=0)
            x_max, y_max = np.max(pts, axis=0)
            cropped = image.crop((x_min, y_min, x_max, y_max))
            regions.append((cropped, int(cls), x_min))
    return regions


def detect_and_process(analog_box_model, ocr_model, reader, image):
    """Detects temp or res regions, applies OCR logic, and returns list of (class_name, value)."""
    results = analog_box_model(image)
    output_values = []

    for r in results:
        regions = crop_regions(image, r.obb)
        regions.sort(key=lambda x: x[2])  # sort by x_min

        for cropped, cls_id, _ in regions:
            cls_name = r.names[cls_id]
            if cls_name == 'temp':
                # YOLO OCR logic
                yolo_out = ocr_model(cropped)
                chars = []
                for out in yolo_out:
                    if out.obb is not None:
                        confs = out.obb.conf.cpu().numpy()
                        for i, c in enumerate(out.obb.cls.cpu().numpy()):
                            if confs[i] < 0.6:
                                continue
                            name = out.names[int(c)]
                            pts = out.obb.xyxyxyxy.cpu().numpy()[i].reshape(4,2)
                            x_ctr = np.mean(pts[:,0])
                            chars.append((name, x_ctr))
                chars.sort(key=lambda x: x[1])
                seq = ['.' if n=='dot' else '°' if n=='degree' else n for n,_ in chars]
                value = ''.join(seq)
            else:
                # EasyOCR for residual class
                text = reader.readtext(np.array(cropped), detail=0)
                value = text[0] if text else ''

            output_values.append((cls_name, value))

    return output_values


def main(box_model_path, ocr_model_path, image_path):
    # load models
    analog_box = YOLO(box_model_path)
    yolo_ocr = YOLO(ocr_model_path)
    reader = easyocr.Reader(['en'], gpu=False)

    image = Image.open(image_path).convert('RGB')
    results = detect_and_process(analog_box, yolo_ocr, reader, image)

    # print results
    for cls, val in results:
        print(f"{cls.capitalize()} Detected: {val}")

if __name__ == '__main__':
    box_model = 'Models/res_temp_box.pt'
    ocr_model = 'Models/temp_ocr.pt'
    img_path = 'test_images/cd_cr/k_2.jpg'
    main(box_model, ocr_model, img_path)
