import os
import cv2
import numpy as np
from ultralytics import YOLO
import tensorflow as tf

class YoloLeNetOCR:
    def __init__(self,
                 yolo_model_path: str,
                 lenet_model_path: str,
                 image_size=(28, 28),
                 conf_threshold=0.25):
        # YOLO detector
        self.detector = YOLO(yolo_model_path)
        # LeNet CNN
        self.cnn = tf.keras.models.load_model(lenet_model_path)

        # Embedded class names and inverse map (no external pkl needed)
        class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'C', 'dot']
        self.inv_map = {i: label for i, label in enumerate(class_names)}

        # params
        self.image_size = image_size
        self.conf_threshold = conf_threshold

    def preprocess(self, crop: np.ndarray) -> np.ndarray:
        # Convert to grayscale for CNN
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, self.image_size)
        normed = resized.astype(np.float32) / 255.0
        # CNN expects shape (1, H, W, 1)
        return normed.reshape(1, *self.image_size, 1)

    def ocr_image(self, image_path: str) -> str:
        # 1) Load as single-channel grayscale
        gray0 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if gray0 is None:
            raise FileNotFoundError(f"Cannot read {image_path}")

        # 2) Convert to BGR by stacking gray into 3 channels
        img = cv2.cvtColor(gray0, cv2.COLOR_GRAY2BGR)

        # 3) Detect boxes on the grayscale-derived BGR image
        res = self.detector.predict(source=img, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        classes = res.boxes.cls.cpu().numpy().astype(int) 
        boxes = boxes[confs >= self.conf_threshold]
        classes = classes[confs >= self.conf_threshold]
        if len(boxes) == 0:
            return ""

        # 4) Sort boxes left-to-right
        sort_idx = np.argsort(boxes[:, 0])
        boxes = boxes[sort_idx]
        classes = classes[sort_idx]
        digits = []

        for (x1, y1, x2, y2), cls_idx in zip(boxes, classes):
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            crop = img[
                max(0, y1):min(img.shape[0], y2),
                max(0, x1):min(img.shape[1], x2)
            ]

            # If YOLO says it's a dot, just append "dot"
            if self.inv_map[cls_idx] == "dot":
                digits.append("dot")
            else:
                # 5) Preprocess and predict with LeNet
                inp = self.preprocess(crop)
                preds = self.cnn.predict(inp, verbose=0)
                idx = int(np.argmax(preds, axis=1)[0])
                digits.append(self.inv_map[idx])

        return "".join(digits)
    
    def process_dataset_folder(self, dataset_dir, output_file=None):
        """
        Process all images in a dataset directory and its subdirectories.
        
        Args:
            dataset_dir (str): Path to the dataset directory
            output_file (str): Optional path to save results to a text file
        
        Returns:
            dict: Dictionary mapping image paths to OCR results
        """
        results = {}
        processed = 0
        
        # Iterate through all subdirectories
        for root, _, files in os.walk(dataset_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(root, file)
                    try:
                        ocr_result = self.ocr_image(image_path)
                        ocr_result = ocr_result.replace("dot", ".")
                        results[image_path] = ocr_result
                        processed += 1
                        
                        # Print progress
                        if processed % 100 == 0:
                            print(f"Processed {processed} images")
                            
                    except Exception as e:
                        print(f"Error processing {image_path}: {str(e)}")
        
        # Save results to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                for img_path, result in results.items():
                    f.write(f"{img_path},{result}\n")
            print(f"Results saved to {output_file}")
            
        print(f"Total images processed: {processed}")
        return results

# Example usage
if __name__ == "__main__":
    ocr = YoloLeNetOCR(
        yolo_model_path="Models/res_detect_v4.pt",
        lenet_model_path="Models/lenet_res_v4.h5",
        conf_threshold=0.3
    )
    
    # Process the entire new_data/res folder
    results = ocr.process_dataset_folder(
        dataset_dir="new_data/res",
        output_file="ocr_results.csv"
    )
    
    # Print sample results
    print("\nSample results:")
    sample_count = 0
    for path, text in results.items():
        print(f"{path}: {text}")
        sample_count += 1
        if sample_count >= 5:
            break


# # -------------------
# # Example usage
# # -------------------
# if __name__ == "__main__":
#     ocr = YoloLeNetOCR(
#         yolo_model_path="Models/res_detect_v3.pt",
#         lenet_model_path="Models/lenet_res_v4.h5",
#         conf_threshold=0.3
#     )
#     result = ocr.ocr_image("new_data/res")
#     result = result.replace("dot", ".")
#     print("OCR result:", result)