import sys
import cv2
import numpy as np
from ultralytics import YOLO

# -----------------------------
# Part 1: Helper functions for cropping
# -----------------------------

def draw_obb(image, obb):
    """Draw oriented bounding boxes on an image."""
    boxes = obb.xyxyxyxy.cpu().numpy()
    for box in boxes:
        pts = box.reshape(4, 2).astype(np.int32)
        cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    return image

def order_points(pts):
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def crop_region(image, obb):
    """
    Crop the meter region from the image using the OBB.
    Uses a perspective transformation based on the minimal area rectangle.
    """
    boxes = obb.xyxyxyxy.cpu().numpy()
    if len(boxes) == 0:
        return None
    # Use the first detected box for cropping.
    box = boxes[0]
    pts = box.reshape(4, 2).astype(np.float32)
    
    # Get the minimal area rectangle for the points.
    rect = cv2.minAreaRect(pts)
    width = int(rect[1][0])
    height = int(rect[1][1])
    if width <= 0 or height <= 0:
        return None

    # Destination points for the warp (top-left, top-right, bottom-right, bottom-left)
    dst_pts = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]], dtype=np.float32)
    
    # Order the source points and compute the perspective transform.
    ordered_pts = order_points(pts)
    M = cv2.getPerspectiveTransform(ordered_pts, dst_pts)
    cropped = cv2.warpPerspective(image, M, (width, height))
    return cropped

def detect_and_crop_region(analog_box_model, image_path):
    """
    Detect the meter region using analog_box.pt and return the cropped image.
    """
    model = YOLO(analog_box_model)
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read image at", image_path)
        sys.exit(1)
    
    results = model(image)
    for r in results:
        if hasattr(r, "obb") and r.obb is not None:
            cropped = crop_region(image, r.obb)
            if cropped is not None:
                return cropped
    print("No meter detected.")
    sys.exit(1)

# -----------------------------
# Part 2: Meter reading functions (provided calculation code)
# -----------------------------

def get_center_point(box):
    """Calculate the center point of a bounding box (4 corners)."""
    pts = box.reshape(4, 2)
    center_x = np.mean(pts[:, 0])
    center_y = np.mean(pts[:, 1])
    return (center_x, center_y)

def calculate_meter_reading(needle_corners, number_positions):
    """
    Given the needle corners and number positions, calculate the meter reading.
    The numbers are standardized as [0, 5, 10, 15, 20, 25, 30].
    """
    number_values = [0, 5, 10, 15, 20, 25, 30]
    
    # Sort number positions left-to-right by x-coordinate.
    sorted_positions = sorted(number_positions, key=lambda x: x[1][0])
    labeled_positions = []
    for i, (_, position) in enumerate(sorted_positions):
        if i < len(number_values):
            labeled_positions.append((number_values[i], position))
    
    # Compute needle tip as midpoint between corner 3 and corner 4.
    needle_tip_x = (needle_corners[2][0] + needle_corners[3][0]) / 2
    needle_tip_y = (needle_corners[2][1] + needle_corners[3][1]) / 2
    needle_tip = np.array([needle_tip_x, needle_tip_y])
    
    # Check if needle tip exactly matches a number position.
    for value, position in labeled_positions:
        distance = np.sqrt((needle_tip[0] - position[0])**2 + (needle_tip[1] - position[1])**2)
        if distance < 15:  # threshold for "exact match"
            return value, "exact_midpoint"
    
    # If not an exact match, find the two numbers between which the needle lies.
    left_value = None
    right_value = None
    left_position = None
    right_position = None
    for i in range(len(labeled_positions) - 1):
        curr_value, curr_pos = labeled_positions[i]
        next_value, next_pos = labeled_positions[i + 1]
        if curr_pos[0] <= needle_tip[0] <= next_pos[0]:
            left_value = curr_value
            right_value = next_value
            left_position = curr_pos
            right_position = next_pos
            break

    # If not between any two, return the closest.
    if left_value is None or right_value is None:
        min_distance = float('inf')
        closest_value = None
        for value, position in labeled_positions:
            distance = np.sqrt((needle_tip[0] - position[0])**2 + (needle_tip[1] - position[1])**2)
            if distance < min_distance:
                min_distance = distance
                closest_value = value
        return closest_value, "closest_midpoint"
    
    # Interpolate based on x-distance.
    total_x_distance = right_position[0] - left_position[0]
    needle_x_distance = needle_tip[0] - left_position[0]
    ratio = needle_x_distance / total_x_distance if total_x_distance > 0 else 0
    value_range = right_value - left_value
    interpolated_value = left_value + (ratio * value_range)
    interpolated_value = round(interpolated_value, 1)
    
    return interpolated_value, "interpolated_midpoint"

def process_meter_reading(analog_reading_model, image):
    """
    Run detection on the provided (cropped) meter image using analog_reading_v2.pt,
    compute the meter reading, and print the result.
    """
    model = YOLO(analog_reading_model)
    results = model(image)
    
    needle_corners = None
    number_positions = []  # Each element is a tuple: (detected_label, center)
    
    # Process each detection result.
    for r in results:
        if hasattr(r, "obb") and r.obb is not None:
            image = draw_obb(image, r.obb)
            boxes = r.obb.xyxyxyxy.cpu().numpy()
            classes = r.obb.cls.cpu().numpy()
            
            for box, class_id in zip(boxes, classes):
                class_name = r.names[int(class_id)]
                center = get_center_point(box)
                cv2.circle(image, (int(center[0]), int(center[1])), 3, (0, 0, 255), -1)
                
                if class_name.lower() == "needle":
                    needle_corners = box.reshape(4, 2)
                # Check if class is a digit (or the word "numbers") representing meter numbers.
                elif class_name.isdigit() or class_name in ["0", "5", "10", "15", "20", "25", "30"] or class_name.lower() == "numbers":
                    number_positions.append((0, center))
    
    # Label the numbers (using standard ordering) on the image.
    if number_positions:
        number_values = [0, 5, 10, 15, 20, 25, 30]
        sorted_positions = sorted(number_positions, key=lambda x: x[1][0])
        for i, (_, position) in enumerate(sorted_positions):
            if i < len(number_values):
                label = str(number_values[i])
                cv2.putText(image, label, 
                            (int(position[0]), int(position[1]) - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Compute and print the meter reading if needle and numbers are detected.
    if needle_corners is not None and number_positions:
        needle_tip_x = (needle_corners[2][0] + needle_corners[3][0]) / 2
        needle_tip_y = (needle_corners[2][1] + needle_corners[3][1]) / 2
        needle_tip = np.array([needle_tip_x, needle_tip_y])
                
        reading, method = calculate_meter_reading(needle_corners, number_positions)
        if reading is not None:
            result_text = f"Meter reading: {reading} ({method})"
            print(result_text)
            
            # Visualize connection between the needle tip and the nearest number.
            number_values = [0, 5, 10, 15, 20, 25, 30]
            sorted_positions = sorted(number_positions, key=lambda x: x[1][0])
            labeled_positions = []
            for i, (_, position) in enumerate(sorted_positions):
                if i < len(number_values):
                    labeled_positions.append((number_values[i], position))
            
            # Find adjacent numbers for interpolation visualization.
            left_pos = None
            right_pos = None
            for i in range(len(labeled_positions) - 1):
                curr_value, curr_pos = labeled_positions[i]
                next_value, next_pos = labeled_positions[i + 1]
                if curr_pos[0] <= needle_tip[0] <= next_pos[0]:
                    left_pos = curr_pos
                    right_pos = next_pos
                    break
            
            if "interpolated" in method and left_pos is not None and right_pos is not None:
                cv2.line(image, 
                         (int(needle_tip[0]), int(needle_tip[1])), 
                         (int(left_pos[0]), int(left_pos[1])), 
                         (255, 0, 255), 1, cv2.LINE_AA)
                cv2.line(image, 
                         (int(needle_tip[0]), int(needle_tip[1])), 
                         (int(right_pos[0]), int(right_pos[1])), 
                         (255, 0, 255), 1, cv2.LINE_AA)
            else:
                # Connect to closest number if not interpolated.
                min_distance = float('inf')
                closest_position = None
                for _, position in labeled_positions:
                    distance = np.sqrt((needle_tip[0] - position[0])**2 + 
                                       (needle_tip[1] - position[1])**2)
                    if distance < min_distance:
                        min_distance = distance
                        closest_position = position
                if closest_position is not None:
                    cv2.line(image, 
                             (int(needle_tip[0]), int(needle_tip[1])), 
                             (int(closest_position[0]), int(closest_position[1])), 
                             (255, 0, 255), 2)
        else:
            print("Needle position is out of range")
    else:
        if needle_corners is None:
            print("Needle not detected")
        if not number_positions:
            print("No numbers detected")
            
    return image