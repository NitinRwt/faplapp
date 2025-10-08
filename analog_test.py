import sys
import cv2
import numpy as np
from ultralytics import YOLO
import math

# -----------------------------
# Part 1: Helper functions for cropping (unchanged)
# -----------------------------

def draw_obb(image, obb):
    """Draw oriented bounding boxes on an image."""
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
    """Crop the meter region from the image using the OBB."""
    boxes = obb.xyxyxyxy.cpu().numpy()
    if len(boxes) == 0:
        return None
    box = boxes[0]
    pts = box.reshape(4, 2).astype(np.float32)
    
    rect = cv2.minAreaRect(pts)
    width = int(rect[1][0])
    height = int(rect[1][1])
    if width <= 0 or height <= 0:
        return None

    dst_pts = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]], dtype=np.float32)
    
    ordered_pts = order_points(pts)
    M = cv2.getPerspectiveTransform(ordered_pts, dst_pts)
    cropped = cv2.warpPerspective(image, M, (width, height))
    return cropped

def detect_and_crop_region(analog_box_model, image_path):
    """Detect the meter region using analog_box.pt and return the cropped image."""
    model = YOLO(analog_box_model)
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    results = model(image)
    for r in results:
        if hasattr(r, "obb") and r.obb is not None:
            cropped = crop_region(image, r.obb)
            if cropped is not None:
                return cropped
    return None

# -----------------------------
# Part 2: Enhanced needle corner coordinate ratio logic
# -----------------------------

def get_center_point(box):
    """Calculate the center point of a bounding box (4 corners)."""
    pts = box.reshape(4, 2)
    center_x = np.mean(pts[:, 0])
    center_y = np.mean(pts[:, 1])
    return (center_x, center_y)

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def find_needle_tip_corners(needle_corners):
    """
    Extract needle tip coordinates from corners 3 and 4.
    Returns both individual corners and their midpoint.
    """
    corner3 = needle_corners[2]  # Corner 3
    corner4 = needle_corners[3]  # Corner 4
    
    # Calculate midpoint of corners 3 and 4
    tip_midpoint = np.array([
        (corner3[0] + corner4[0]) / 2,
        (corner3[1] + corner4[1]) / 2
    ])
    
    return corner3, corner4, tip_midpoint

def find_digit_boundaries(labeled_positions):
    """
    Create boundaries between consecutive digits for range detection.
    Returns list of (lower_value, upper_value, lower_pos, upper_pos) tuples.
    """
    boundaries = []
    for i in range(len(labeled_positions) - 1):
        lower_value, lower_pos = labeled_positions[i]
        upper_value, upper_pos = labeled_positions[i + 1]
        boundaries.append((lower_value, upper_value, lower_pos, upper_pos))
    return boundaries

def calculate_needle_corner_ratio_approximation(needle_corners, number_positions):
    """
    Enhanced ratio approximation logic based on needle corners 3 and 4 coordinates.
    """
    number_values = [0, 5, 10, 15, 20, 25, 30]
    
    # Sort number positions left-to-right by x-coordinate
    sorted_positions = sorted(number_positions, key=lambda x: x[1][0])
    labeled_positions = []
    for i, (_, position) in enumerate(sorted_positions):
        if i < len(number_values):
            labeled_positions.append((number_values[i], position))
    
    if len(labeled_positions) < 2:
        return None, "insufficient_digits"
    
    # Get needle tip corner coordinates
    corner3, corner4, tip_midpoint = find_needle_tip_corners(needle_corners)
    
    # Define proximity thresholds
    EXACT_MATCH_THRESHOLD = 15  # Very close to digit
    PROXIMITY_THRESHOLD = 50    # Within range for ratio calculation
    
    # Check each needle corner position
    needle_points = [
        ("Corner 3", corner3),
        ("Corner 4", corner4),
        ("Tip Midpoint", tip_midpoint)
    ]
    
    best_reading = None
    best_method = None
    best_confidence = 0
    
    for point_name, needle_point in needle_points:
        
        # Find closest digit to this needle point
        closest_distance = float('inf')
        closest_digit = None
        closest_position = None
        
        for value, position in labeled_positions:
            distance = calculate_distance(needle_point, position)
            if distance < closest_distance:
                closest_distance = distance
                closest_digit = value
                closest_position = position
        
        # Check if needle point is very close to a digit (exact match)
        if closest_distance < EXACT_MATCH_THRESHOLD:
            confidence = 1.0 - (closest_distance / EXACT_MATCH_THRESHOLD)
            if confidence > best_confidence:
                best_reading = float(closest_digit)
                best_method = f"exact_match_{point_name.lower().replace(' ', '_')}"
                best_confidence = confidence
            continue
        
        # Check if needle point is within proximity for ratio approximation
        if closest_distance < PROXIMITY_THRESHOLD:
            # Find the digit range this needle point falls into
            digit_boundaries = find_digit_boundaries(labeled_positions)
            
            for lower_value, upper_value, lower_pos, upper_pos in digit_boundaries:
                # Check if needle point is between these two digits
                lower_dist = calculate_distance(needle_point, lower_pos)
                upper_dist = calculate_distance(needle_point, upper_pos)
                
                # Check if needle X coordinate is between the two digit X coordinates
                if lower_pos[0] <= needle_point[0] <= upper_pos[0]:
                    
                    # Calculate ratio based on X-coordinate position
                    x_range = upper_pos[0] - lower_pos[0]
                    x_offset = needle_point[0] - lower_pos[0]
                    x_ratio = x_offset / x_range if x_range > 0 else 0
                    
                    # Calculate ratio based on distance
                    total_distance = lower_dist + upper_dist
                    distance_ratio = upper_dist / total_distance if total_distance > 0 else 0
                    
                    # Combine ratios for better accuracy
                    combined_ratio = (x_ratio + distance_ratio) / 2
                    
                    # Calculate interpolated value
                    value_range = upper_value - lower_value
                    interpolated_value = lower_value + (combined_ratio * value_range)
                    
                    # Check which digit the needle is closer to
                    if lower_dist < upper_dist:
                        closer_digit = lower_value
                        closer_distance = lower_dist
                    else:
                        closer_digit = upper_value
                        closer_distance = upper_dist
                    
                    # If very close to the closer digit, return that digit
                    if closer_distance < EXACT_MATCH_THRESHOLD * 1.5:  # Slightly more lenient
                        confidence = 1.0 - (closer_distance / (EXACT_MATCH_THRESHOLD * 1.5))
                        if confidence > best_confidence:
                            best_reading = float(closer_digit)
                            best_method = f"close_to_digit_{point_name.lower().replace(' ', '_')}"
                            best_confidence = confidence
                    else:
                        # Use ratio approximation
                        confidence = 1.0 - (min(lower_dist, upper_dist) / PROXIMITY_THRESHOLD)
                        if confidence > best_confidence:
                            best_reading = round(interpolated_value, 1)
                            best_method = f"ratio_approximation_{point_name.lower().replace(' ', '_')}"
                            best_confidence = confidence
                    
                    break
    
    # Return the best result found
    if best_reading is not None:
        return best_reading, best_method
    
    # Fallback: use closest digit to tip midpoint
    closest_distance = float('inf')
    closest_value = None
    for value, position in labeled_positions:
        distance = calculate_distance(tip_midpoint, position)
        if distance < closest_distance:
            closest_distance = distance
            closest_value = value
    
    return float(closest_value), "fallback_closest"

def process_meter_reading_with_corner_logic(analog_reading_model, image):
    """
    Process meter reading using the enhanced needle corner coordinate logic.
    """
    model = YOLO(analog_reading_model)
    results = model(image)
    
    needle_corners = None
    number_positions = []
    
    # Process each detection result
    for r in results:
        if hasattr(r, "obb") and r.obb is not None:
            boxes = r.obb.xyxyxyxy.cpu().numpy()
            classes = r.obb.cls.cpu().numpy()
            
            for box, class_id in zip(boxes, classes):
                class_name = r.names[int(class_id)]
                center = get_center_point(box)
                
                if class_name.lower() == "needle":
                    needle_corners = box.reshape(4, 2)
                elif class_name.isdigit() or class_name in ["0", "5", "10", "15", "20", "25", "30"] or class_name.lower() == "numbers":
                    number_positions.append((0, center))
    
    # Calculate meter reading using enhanced corner logic
    if needle_corners is not None and number_positions:
        reading, method = calculate_needle_corner_ratio_approximation(needle_corners, number_positions)
        
        if reading is not None:
            print(f"Meter Reading: {reading}")
        
    return image

# -----------------------------
# Main Integration
# -----------------------------

def main():
    # Paths for models and the input image
    analog_box_model = "Models/analog_box_v2.pt"
    analog_reading_model = "Models/analog_reading_v2.pt"
    full_image_path = "extracted_frames/265.png"
    
    # Step 1: Detect and crop the meter region
    cropped_meter = detect_and_crop_region(analog_box_model, full_image_path)
    if cropped_meter is None:
        return
        
    # Step 2: Process with needle corner coordinate logic
    processed_image = process_meter_reading_with_corner_logic(analog_reading_model, cropped_meter)

if __name__ == "__main__":
    main()