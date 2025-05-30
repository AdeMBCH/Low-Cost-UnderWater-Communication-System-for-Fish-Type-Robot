import cv2
import numpy as np
from collections import deque

hsv_value = [100, 255, 255]

def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2

    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0.0

    return inter_area / union_area

def select_hsv_from_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"IMAGE PATH ERROR : {image_path}")

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    roi = cv2.selectROI("Selction ROI", img, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Selction ROI")

    x, y, w, h = roi
    if w == 0 or h == 0:
        raise ValueError("Empty ROI Selected")

    roi_hsv = hsv_img[y:y+h, x:x+w]
    mean_hsv = cv2.mean(roi_hsv)[:3]
    hsv_value = np.array(mean_hsv, dtype=np.uint8)
    print(f"HSV in average into ROI : {hsv_value}")

    tolerance = np.array([7, 30, 30])
    lower_bound = np.clip(hsv_value - tolerance, 0, 255)
    upper_bound = np.clip(hsv_value + tolerance, 0, 255)
    return lower_bound, upper_bound

class BlackRegionDetector:
    def __init__(self, min_area=500, smoothing=5):
        self.min_area = min_area
        self.smoothing = smoothing
        self.rect_history = deque(maxlen=smoothing)

    def preprocess(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        return closed

    def detect_largest_black_rectangle(self, image):
        mask = self.preprocess(image)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, mask

        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        if area < self.min_area:
            return None, mask

        x, y, w, h = cv2.boundingRect(largest_contour)

        self.rect_history.append((x, y, w, h))
        xs, ys, ws, hs = zip(*self.rect_history)
        x = int(np.mean(xs))
        y = int(np.mean(ys))
        w = int(np.mean(ws))
        h = int(np.mean(hs))

        return (x, y, w, h), mask

def detect_leds_hsv(image, lower_bound, upper_bound, roi_coords=None, max_leds=2, merge_dist_thresh=80):
    if roi_coords:
        x, y, w, h = roi_coords
        roi = image[y:y+h, x:x+w]
    else:
        roi = image

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    raw_boxes = []
    for c in contours:
        hull = cv2.convexHull(c)
        area = cv2.contourArea(hull)
        if 50 < area :
            x_, y_, w_, h_ = cv2.boundingRect(hull)
            raw_boxes.append((x_, y_, w_, h_))

    merged_boxes = []
    for box in raw_boxes:
        x1, y1, w1, h1 = box
        cx1, cy1 = x1 + w1 // 2, y1 + h1 // 2
        found = False
        for i, (mx, my, mw, mh) in enumerate(merged_boxes):
            cx2, cy2 = mx + mw // 2, my + mh // 2
            dist = np.hypot(cx1 - cx2, cy1 - cy2)
            if dist < merge_dist_thresh:
                new_x = min(x1, mx)
                new_y = min(y1, my)
                new_w = max(x1 + w1, mx + mw) - new_x
                new_h = max(y1 + h1, my + mh) - new_y
                merged_boxes[i] = (new_x, new_y, new_w, new_h)
                found = True
                break
        if not found:
            merged_boxes.append(box)

    merged_boxes = sorted(merged_boxes, key=lambda b: b[1])[:max_leds]

    out_img = image.copy()
    for (x_, y_, w_, h_) in merged_boxes:
        if roi_coords:
            x_, y_ = x_ + roi_coords[0], y_ + roi_coords[1]
        cv2.rectangle(out_img, (x_, y_), (x_ + w_, y_ + h_), (0, 0, 255), 2)
        cx, cy = x_ + w_ // 2, y_ + h_ // 2
        cv2.circle(out_img, (cx, cy), 3, (0, 255, 0), -1)

    if roi_coords:
        full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        full_mask[y:y+h, x:x+w] = mask
    else:
        full_mask = mask

    return out_img, full_mask

def process_video(video_path, ref_img_path, out_annotated_path, out_mask_path):
    lower, upper = select_hsv_from_image(ref_img_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error video opening")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(3)), int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out_annotated = cv2.VideoWriter(out_annotated_path, fourcc, fps, (width, height))
    out_mask = cv2.VideoWriter(out_mask_path, fourcc, fps, (width, height))

    detector = BlackRegionDetector(min_area=500)
    frame_idx = 0
    led_frames_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        roi_rect, _ = detector.detect_largest_black_rectangle(frame)
        annotated, mask = detect_leds_hsv(frame, lower, upper, roi_coords=roi_rect)

        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        out_annotated.write(annotated)
        out_mask.write(mask_bgr)

        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"Frame {frame_idx}")

    cap.release()
    out_annotated.release()
    out_mask.release()
    print("END")

if __name__ == "__main__":
    ref_image = "Untitled 2.png"
    input_video = ".mp4"
    output_annotated = "output_opencv.mp4"
    output_mask = "output_maskopencv.mp4"

    process_video(input_video, ref_image, output_annotated, output_mask)