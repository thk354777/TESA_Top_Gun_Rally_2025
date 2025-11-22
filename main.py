import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from sklearn.ensemble import RandomForestRegressor
from joblib import load
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
import json
import io
import requests
import math
import os
from dotenv import load_dotenv

load_dotenv()

API_ENDPOINT = os.getenv("API_ENDPOINT", "")
CAMERA_ID = os.getenv("CAMERA_ID", "")
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "")

def _post_detection_event(cam_id, frame, objects_array, current_timestamp):
    if not objects_array:
        return
    success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if not success:
        print("ERROR: Could not encode frame to JPEG.")
        return
    image_bytes = io.BytesIO(buffer.tobytes())
    files = {
        'image': ('detection_frame.jpg', image_bytes, 'image/jpeg')
    }
    data = {
        'objects': json.dumps(objects_array),
        'timestamp': current_timestamp
    }
    headers = {"x-camera-token": AUTH_TOKEN}
    url = f"{API_ENDPOINT}{cam_id}"
    try:
        response = requests.post(url, headers=headers, data=data, files=files, timeout=10)
        response.raise_for_status()
        print(f"✅ Successfully posted {len(objects_array)} objects at {current_timestamp}. Status: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"⚠️ ERROR posting data to API: {e}")

def _interpolate_color(color_start, color_end, factor):
    b_start, g_start, r_start = color_start
    b_end, g_end, r_end = color_end
    b = int(b_start + (b_end - b_start) * factor)
    g = int(g_start + (g_end - g_start) * factor)
    r = int(r_start + (r_end - r_start) * factor)
    return (b, g, r)

def _get_color(track_id):
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
              (0, 255, 255), (255, 0, 255), (255, 255, 0)]
    try:
        track_num = int(track_id)
    except ValueError:
        track_num = 0
    return colors[track_num % len(colors)]

def _draw_annotations(frame, obj_id, lat, lon, alt, x1, y1, x2, y2, color, track_index):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale_main = 0.8
    font_scale_sub = 0.6
    font_thickness_main = 2
    font_thickness_sub = 1
    padding = 5
    block_height = 100
    y_offset_group = 30 + (track_index * block_height)
    lines = [
        f"ID: {obj_id}", f"Alt: {alt:.2f} m", f"Lat: {lat:.6f}", f"Lon: {lon:.6f}"
    ]
    current_y = y_offset_group
    for i, line in enumerate(lines):
        scale = font_scale_sub if i > 0 else font_scale_main
        thickness = font_thickness_sub if i > 0 else font_thickness_main
        (text_w, text_h), baseline = cv2.getTextSize(line, font, scale, thickness)
        current_y = current_y + text_h + padding
        cv2.putText(frame, line, (20, current_y), font, scale, color, thickness, cv2.LINE_AA)
    bbox_id_text = f"ID:{obj_id}"
    bbox_id_y = int(y1) - 10
    if bbox_id_y < 0: bbox_id_y = 0
    cv2.putText(frame, bbox_id_text, (int(x1), bbox_id_y), font, font_scale_main, color, font_thickness_main, cv2.LINE_AA)

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH
    areaA = max(1, (boxA[2] - boxA[0])) * max(1, (boxA[3] - boxA[1]))
    areaB = max(1, (boxB[2] - boxB[0])) * max(1, (boxB[3] - boxB[1]))
    return inter / float(areaA + areaB - inter + 1e-9)

def expand_box(box, frame_w, frame_h, margin=0.5):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    ex = int(max(1, w * margin))
    ey = int(max(1, h * margin))
    nx1 = max(0, int(x1 - ex))
    ny1 = max(0, int(y1 - ey))
    nx2 = min(frame_w - 1, int(x2 + ex))
    ny2 = min(frame_h - 1, int(y2 + ey))
    return [nx1, ny1, nx2, ny2]

def crop_safe(img, box):
    x1, y1, x2, y2 = box
    h, w = img.shape[:2]
    x1 = max(0, min(w-1, int(x1)))
    y1 = max(0, min(h-1, int(y1)))
    x2 = max(0, min(w, int(x2)))
    y2 = max(0, min(h, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]

def match_template(prev_crop, search_region):
    if prev_crop is None or search_region is None:
        return 0.0, (0,0)
    ph, pw = prev_crop.shape[:2]
    sh, sw = search_region.shape[:2]
    if ph > sh or pw > sw:
        scale = min(sh / ph, sw / pw)
        new_w = max(1, int(pw * scale))
        new_h = max(1, int(ph * scale))
        prev_resized = cv2.resize(prev_crop, (new_w, new_h))
        method = cv2.TM_CCOEFF_NORMED
        res = cv2.matchTemplate(search_region, prev_resized, method)
    else:
        method = cv2.TM_CCOEFF_NORMED
        res = cv2.matchTemplate(search_region, prev_crop, method)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(res)
    return maxVal, maxLoc

conf_h = 0.6
conf_l = 0.25
iou_th = 0.4
corr_th = 0.7
search_margin = 1.0
replace_with_prev_conf = True

model = YOLO("best (2).onnx")
rf_loaded = load("random_forest_model.pkl")
video_path = "VIDEOS/P3_VIDEO.mp4"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"Cannot open video file: {video_path}")

tracker = DeepSort(max_age=180)
tracks = {}
track_colors = {}
OUTPUT_SIZE = (400, 400)
FIXED_LINE_THICKNESS = 2
FADE_OUT_COLOR = (255, 255, 255)
window_name = "YOLO + DeepSort Tracking"
FRAME_SKIP = 3

prev_frame = None
prev_valid_detections = []
last_prev_crop_img = None
last_new_crop_img = None

try:
    while cap.isOpened():
        current_frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        success, frame = cap.read()
        if not success:
            break
        frame_h, frame_w = frame.shape[:2]
        current_timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        results = model.predict(frame, conf=conf_l, iou=0.45, max_det=500, agnostic_nms=True, verbose=False)
        boxes = results[0].boxes
        detections_for_tracker = []
        raw_detections = []

        if boxes is not None and boxes.xyxy is not None and len(boxes.xyxy) > 0:
            for i, box in enumerate(boxes.xyxy):
                x1, y1, x2, y2 = box.tolist()
                x1 = max(0, x1); y1 = max(0, y1); x2 = min(frame_w-1, x2); y2 = min(frame_h-1, y2)
                w = x2 - x1
                h = y2 - y1
                if w <= 0 or h <= 0:
                    continue
                conf = float(boxes.conf[i]) if boxes.conf is not None else 1.0
                detections_for_tracker.append([[x1, y1, w, h], conf])
                raw_detections.append({'bbox_tlbr': [x1, y1, x2, y2], 'conf': conf})

        if not prev_valid_detections:
            initial_valids = [d for d in raw_detections if d['conf'] > conf_h]
            if initial_valids:
                prev_valid_detections = []
                for d in initial_valids:
                    prev_valid_detections.append({'bbox': d['bbox_tlbr'], 'conf': d['conf']})
                prev_frame = frame.copy()
                print(f"[init] set {len(prev_valid_detections)} prev_valid_detections at frame {current_frame_pos}")

        detections_after_cc = []

        for d in raw_detections:
            bbox = d['bbox_tlbr']
            conf = d['conf']

            if conf >= conf_h:
                x1, y1, x2, y2 = bbox
                detections_after_cc.append([[x1, y1, x2 - x1, y2 - y1], conf])
                continue

            if conf_l <= conf < conf_h and prev_valid_detections:
                best_iou = 0.0
                best_prev = None
                for pv in prev_valid_detections:
                    i = iou(bbox, pv['bbox'])
                    if i > best_iou:
                        best_iou = i
                        best_prev = pv

                if best_prev and best_iou >= iou_th:
                    new_conf = best_prev['conf'] if replace_with_prev_conf else conf_h
                    x1, y1, x2, y2 = bbox
                    detections_after_cc.append([[x1, y1, x2 - x1, y2 - y1], float(new_conf)])
                    continue

                if best_prev is not None:
                    prev_box = best_prev['bbox']
                    exp_prev = expand_box(prev_box, frame_w, frame_h, margin=search_margin)
                    prev_crop = crop_safe(prev_frame, exp_prev) if prev_frame is not None else None
                    search_region = crop_safe(frame, exp_prev)

                    if prev_crop is not None and search_region is not None:
                        last_prev_crop_img = prev_crop.copy()
                        last_new_crop_img = search_region.copy()

                    corr_val, best_loc = match_template(prev_crop, search_region)
                    if prev_crop is not None and search_region is not None:
                        pred_x_in_search = best_loc[0]
                        pred_y_in_search = best_loc[1]
                        ex_x1, ex_y1, ex_x2, ex_y2 = exp_prev
                        ph, pw = prev_crop.shape[:2] if prev_crop is not None else (0,0)
                        sh, sw = search_region.shape[:2] if search_region is not None else (0,0)
                        rel_x1 = int(prev_box[0] - ex_x1)
                        rel_y1 = int(prev_box[1] - ex_y1)
                        rel_x2 = int(prev_box[2] - ex_x1)
                        rel_y2 = int(prev_box[3] - ex_y1)
                        predicted_x1 = ex_x1 + pred_x_in_search + rel_x1
                        predicted_y1 = ex_y1 + pred_y_in_search + rel_y1
                        predicted_x2 = predicted_x1 + (rel_x2 - rel_x1)
                        predicted_y2 = predicted_y1 + (rel_y2 - rel_y1)
                        predicted_box = [
                            max(0, min(frame_w-1, int(predicted_x1))),
                            max(0, min(frame_h-1, int(predicted_y1))),
                            max(0, min(frame_w, int(predicted_x2))),
                            max(0, min(frame_h, int(predicted_y2)))
                        ]
                        iou_val = iou(predicted_box, bbox)
                        if iou_val >= iou_th:
                            new_conf = best_prev['conf'] if replace_with_prev_conf else conf_h
                            x1, y1, x2, y2 = bbox
                            detections_after_cc.append([[x1, y1, x2 - x1, y2 - y1], float(new_conf)])
                            continue
                        elif corr_val >= corr_th:
                            new_conf = best_prev['conf'] if replace_with_prev_conf else conf_h
                            x1, y1, x2, y2 = predicted_box
                            detections_after_cc.append([[x1, y1, x2 - x1, y2 - y1], float(new_conf)])
                            continue
                        else:
                            continue
                    else:
                        continue
                else:
                    continue
            else:
                continue

        tracked_objects = tracker.update_tracks(detections_after_cc, frame=frame)

        annotated_frame = frame.copy()
        objects_to_post = []

        for i, obj in enumerate(tracked_objects):
            if not obj.is_confirmed():
                continue
            obj_id = obj.track_id
            x1, y1, x2, y2 = obj.to_tlbr()
            center_x, center_y = int((x1+x2)/2), int((y1+y2)/2)
            width, height = int(x2-x1), int(y2-y1)
            if width <= 0 or height <= 0:
                continue
            area = width * height
            if obj_id not in track_colors:
                track_colors[obj_id] = _get_color(obj_id)
            color = track_colors[obj_id]
            if obj_id not in tracks:
                tracks[obj_id] = []
            tracks[obj_id].append((center_x, center_y))
            X_input = pd.DataFrame([[center_x, center_y, area]], columns=['center_x', 'center_y', 'area'])
            y_pred = rf_loaded.predict(X_input)
            lat, lon, alt = y_pred[0]
            object_data = {
                "obj_id": str(obj_id),
                "type": "vehicle",
                "lat": float(lat),
                "lon": float(lon),
                "alt": float(alt),
                "objective": "tracking",
                "size": "medium",
                "bbox_tlbr": [x1, y1, x2, y2],
                "details": {"area": area, "confidence": 1.0}
            }
            objects_to_post.append(object_data)
            cv2.circle(annotated_frame, (center_x, center_y), 4, (0, 255, 255), -1)
            pts = tracks[obj_id]
            for j in range(1, len(pts)):
                fade_factor = j / len(pts)
                line_color = _interpolate_color(FADE_OUT_COLOR, color, fade_factor)
                cv2.line(annotated_frame, pts[j-1], pts[j], line_color, FIXED_LINE_THICKNESS)
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
            _draw_annotations(annotated_frame, obj_id, lat, lon, alt, x1, y1, x2, y2, color, i)

        if objects_to_post:
            _post_detection_event(CAMERA_ID, frame, objects_to_post, current_timestamp)

        cv2.imshow(window_name, annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        prev_valid_detections = []
        for det in detections_after_cc:
            bbox_tlbr = det[0]
            x, y, w, h = bbox_tlbr
            prev_valid_detections.append({'bbox': [x, y, x + w, y + h], 'conf': det[1]})
        if prev_valid_detections:
            prev_frame = frame.copy()
        next_frame_pos = current_frame_pos + FRAME_SKIP
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if next_frame_pos >= total_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame_pos)

except Exception as e:
    print(f"⚠️ An error occurred: {e}")

finally:
    if last_prev_crop_img is not None:
        cv2.imshow("last_prev_crop", last_prev_crop_img)
        cv2.imwrite("last_prev_crop.jpg", last_prev_crop_img)
        print("Saved last_prev_crop.jpg")
    else:
        print("No last_prev_crop available")

    if last_new_crop_img is not None:
        cv2.imshow("last_new_crop (search region)", last_new_crop_img)
        cv2.imwrite("last_new_crop.jpg", last_new_crop_img)
        print("Saved last_new_crop.jpg")
    else:
        print("No last_new_crop available")

    if (last_prev_crop_img is not None) or (last_new_crop_img is not None):
        print("Press any key in any image window to close...")
        cv2.waitKey(0)

    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
