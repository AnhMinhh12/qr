import cv2
import os
import sys
import numpy as np
from datetime import datetime
from ultralytics import YOLO

# --- CONFIGURATION CONSTANTS ---
# Sử dụng chính Model 'best.pt' bạn vừa huấn luyện xong
YOLO_MODEL_PATH = "best.pt" 
CONF_THRESHOLD = 0.25      # Model bạn train 1.0 recall nên 0.25 là cực kỳ an toàn
CROP_PADDING = 20
BBOX_THICKNESS = 7         # Tăng đáng kể (từ 2 lên 6)
FAIL_BBOX_THICKNESS = 14    # Tăng độ dày lỗi lên 9
OVERLAY_ALPHA = 0.2
IMGSZ = 640 # Model được train ở 640 nên chạy inference 640 là chuẩn nhất

# Colors (BGR)
COLOR_OK = (0, 255, 0)      # Bright Green
COLOR_FAIL = (0, 0, 255)    # Bright Red
COLOR_TEXT = (255, 255, 255) # White
COLOR_STATUS_BG = (20, 20, 20) # Almost black

def load_models(model_dir: str = "wechat_model"):
    """
    Load YOLO Custom (Huấn luyện riêng) và WeChat Decode.
    """
    print(f"Loading custom Pallet QR Model: {YOLO_MODEL_PATH}...")
    if not os.path.exists(YOLO_MODEL_PATH):
        raise FileNotFoundError(f"Không tìm thấy file {YOLO_MODEL_PATH} trong thư mục dự án!")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    
    print("Loading WeChat QR Decoder...")
    detect_proto = os.path.join(model_dir, "detect.prototxt")
    detect_model = os.path.join(model_dir, "detect.caffemodel")
    sr_proto = os.path.join(model_dir, "sr.prototxt")
    sr_model = os.path.join(model_dir, "sr.caffemodel")
    
    try:
        decoder_wechat = cv2.wechat_qrcode.WeChatQRCode(
            detect_proto, detect_model, sr_proto, sr_model
        )
    except Exception as e:
        raise RuntimeError(f"Could not initialize WeChatQRCode: {e}")

    return yolo_model, decoder_wechat

def preprocess_crop(img: np.ndarray, bbox: list, padding: int = CROP_PADDING):
    """Crop and enhance area for decoding."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1-padding), max(0, y1-padding)
    x2, y2 = min(w, x2+padding), min(h, y2+padding)
    
    crop = img[y1:y2, x1:x2]
    if crop.size == 0: return None
        
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    return clahe.apply(gray)

def process_pallet_image(img_path: str):
    """Full pipeline using Custom YOLO Model."""
    if not os.path.exists(img_path): raise FileNotFoundError(img_path)
    img = cv2.imread(img_path)
    yolo_model, decoder = load_models()
    
    # 1. Nhận diện QR bằng Model "Nội bộ"
    results = yolo_model(img, conf=CONF_THRESHOLD, imgsz=IMGSZ)[0]
    bboxes = results.boxes.xyxy.cpu().numpy()
    confidences = results.boxes.conf.cpu().numpy()
    
    print(f"Total QR candidates found: {len(bboxes)}")
    
    ok_count, fail_count = 0, 0
    img_h, img_w = img.shape[:2]
    
    for i, bbox in enumerate(bboxes):
        conf = confidences[i]
        print(f"Candidate {i+1}: conf={conf:.2f}")

        # Decoding
        crop = preprocess_crop(img, bbox)
        try:
            res_list, points = decoder.detectAndDecode(crop)
            qr_data = res_list[0] if res_list and len(res_list) > 0 and res_list[0] != "" else None
        except:
            qr_data = None
        
        x1, y1, x2, y2 = map(int, bbox)
        if qr_data:
            print(f"   -> Success: {qr_data[:15]}...")
            ok_count += 1
            cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_OK, BBOX_THICKNESS)
            cv2.putText(img, "OK", (x1, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_OK, 3)
        else:
            print("   -> Failed to decode.")
            fail_count += 1
            cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_FAIL, FAIL_BBOX_THICKNESS)
            overlay = img.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_FAIL, -1)
            cv2.addWeighted(overlay, OVERLAY_ALPHA, img, 1-OVERLAY_ALPHA, 0, img)
            cv2.putText(img, "NG", (x1, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_FAIL, 3)
            
    # Summary Bar
    res_status = "PASS" if fail_count == 0 and ok_count > 0 else "FAIL"
    summary = f"{res_status}: {ok_count} OK, {fail_count} FAILED"
    bar_color = COLOR_OK if res_status == "PASS" else COLOR_FAIL
    
    cv2.rectangle(img, (0, 0), (img.shape[1], 70), (40, 40, 40), -1)
    cv2.putText(img, summary, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, bar_color, 4)
    
    out_path = img_path.replace(".jpg", "_result_final.jpg").replace(".png", "_result_final.png")
    cv2.imwrite(out_path, img)
    return {"status": res_status, "ok": ok_count, "fail": fail_count, "path": out_path}

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "vid_IMG_3256_f000020.jpg"
    print(f"Inspecting Pallet with Custom AI: {target}...")
    try:
        r = process_pallet_image(target)
        print(f"\n--- REPORT: {r['status']} ---")
        print(f"Found: {r['ok']} valid, {r['fail']} unreadable.")
        print(f"Result Image: {r['path']}\n")
    except Exception as e:
        print(f"Error: {e}")
