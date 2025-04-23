#python detect_with_distance.py --model yolo11x.pt --calib calibration.npz  --height 1.63 --cam 0

#!/usr/bin/env python3
import cv2
import numpy as np
import torch
import argparse
from ultralytics import YOLO

def load_calibration(path: str):
    data = np.load(path)
    return data['camera_matrix'], data['dist_coeffs']

def undistort_frame(frame, camera_matrix, dist_coeffs):
    h, w = frame.shape[:2]
    new_cam_mtx, _ = cv2.getOptimalNewCameraMatrix(camera_matrix,
                                                   dist_coeffs,
                                                   (w, h),
                                                   1, (w, h))
    return cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_cam_mtx)

def detect_and_annotate(frame, model, camera_matrix, real_height_m,
                        device, conf, iou, max_det):
    # 1) Undistort
    undist = undistort_frame(frame, camera_matrix, dist_coeffs)

    # 2) Inference
    results = model(undist, device=device,
                    conf=conf, iou=iou, max_det=max_det)[0]

    fx = camera_matrix[0, 0]  # focal length in px (x axis)
    annotated = undist.copy()

    for box in results.boxes:
        # box coords
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        pw, ph = x2 - x1, y2 - y1

        # estimate distance: D = (H_real * f_px) / pixel_height
        if ph > 0:
            dist_m = (real_height_m * fx) / ph
            dist_text = f"{dist_m:.2f}m"
        else:
            dist_text = "∞"

        # draw box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # prepare text: "<class> WxHpx, D"
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        text  = f"{label} {pw}x{ph}px, {dist_text}"

        # text background
        (tw, th), _ = cv2.getTextSize(text,
                                      cv2.FONT_HERSHEY_SIMPLEX,
                                      0.5, 1)
        cv2.rectangle(
            annotated,
            (x1, y1 - th - 4),
            (x1 + tw, y1),
            (0, 255, 0),
            thickness=cv2.FILLED
        )
        # text itself
        cv2.putText(
            annotated,
            text,
            (x1, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )

    return annotated

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="YOLOv8 live detection with distance estimation"
    )
    p.add_argument('--model', '-m', default='yolov8x.pt',
                   help="Path to YOLOv8 .pt model")
    p.add_argument('--calib', '-c', default='calibration.npz',
                   help="Path to calibration file")
    p.add_argument('--height', '-H', type=float, default=1.7,
                   help="Real object height in metres (for all detections)")
    p.add_argument('--cam', '-C', type=int, default=0,
                   help="Camera index")
    p.add_argument('--conf', type=float, default=0.25,
                   help="Confidence threshold")
    p.add_argument('--iou', type=float, default=0.45,
                   help="NMS IOU threshold")
    p.add_argument('--maxdet', type=int, default=100,
                   help="Max detections per frame")
    args = p.parse_args()

    # load calibration
    camera_matrix, dist_coeffs = load_calibration(args.calib)

    # choose device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on device: {device}")

    # load YOLO
    model = YOLO(args.model)

    # open webcam
    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.cam}")

    print("▶️ Press Q to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated = detect_and_annotate(frame, model,
                                        camera_matrix,
                                        args.height,
                                        device,
                                        args.conf,
                                        args.iou,
                                        args.maxdet)
        cv2.imshow('YOLO + Distance', annotated)

        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
            break

    cap.release()
    cv2.destroyAllWindows()
