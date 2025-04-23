import cv2
import torch
from ultralytics import YOLO
import argparse

import cv2
import numpy as np
from ultralytics import YOLO

def annotate_with_sizes(frame: np.ndarray, results, model: YOLO,
                        box_color=(0,255,0),
                        font=cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale=0.5,
                        font_thickness=1) -> np.ndarray:
    """
    Draws YOLOv8 detections on `frame` and labels each box with "<class> W×H px".

    Args:
        frame:      Original BGR image.
        results:    The Ultralytics results object from model(frame,…).
        model:      The YOLO model instance (for model.names lookup).
        box_color:  RGB tuple for box & label background.
        font:       cv2 font face.
        font_scale: Text scale factor.
        font_thickness: Text stroke thickness.

    Returns:
        Annotated image (copy of `frame`).
    """
    annotated = frame.copy()
    res = results[0]  # single image batch

    for box in res.boxes:
        # coords
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        pw, ph = x2 - x1, y2 - y1

        # draw box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

        # text = "label W×Hpx"
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        text = f"{label} {int(pw)}x{int(ph)}px"


        # text size
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        # background
        cv2.rectangle(
            annotated,
            (x1, y1 - th - 4),
            (x1 + tw, y1),
            box_color,
            thickness=cv2.FILLED
        )
        # text
        cv2.putText(
            annotated,
            text,
            (x1, y1 - 2),
            font,
            font_scale,
            (0,0,0),
            font_thickness,
            cv2.LINE_AA
        )

    return annotated


def detect_live_webcam(model_path: str='yolo11x.pt', cam_index: int=0):
    """
    Run YOLOv8 on webcam frames, using CUDA if available, and display live detections.
    """
    # On Windows with CUDA: torch.cuda.is_available()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on device: {device}")

    # Load model (no device arg here)
    model = YOLO(model_path)
    # 1) Load calibration
    calib = np.load('calibration.npz')
    camera_matrix = calib['camera_matrix']
    dist_coeffs    = calib['dist_coeffs']
    # Open the webcam
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)  # CAP_DSHOW often works better on Windows
    if not cap.isOpened():
        print(f"❌ Unable to open camera #{cam_index}")
        return

    print("▶️ Press Q to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame")
            break

        # Inference—pass device here
        results = model(frame, device=device, conf=0.25, iou=0.45, max_det=100)
        # res = results[0]

        # for box in res.boxes:
        #     cls_id = int(box.cls[0])
        #     label  = model.names[cls_id]
        #     x1, y1, x2, y2 = box.xyxy[0]  # float coords
        #     pw = x2 - x1                  # pixel width
        #     ph = y2 - y1                  # pixel height
        #     print(f"{label:10s}  size: {pw:.0f}px × {ph:.0f}px  (confidence {box.conf[0]:.2f})")

        # Annotate and display
        # annotated = results[0].plot()
        # cv2.imshow('YOLOv11 Live (CUDA)' if device=='cuda' else 'YOLOv11 Live (CPU)', annotated)
        annotated = annotate_with_sizes(frame, results, model)
        cv2.imshow('YOLOv8 Live w/ Sizes', annotated)

        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YOLOv8 Live Webcam Detection (CUDA/CPU on Windows)"
    )
    parser.add_argument(
        '--model', '-m',
        default='yolo11x.pt',
        help="Path to YOLOv11 model (.pt). Defaults to 'yolo11x.pt'."
    )
    parser.add_argument(
        '--cam', '-c',
        type=int,
        default=0,
        help="Camera index (default 0)."
    )
    args = parser.parse_args()
    detect_live_webcam(args.model, args.cam)
